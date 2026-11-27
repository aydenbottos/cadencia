#!/usr/bin/env python3
"""
Convert genuine NVIDIA MagpieTTS Multilingual 357M v2607 (July 2026, GRPO,
frame_stacking_factor=2, 2-layer 12-head 768-dim LocalTransformer, 217-frame
speaker context, attention prior) directly from NVIDIA's public FP16 GGUF
(`nvidia/magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.v2607.f16.gguf`)
into Cadencia's on-device Core ML layout — ZERO NeMo install required!

Outputs (`--out <dir>`):
  - text_encoder.mlpackage       (FP16 ANE, vocab=3359, 6 causal layers)
  - decoder_prefill.mlpackage    (FP16 ANE, T_ctx=217, 12 layers -> 24 rank-4 KV caches)
  - decoder_step.mlpackage       (FP16 GPU/ANE, attention prior + alignment scores + 32384 logits)
  - constants/
      constants.json
      speaker_{0..4}.npy         (FP16 [217, 768]: Aria, Jason, John, Leo, Sofia)
      audio_embedding_{0..15}.npy (FP16 [2024, 768], 8 codebooks x 2 stacked frames)
      local_transformer/
        pos_emb.npy              (FP16 [18, 768])
        layer_{0,1}_{norm1_weight,sa_qkv_weight,sa_o_weight,norm2_weight,ffn_conv1_weight,ffn_conv2_weight}.npy
        out_proj_{0..15}_{weight,bias}.npy
  - tokenizer/
      spanish_phoneme_token2id.json
      spanish_phoneme_phoneme_dict.json

NanoCodec (`nanocodec_decoder_v3.mlmodelc`, full FP32 CPU) is identical between
v2512/v2602 and v2607 (`nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V2607_GGUF_URL = (
    "https://huggingface.co/nvidia/magpie_tts_multilingual_357m/resolve/main/"
    "magpie_tts_multilingual_357m.v2607.f16.gguf"
)
SPANISH_TOKEN2ID_URL = (
    "https://huggingface.co/FluidInference/magpie-tts-multilingual-357m-coreml/resolve/main/"
    "tokenizer/spanish_phoneme_token2id.json"
)
SPANISH_DICT_URL = (
    "https://huggingface.co/FluidInference/magpie-tts-multilingual-357m-coreml/resolve/main/"
    "tokenizer/spanish_phoneme_phoneme_dict.json"
)

MASK_NEG = -3.0e4


# MARK: - Zero-Dependency GGUF v3 Reader

class GGUFReader:
    """Minimal, exact GGUF v3 reader supporting F32 (0), F16 (1), and I32 (26)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.kvs: Dict[str, Any] = {}
        self.tensors: Dict[str, Tuple[List[int], int, int]] = {}
        self.data_offset: int = 0
        self._parse()

    def _read_u32(self, pos: int) -> Tuple[int, int]:
        return struct.unpack_from("<I", self.data, pos)[0], pos + 4

    def _read_u64(self, pos: int) -> Tuple[int, int]:
        return struct.unpack_from("<Q", self.data, pos)[0], pos + 8

    def _read_str(self, pos: int) -> Tuple[str, int]:
        length, pos = self._read_u64(pos)
        s = bytes(self.data[pos : pos + length]).decode("utf-8", errors="replace")
        return s, pos + length

    def _read_val(self, pos: int, vtype: int) -> Tuple[Any, int]:
        if vtype == 0:
            return int(self.data[pos]), pos + 1
        if vtype == 1:
            return struct.unpack_from("<b", self.data, pos)[0], pos + 1
        if vtype == 2:
            return struct.unpack_from("<H", self.data, pos)[0], pos + 2
        if vtype == 3:
            return struct.unpack_from("<h", self.data, pos)[0], pos + 2
        if vtype == 4:
            return struct.unpack_from("<I", self.data, pos)[0], pos + 4
        if vtype == 5:
            return struct.unpack_from("<i", self.data, pos)[0], pos + 4
        if vtype == 6:
            return struct.unpack_from("<f", self.data, pos)[0], pos + 4
        if vtype == 7:
            return bool(self.data[pos]), pos + 1
        if vtype == 8:
            return self._read_str(pos)
        if vtype == 9:
            etype, alen = struct.unpack_from("<IQ", self.data, pos)
            pos += 12
            arr = []
            for _ in range(alen):
                v, pos = self._read_val(pos, etype)
                arr.append(v)
            return arr, pos
        if vtype == 10:
            return struct.unpack_from("<Q", self.data, pos)[0], pos + 8
        if vtype == 11:
            return struct.unpack_from("<q", self.data, pos)[0], pos + 8
        if vtype == 12:
            return struct.unpack_from("<d", self.data, pos)[0], pos + 8
        raise ValueError(f"Unsupported GGUF metadata type {vtype}")

    def _parse(self) -> None:
        magic = bytes(self.data[:4])
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic: {magic!r}")
        ver, n_tensors, n_kv = struct.unpack_from("<IQQ", self.data, 4)
        if ver < 2:
            raise ValueError(f"Unsupported GGUF version: {ver}")

        pos = 24
        for _ in range(n_kv):
            key, pos = self._read_str(pos)
            vtype, pos = self._read_u32(pos)
            val, pos = self._read_val(pos, vtype)
            self.kvs[key] = val

        for _ in range(n_tensors):
            name, pos = self._read_str(pos)
            ndim, pos = self._read_u32(pos)
            dims = list(struct.unpack_from(f"<{ndim}Q", self.data, pos))
            pos += 8 * ndim
            ttype, pos = self._read_u32(pos)
            offset, pos = self._read_u64(pos)
            self.tensors[name] = (dims, ttype, offset)

        alignment = int(self.kvs.get("general.alignment", 32))
        self.data_offset = ((pos + alignment - 1) // alignment) * alignment

    def get_numpy(self, name: str) -> np.ndarray:
        dims, ttype, offset = self.tensors[name]
        # GGUF stores dims reversed relative to C-order PyTorch shape
        shape = tuple(reversed(dims)) if dims else ()
        dtype_map = {0: np.float32, 1: np.float16, 26: np.int32}
        if ttype not in dtype_map:
            raise ValueError(f"Unsupported GGML tensor dtype {ttype} for {name}")
        dtype = dtype_map[ttype]
        count = int(np.prod(shape)) if shape else 1
        nbytes = count * np.dtype(dtype).itemsize
        start = self.data_offset + offset
        raw = self.data[start : start + nbytes]
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    def get_torch_f32(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.get_numpy(name).astype(np.float32))


# MARK: - Traceable PyTorch Modules (Exact NVIDIA Magpie v2607 Architecture)

class CausalEncoderSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
        attn = attn + (1.0 - causal.unsqueeze(0).unsqueeze(0)) * MASK_NEG
        m = mask.to(x.dtype).unsqueeze(1).unsqueeze(2)
        attn = attn + (1.0 - m) * MASK_NEG
        attn = F.softmax(attn, dim=-1) * m
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class CausalConvFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(d_model, d_ffn, kernel_size, padding=0, bias=False)
        self.conv2 = nn.Conv1d(d_ffn, d_model, kernel_size, padding=0, bias=False)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        if mask is not None:
            m = mask.to(x.dtype).unsqueeze(1)
            x = x * m
        if self.kernel_size > 1:
            x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.act(self.conv1(x))
        if mask is not None:
            x = x * m
        if self.kernel_size > 1:
            x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv2(x)
        if mask is not None:
            x = x * m
        return x.transpose(1, 2)


class V2607EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, n_heads: int, kernel_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, bias=False)
        self.self_attn = CausalEncoderSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model, bias=False)
        self.ffn = CausalConvFFN(d_model, d_ffn, kernel_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x), mask)
        return x


class V2607TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 3359,
        n_layers: int = 6,
        d_model: int = 768,
        d_ffn: int = 3072,
        n_heads: int = 12,
        kernel_size: int = 3,
        max_pos: int = 2048,
    ) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_pos, d_model)
        self.layers = nn.ModuleList(
            [V2607EncoderLayer(d_model, d_ffn, n_heads, kernel_size) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model, bias=False)

    def forward(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        x = self.text_embedding(text_tokens)
        T = x.shape[1]
        positions = torch.arange(T, device=x.device)
        x = (x + self.position_embeddings(positions)) * text_mask.unsqueeze(-1).to(x.dtype)
        for layer in self.layers:
            x = layer(x, text_mask)
        return self.norm_out(x)

    @classmethod
    def from_gguf(cls, reader: GGUFReader) -> "V2607TextEncoder":
        m = cls(
            vocab_size=int(reader.kvs["magpietts.text_vocab_size"]),
            n_layers=int(reader.kvs["magpietts.encoder.layers"]),
            d_model=int(reader.kvs["magpietts.embedding_dim"]),
            d_ffn=int(reader.kvs["magpietts.ffn_dim"]),
            n_heads=int(reader.kvs["magpietts.encoder.heads"]),
            kernel_size=int(reader.kvs["magpietts.encoder.kernel_size"]),
            max_pos=int(reader.kvs["magpietts.context_length"]),
        )
        m.text_embedding.weight.data.copy_(reader.get_torch_f32("text_embedding.weight"))
        m.position_embeddings.weight.data.copy_(reader.get_torch_f32("encoder.position_embeddings.weight"))
        m.norm_out.weight.data.copy_(reader.get_torch_f32("encoder.norm_out.weight"))
        K = int(reader.kvs["magpietts.encoder.kernel_size"])
        for i, layer in enumerate(m.layers):
            p = f"encoder.layers.{i}."
            layer.norm1.weight.data.copy_(reader.get_torch_f32(p + "norm_self.weight"))
            layer.self_attn.qkv_proj.weight.data.copy_(reader.get_torch_f32(p + "self_attention.qkv_net.weight"))
            layer.self_attn.o_proj.weight.data.copy_(reader.get_torch_f32(p + "self_attention.o_net.weight"))
            layer.norm2.weight.data.copy_(reader.get_torch_f32(p + "norm_pos_ff.weight"))
            c1 = torch.stack([reader.get_torch_f32(f"{p}pos_ff.proj.conv.weight.k{k}") for k in range(K)], dim=2)
            c2 = torch.stack([reader.get_torch_f32(f"{p}pos_ff.o_net.conv.weight.k{k}") for k in range(K)], dim=2)
            layer.ffn.conv1.weight.data.copy_(c1)
            layer.ffn.conv2.weight.data.copy_(c2)
        return m


class PrefillCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)) * self.scale
        causal = torch.tril(torch.ones(T, T, dtype=x.dtype, device=x.device))
        attn = F.softmax(attn + (1.0 - causal) * MASK_NEG, dim=-1)
        out = torch.matmul(attn, v.transpose(1, 2)).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), k, v


class DecoderCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * n_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T_q, _ = x.shape
        T_m = memory.shape[1]
        q = self.q_proj(x).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(memory).view(B, T_m, 2, self.n_heads, self.d_head)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + attn_bias.to(x.dtype).unsqueeze(1).unsqueeze(2)
        probs = F.softmax(scores, dim=-1)  # (B, H_x, T_q, T_m)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(B, T_q, -1)
        return self.o_proj(out), probs


class V2607PrefillLayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, sa_heads: int, xa_heads: int, xa_d_head: int) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model, bias=False)
        self.self_attn = PrefillCausalSelfAttention(d_model, sa_heads)
        self.norm_xa_q = nn.LayerNorm(d_model, bias=False)
        self.norm_xa_m = nn.LayerNorm(d_model, bias=False)
        self.cross_attn = DecoderCrossAttention(d_model, xa_heads, xa_d_head)
        self.norm_ff = nn.LayerNorm(d_model, bias=False)
        self.ffn = CausalConvFFN(d_model, d_ffn, kernel_size=1)

    def forward(
        self, x: torch.Tensor, enc_out: torch.Tensor, mask_bias: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sa_out, k, v = self.self_attn(self.norm_sa(x))
        x = x + sa_out
        xa_out, _ = self.cross_attn(self.norm_xa_q(x), self.norm_xa_m(enc_out), mask_bias)
        x = x + xa_out
        x = x + self.ffn(self.norm_ff(x))
        return x, k, v


class V2607DecoderPrefill(nn.Module):
    """Batched 217-frame speaker context prefill returning 24 rank-4 KV caches (k0..k11, v0..v11)."""

    def __init__(
        self,
        n_layers: int = 12,
        d_model: int = 768,
        d_ffn: int = 3072,
        sa_heads: int = 12,
        xa_heads: int = 1,
        xa_d_head: int = 128,
        t_ctx: int = 217,
        max_seq_len: int = 512,
        max_pos: int = 2048,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.t_ctx = t_ctx
        self.max_seq_len = max_seq_len
        self.position_embeddings = nn.Embedding(max_pos, d_model)
        self.layers = nn.ModuleList(
            [V2607PrefillLayer(d_model, d_ffn, sa_heads, xa_heads, xa_d_head) for _ in range(n_layers)]
        )

    def forward(
        self, audio_embed: torch.Tensor, encoder_output: torch.Tensor, encoder_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        pad_len = self.max_seq_len - self.t_ctx
        pos_ids = torch.arange(self.t_ctx, device=audio_embed.device).unsqueeze(0)
        x = audio_embed + self.position_embeddings(pos_ids)
        mask_bias = (1.0 - encoder_mask.to(x.dtype)) * MASK_NEG
        ks: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for layer in self.layers:
            x, k, v = layer(x, encoder_output, mask_bias)
            ks.append(F.pad(k, (0, 0, 0, 0, 0, pad_len)))
            vs.append(F.pad(v, (0, 0, 0, 0, 0, pad_len)))
        return tuple(ks + vs)

    @classmethod
    def from_gguf(cls, reader: GGUFReader, max_seq_len: int = 512) -> "V2607DecoderPrefill":
        m = cls(
            n_layers=int(reader.kvs["magpietts.decoder.layers"]),
            d_model=int(reader.kvs["magpietts.embedding_dim"]),
            d_ffn=int(reader.kvs["magpietts.ffn_dim"]),
            sa_heads=int(reader.kvs["magpietts.decoder.heads"]),
            xa_heads=int(reader.kvs["magpietts.decoder.cross_heads"]),
            xa_d_head=int(reader.kvs["magpietts.decoder.cross_head_dim"]),
            t_ctx=int(reader.kvs["magpietts.baked_context_length"]),
            max_seq_len=max_seq_len,
            max_pos=int(reader.kvs["magpietts.context_length"]),
        )
        m.position_embeddings.weight.data.copy_(reader.get_torch_f32("decoder.position_embeddings.weight"))
        for i, layer in enumerate(m.layers):
            p = f"decoder.layers.{i}."
            layer.norm_sa.weight.data.copy_(reader.get_torch_f32(p + "norm_self.weight"))
            layer.self_attn.qkv_proj.weight.data.copy_(reader.get_torch_f32(p + "self_attention.qkv_net.weight"))
            layer.self_attn.o_proj.weight.data.copy_(reader.get_torch_f32(p + "self_attention.o_net.weight"))
            layer.norm_xa_q.weight.data.copy_(reader.get_torch_f32(p + "norm_xattn_query.weight"))
            layer.norm_xa_m.weight.data.copy_(reader.get_torch_f32(p + "norm_xattn_memory.weight"))
            layer.cross_attn.q_proj.weight.data.copy_(reader.get_torch_f32(p + "cross_attention.q_net.weight"))
            layer.cross_attn.kv_proj.weight.data.copy_(reader.get_torch_f32(p + "cross_attention.kv_net.weight"))
            layer.cross_attn.o_proj.weight.data.copy_(reader.get_torch_f32(p + "cross_attention.o_net.weight"))
            layer.norm_ff.weight.data.copy_(reader.get_torch_f32(p + "norm_pos_ff.weight"))
            layer.ffn.conv1.weight.data.copy_(reader.get_torch_f32(p + "pos_ff.proj.conv.weight.k0").unsqueeze(-1))
            layer.ffn.conv2.weight.data.copy_(reader.get_torch_f32(p + "pos_ff.o_net.conv.weight.k0").unsqueeze(-1))
        return m


class StepCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, kv_k: torch.Tensor, kv_v: torch.Tensor, position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        max_seq = kv_k.shape[1]
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        positions_range = torch.arange(max_seq, dtype=x.dtype, device=x.device)
        pos_cast = position.to(x.dtype)
        write_mask = (positions_range == pos_cast).to(x.dtype).view(1, max_seq, 1, 1)
        new_k = kv_k * (1.0 - write_mask) + k.expand(B, max_seq, self.n_heads, self.d_head) * write_mask
        new_v = kv_v * (1.0 - write_mask) + v.expand(B, max_seq, self.n_heads, self.d_head) * write_mask

        causal_mask = (positions_range <= pos_cast).to(x.dtype).view(1, 1, 1, max_seq)
        q4 = q.transpose(1, 2)
        k4 = new_k.permute(0, 2, 1, 3)
        v4 = new_v.permute(0, 2, 1, 3)
        attn = torch.matmul(q4, k4.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn + (1.0 - causal_mask) * MASK_NEG, dim=-1)
        out = torch.matmul(attn, v4).transpose(1, 2).reshape(B, 1, -1)
        return self.o_proj(out), new_k, new_v


class V2607StepLayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, sa_heads: int, xa_heads: int, xa_d_head: int) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model, bias=False)
        self.self_attn = StepCausalSelfAttention(d_model, sa_heads)
        self.norm_xa_q = nn.LayerNorm(d_model, bias=False)
        self.norm_xa_m = nn.LayerNorm(d_model, bias=False)
        self.cross_attn = DecoderCrossAttention(d_model, xa_heads, xa_d_head)
        self.norm_ff = nn.LayerNorm(d_model, bias=False)
        self.ffn = CausalConvFFN(d_model, d_ffn, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        kv_k: torch.Tensor,
        kv_v: torch.Tensor,
        position: torch.Tensor,
        enc_out: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sa_out, new_k, new_v = self.self_attn(self.norm_sa(x), kv_k, kv_v, position)
        x = x + sa_out
        xa_out, probs = self.cross_attn(self.norm_xa_q(x), self.norm_xa_m(enc_out), attn_bias)
        x = x + xa_out
        x = x + self.ffn(self.norm_ff(x))
        return x, new_k, new_v, probs


class V2607DecoderStep(nn.Module):
    """Single AR decoder step (generates 2 stacked frames) with full NeMo attention prior & alignment output."""

    def __init__(
        self,
        n_layers: int = 12,
        d_model: int = 768,
        d_ffn: int = 3072,
        sa_heads: int = 12,
        xa_heads: int = 1,
        xa_d_head: int = 128,
        stacked_codebooks: int = 16,
        audio_vocab: int = 2024,
        max_pos: int = 2048,
        apply_prior_to_layers: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
        estimate_alignment_from_layers: Tuple[int, ...] = (4, 5, 8, 9),
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.apply_prior_set = set(apply_prior_to_layers)
        self.alignment_set = set(estimate_alignment_from_layers)
        self.position_embeddings = nn.Embedding(max_pos, d_model)
        self.layers = nn.ModuleList(
            [V2607StepLayer(d_model, d_ffn, sa_heads, xa_heads, xa_d_head) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model, bias=False)
        self.final_proj = nn.Linear(d_model, stacked_codebooks * audio_vocab)

    def forward(
        self,
        audio_embed: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        attn_prior_bias: torch.Tensor,
        position: torch.Tensor,
        ck0: torch.Tensor, ck1: torch.Tensor, ck2: torch.Tensor, ck3: torch.Tensor,
        ck4: torch.Tensor, ck5: torch.Tensor, ck6: torch.Tensor, ck7: torch.Tensor,
        ck8: torch.Tensor, ck9: torch.Tensor, ck10: torch.Tensor, ck11: torch.Tensor,
        cv0: torch.Tensor, cv1: torch.Tensor, cv2: torch.Tensor, cv3: torch.Tensor,
        cv4: torch.Tensor, cv5: torch.Tensor, cv6: torch.Tensor, cv7: torch.Tensor,
        cv8: torch.Tensor, cv9: torch.Tensor, cv10: torch.Tensor, cv11: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        cks = [ck0, ck1, ck2, ck3, ck4, ck5, ck6, ck7, ck8, ck9, ck10, ck11]
        cvs = [cv0, cv1, cv2, cv3, cv4, cv5, cv6, cv7, cv8, cv9, cv10, cv11]

        pos_idx = position.to(torch.long)
        x = audio_embed + self.position_embeddings(pos_idx).unsqueeze(0)
        mask_bias = (1.0 - encoder_mask.to(x.dtype)) * MASK_NEG

        new_ks: List[torch.Tensor] = []
        new_vs: List[torch.Tensor] = []
        align_accum: torch.Tensor | None = None
        align_count = 0

        for i, layer in enumerate(self.layers):
            bias = attn_prior_bias if i in self.apply_prior_set else mask_bias
            x, nk, nv, probs = layer(x, cks[i], cvs[i], position, encoder_output, bias)
            new_ks.append(nk)
            new_vs.append(nv)
            if i in self.alignment_set:
                # probs: (1, H_x, 1, T_enc) -> mean over heads -> (1, T_enc)
                layer_align = probs.mean(dim=1).squeeze(1)
                align_accum = layer_align if align_accum is None else (align_accum + layer_align)
                align_count += 1

        assert align_accum is not None
        alignment_scores = align_accum / float(align_count)
        decoder_hidden = self.norm_out(x)
        logits = self.final_proj(decoder_hidden)
        return tuple([logits, decoder_hidden, alignment_scores] + new_ks + new_vs)

    @classmethod
    def from_gguf(cls, reader: GGUFReader) -> "V2607DecoderStep":
        apply_layers = tuple(reader.kvs.get("magpietts.inference.apply_prior_to_layers", [2, 3, 4, 5, 6, 7, 8, 9, 10]))
        align_layers = tuple(reader.kvs.get("magpietts.inference.estimate_alignment_from_layers", [4, 5, 8, 9]))
        m = cls(
            n_layers=int(reader.kvs["magpietts.decoder.layers"]),
            d_model=int(reader.kvs["magpietts.embedding_dim"]),
            d_ffn=int(reader.kvs["magpietts.ffn_dim"]),
            sa_heads=int(reader.kvs["magpietts.decoder.heads"]),
            xa_heads=int(reader.kvs["magpietts.decoder.cross_heads"]),
            xa_d_head=int(reader.kvs["magpietts.decoder.cross_head_dim"]),
            stacked_codebooks=int(reader.kvs["magpietts.stacked_audio_codebooks"]),
            audio_vocab=int(reader.kvs["magpietts.audio_vocab_size"]),
            max_pos=int(reader.kvs["magpietts.context_length"]),
            apply_prior_to_layers=apply_layers,
            estimate_alignment_from_layers=align_layers,
        )
        m.position_embeddings.weight.data.copy_(reader.get_torch_f32("decoder.position_embeddings.weight"))
        m.norm_out.weight.data.copy_(reader.get_torch_f32("decoder.norm_out.weight"))
        m.final_proj.weight.data.copy_(reader.get_torch_f32("final_proj.weight"))
        m.final_proj.bias.data.copy_(reader.get_torch_f32("final_proj.bias"))
        for i, layer in enumerate(m.layers):
            p = f"decoder.layers.{i}."
            layer.norm_sa.weight.data.copy_(reader.get_torch_f32(p + "norm_self.weight"))
            layer.self_attn.qkv_proj.weight.data.copy_(reader.get_torch_f32(p + "self_attention.qkv_net.weight"))
            layer.self_attn.o_proj.weight.data.copy_(reader.get_torch_f32(p + "self_attention.o_net.weight"))
            layer.norm_xa_q.weight.data.copy_(reader.get_torch_f32(p + "norm_xattn_query.weight"))
            layer.norm_xa_m.weight.data.copy_(reader.get_torch_f32(p + "norm_xattn_memory.weight"))
            layer.cross_attn.q_proj.weight.data.copy_(reader.get_torch_f32(p + "cross_attention.q_net.weight"))
            layer.cross_attn.kv_proj.weight.data.copy_(reader.get_torch_f32(p + "cross_attention.kv_net.weight"))
            layer.cross_attn.o_proj.weight.data.copy_(reader.get_torch_f32(p + "cross_attention.o_net.weight"))
            layer.norm_ff.weight.data.copy_(reader.get_torch_f32(p + "norm_pos_ff.weight"))
            layer.ffn.conv1.weight.data.copy_(reader.get_torch_f32(p + "pos_ff.proj.conv.weight.k0").unsqueeze(-1))
            layer.ffn.conv2.weight.data.copy_(reader.get_torch_f32(p + "pos_ff.o_net.conv.weight.k0").unsqueeze(-1))
        return m


# MARK: - Download Helpers & Constant / Tokenizer Export

def download_file(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    print(f"Downloading {url} -> {dst} ...")
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)
    print(f"  Saved {dst.stat().st_size / (1024 * 1024):.1f} MB")
    return dst


def export_constants_and_tokenizer(reader: GGUFReader, out_dir: Path) -> None:
    const_dir = out_dir / "constants"
    lt_dir = const_dir / "local_transformer"
    tok_dir = out_dir / "tokenizer"
    lt_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    # Speakers: baked_context_embedding.weight is [5, 217*768] FP16 -> save 5 x [217, 768] FP16
    baked = reader.get_numpy("baked_context_embedding.weight")  # (5, 166656) f16
    t_ctx = int(reader.kvs["magpietts.baked_context_length"])
    d_model = int(reader.kvs["magpietts.embedding_dim"])
    speaker_names = list(reader.kvs["magpietts.speaker_names"])
    for spk in range(baked.shape[0]):
        emb = baked[spk].reshape(t_ctx, d_model).astype(np.float16)
        np.save(const_dir / f"speaker_{spk}.npy", emb)

    # 16 stacked audio embedding tables (FP16 [2024, 768])
    stacked = int(reader.kvs["magpietts.stacked_audio_codebooks"])
    for cb in range(stacked):
        arr = reader.get_numpy(f"audio_embeddings.{cb}.weight").astype(np.float16)
        np.save(const_dir / f"audio_embedding_{cb}.npy", arr)

    # 2-layer 12-head 768-dim LocalTransformer weights (FP16 weights, FP32 norms & biases)
    lt_layers = int(reader.kvs["magpietts.local_transformer.layers"])
    np.save(lt_dir / "pos_emb.npy", reader.get_numpy("local_transformer.position_embeddings.weight").astype(np.float16))
    for l in range(lt_layers):
        p = f"local_transformer.layers.{l}."
        np.save(lt_dir / f"layer_{l}_norm1_weight.npy", reader.get_numpy(p + "norm_self.weight").astype(np.float32))
        np.save(lt_dir / f"layer_{l}_sa_qkv_weight.npy", reader.get_numpy(p + "self_attention.qkv_net.weight").astype(np.float16))
        np.save(lt_dir / f"layer_{l}_sa_o_weight.npy", reader.get_numpy(p + "self_attention.o_net.weight").astype(np.float16))
        np.save(lt_dir / f"layer_{l}_norm2_weight.npy", reader.get_numpy(p + "norm_pos_ff.weight").astype(np.float32))
        np.save(lt_dir / f"layer_{l}_ffn_conv1_weight.npy", reader.get_numpy(p + "pos_ff.proj.conv.weight.k0").astype(np.float16))
        np.save(lt_dir / f"layer_{l}_ffn_conv2_weight.npy", reader.get_numpy(p + "pos_ff.o_net.conv.weight.k0").astype(np.float16))
    for cb in range(stacked):
        np.save(lt_dir / f"out_proj_{cb}_weight.npy", reader.get_numpy(f"local_transformer_out_projections.{cb}.weight").astype(np.float16))
        np.save(lt_dir / f"out_proj_{cb}_bias.npy", reader.get_numpy(f"local_transformer_out_projections.{cb}.bias").astype(np.float32))

    # Tokenizer JSONs (exact 103-token es-ES IPA table & es_ES_nv230301 dictionary)
    download_file(SPANISH_TOKEN2ID_URL, tok_dir / "spanish_phoneme_token2id.json")
    download_file(SPANISH_DICT_URL, tok_dir / "spanish_phoneme_phoneme_dict.json")

    constants_meta = {
        "version": "v2607",
        "model_repo": "nvidia/magpie_tts_multilingual_357m",
        "tokenizer_profile": str(reader.kvs.get("magpietts.tokenizer_profile", "v2607")),
        "model_type": str(reader.kvs.get("magpietts.model_type", "decoder_ce")),
        "embedding_dim": d_model,
        "frame_stacking_factor": int(reader.kvs["magpietts.frame_stacking_factor"]),
        "num_audio_codebooks": int(reader.kvs["magpietts.audio_codebooks"]),
        "stacked_audio_codebooks": stacked,
        "codebook_size": int(reader.kvs["magpietts.audio_codebook_size"]),
        "num_all_tokens_per_codebook": int(reader.kvs["magpietts.audio_vocab_size"]),
        "sample_rate": 22050,
        "output_sample_rate": 22050,
        "codec_samples_per_frame": 1024,
        "baked_context_length": t_ctx,
        "speaker_names": speaker_names,
        "text_vocab_size": int(reader.kvs["magpietts.text_vocab_size"]),
        "spanish_phoneme_offset": 480,
        "spanish_pad_token_id": 480 + 101,
        "special_tokens": {
            "audio_bos_id": int(reader.kvs["magpietts.audio_bos_id"]),
            "audio_eos_id": int(reader.kvs["magpietts.audio_eos_id"]),
            "context_audio_bos_id": int(reader.kvs["magpietts.context_audio_bos_id"]),
            "context_audio_eos_id": int(reader.kvs["magpietts.context_audio_eos_id"]),
            "mask_token_id": int(reader.kvs["magpietts.mask_token_id"]),
            "text_bos_id": int(reader.kvs["magpietts.text_vocab_size"]) - 2,
            "text_eos_id": int(reader.kvs["magpietts.text_vocab_size"]) - 1,
        },
        "local_transformer": {
            "n_layers": lt_layers,
            "n_heads": int(reader.kvs["magpietts.local_transformer.heads"]),
            "hidden_dim": int(reader.kvs["magpietts.local_transformer.hidden_dim"]),
            "context_length": int(reader.kvs["magpietts.local_transformer.context_length"]),
            "has_in_proj": False,
        },
        "inference": {
            "max_decoder_steps": int(reader.kvs["magpietts.inference.max_decoder_steps"]),
            "topk": int(reader.kvs["magpietts.inference.topk"]),
            "min_generated_frames": int(reader.kvs["magpietts.inference.min_generated_frames"]),
            "temperature": float(reader.kvs["magpietts.inference.temperature"]),
            "cfg_scale": float(reader.kvs["magpietts.inference.cfg_scale"]),
            "apply_attention_prior": bool(reader.kvs["magpietts.inference.apply_attention_prior"]),
            "attention_prior_epsilon": float(reader.kvs["magpietts.inference.attention_prior_epsilon"]),
            "attention_prior_lookahead_window": int(reader.kvs["magpietts.inference.attention_prior_lookahead_window"]),
            "start_prior_after_n_audio_steps": int(reader.kvs["magpietts.inference.start_prior_after_n_audio_steps"]),
            "attention_prior_advance_threshold": int(reader.kvs["magpietts.inference.attention_prior_advance_threshold"]),
            "attention_prior_decay_threshold": int(reader.kvs["magpietts.inference.attention_prior_decay_threshold"]),
            "estimate_alignment_from_layers": list(reader.kvs["magpietts.inference.estimate_alignment_from_layers"]),
            "apply_prior_to_layers": list(reader.kvs["magpietts.inference.apply_prior_to_layers"]),
        },
    }
    (const_dir / "constants.json").write_text(json.dumps(constants_meta, indent=2), encoding="utf-8")
    (const_dir / "speaker_info.json").write_text(
        json.dumps(
            {
                "num_speakers": len(speaker_names),
                "T": t_ctx,
                "D": d_model,
                "lens": [t_ctx] * len(speaker_names),
                "names": {str(i): name for i, name in enumerate(speaker_names)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Exported Magpie v2607 constants + tokenizer to {out_dir}")


# MARK: - Core ML Conversion

def convert_to_coreml(reader: GGUFReader, out_dir: Path, max_text_len: int = 256, max_seq_len: int = 512) -> None:
    import coremltools as ct

    out_dir.mkdir(parents=True, exist_ok=True)
    d_model = int(reader.kvs["magpietts.embedding_dim"])
    n_layers = int(reader.kvs["magpietts.decoder.layers"])
    sa_heads = int(reader.kvs["magpietts.decoder.heads"])
    d_head = d_model // sa_heads
    t_ctx = int(reader.kvs["magpietts.baked_context_length"])

    # 1. Text Encoder
    print("[1/3] Tracing & converting V2607TextEncoder (FP16)...")
    t0 = time.time()
    enc = V2607TextEncoder.from_gguf(reader).eval()
    toks_ex = torch.zeros((1, max_text_len), dtype=torch.int32)
    mask_ex = torch.ones((1, max_text_len), dtype=torch.float32)
    with torch.no_grad():
        traced_enc = torch.jit.trace(enc, (toks_ex, mask_ex))
    ml_enc = ct.convert(
        traced_enc,
        inputs=[
            ct.TensorType(name="text_tokens", shape=(1, max_text_len), dtype=np.int32),
            ct.TensorType(name="text_mask", shape=(1, max_text_len), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="encoder_output", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    ml_enc.save(str(out_dir / "text_encoder.mlpackage"))
    del enc, traced_enc, ml_enc
    print(f"  Saved text_encoder.mlpackage in {time.time() - t0:.1f}s")

    # 2. Decoder Prefill (T_ctx=217 -> 24 rank-4 KV caches k0..k11, v0..v11)
    print("[2/3] Tracing & converting V2607DecoderPrefill (T_ctx=217, FP16)...")
    t0 = time.time()
    prefill = V2607DecoderPrefill.from_gguf(reader, max_seq_len=max_seq_len).eval()
    ae_ex = torch.randn((1, t_ctx, d_model), dtype=torch.float32)
    eo_ex = torch.randn((1, max_text_len, d_model), dtype=torch.float32)
    em_ex = torch.ones((1, max_text_len), dtype=torch.float32)
    with torch.no_grad():
        traced_pf = torch.jit.trace(prefill, (ae_ex, eo_ex, em_ex))
    pf_outputs = [ct.TensorType(name=f"k{i}", dtype=np.float16) for i in range(n_layers)] + [
        ct.TensorType(name=f"v{i}", dtype=np.float16) for i in range(n_layers)
    ]
    ml_pf = ct.convert(
        traced_pf,
        inputs=[
            ct.TensorType(name="audio_embed", shape=(1, t_ctx, d_model), dtype=np.float16),
            ct.TensorType(name="encoder_output", shape=(1, max_text_len, d_model), dtype=np.float16),
            ct.TensorType(name="encoder_mask", shape=(1, max_text_len), dtype=np.float32),
        ],
        outputs=pf_outputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    ml_pf.save(str(out_dir / "decoder_prefill.mlpackage"))
    del prefill, traced_pf, ml_pf
    print(f"  Saved decoder_prefill.mlpackage in {time.time() - t0:.1f}s")

    # 3. Decoder Step (with attention prior, alignment scores, and 32384 logits!)
    print("[3/3] Tracing & converting V2607DecoderStep (FP16, attention prior + alignment + logits)...")
    t0 = time.time()
    step = V2607DecoderStep.from_gguf(reader).eval()
    s_ae = torch.randn((1, 1, d_model), dtype=torch.float32)
    s_eo = torch.randn((1, max_text_len, d_model), dtype=torch.float32)
    s_em = torch.ones((1, max_text_len), dtype=torch.float32)
    s_pr = torch.zeros((1, max_text_len), dtype=torch.float32)
    s_pos = torch.tensor([float(t_ctx)], dtype=torch.float32)
    cks = [torch.zeros((1, max_seq_len, sa_heads, d_head), dtype=torch.float32) for _ in range(n_layers)]
    cvs = [torch.zeros((1, max_seq_len, sa_heads, d_head), dtype=torch.float32) for _ in range(n_layers)]
    with torch.no_grad():
        traced_step = torch.jit.trace(step, tuple([s_ae, s_eo, s_em, s_pr, s_pos] + cks + cvs))

    step_inputs = [
        ct.TensorType(name="audio_embed", shape=(1, 1, d_model), dtype=np.float16),
        ct.TensorType(name="encoder_output", shape=(1, max_text_len, d_model), dtype=np.float16),
        ct.TensorType(name="encoder_mask", shape=(1, max_text_len), dtype=np.float32),
        ct.TensorType(name="attn_prior_bias", shape=(1, max_text_len), dtype=np.float32),
        ct.TensorType(name="position", shape=(1,), dtype=np.float16),
    ]
    for i in range(n_layers):
        step_inputs.append(ct.TensorType(name=f"cache_k{i}", shape=(1, max_seq_len, sa_heads, d_head), dtype=np.float16))
    for i in range(n_layers):
        step_inputs.append(ct.TensorType(name=f"cache_v{i}", shape=(1, max_seq_len, sa_heads, d_head), dtype=np.float16))

    step_outputs = [
        ct.TensorType(name="logits", dtype=np.float16),
        ct.TensorType(name="decoder_hidden", dtype=np.float16),
        ct.TensorType(name="alignment_scores", dtype=np.float16),
    ]
    for i in range(n_layers):
        step_outputs.append(ct.TensorType(name=f"new_k{i}", dtype=np.float16))
    for i in range(n_layers):
        step_outputs.append(ct.TensorType(name=f"new_v{i}", dtype=np.float16))

    ml_step = ct.convert(
        traced_step,
        inputs=step_inputs,
        outputs=step_outputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    ml_step.save(str(out_dir / "decoder_step.mlpackage"))
    del step, traced_step, ml_step
    print(f"  Saved decoder_step.mlpackage in {time.time() - t0:.1f}s")


# MARK: - PyTorch Self-Verification on Real Spanish Utterances

def verify_pytorch_spanish(reader: GGUFReader, tok_dir: Path, speaker: int = 4) -> None:
    """Runs genuine v2607 PyTorch inference (encoder + prefill + AR step + 2-layer LT +
    attention prior + argmax_or_multinomial_any EOS) on real Spanish words & sentences
    to verify clean alignment, exact EOS termination, and zero runaway trailing frames."""
    token2id = json.loads((tok_dir / "spanish_phoneme_token2id.json").read_text(encoding="utf-8"))
    ph_dict = json.loads((tok_dir / "spanish_phoneme_phoneme_dict.json").read_text(encoding="utf-8"))

    offset = 480
    eos_id = 3358
    pad_id = offset + token2id["<pad>"]
    space_id = offset + token2id[" "]

    def tokenize_es(text: str) -> List[int]:
        s = text.strip()
        if s and s[-1] not in ".?!…":
            s += "."
        ids = [space_id]
        words = s.split()
        for wi, w in enumerate(words):
            lp = ""
            for ch in w:
                if not ch.isalnum():
                    lp += ch
                else:
                    break
            rp = ""
            for ch in reversed(w):
                if not ch.isalnum():
                    rp = ch + rp
                else:
                    break
            core = w[len(lp) : len(w) - len(rp) if rp else len(w)]
            for ch in lp:
                if ch in token2id:
                    ids.append(offset + token2id[ch])
            if core:
                up = core.upper()
                if up in ph_dict:
                    for p in ph_dict[up]:
                        if p in token2id:
                            ids.append(offset + token2id[p])
                else:
                    for ch in up:
                        if ch in token2id:
                            ids.append(offset + token2id[ch])
            for ch in rp:
                if ch in token2id:
                    ids.append(offset + token2id[ch])
            if wi + 1 < len(words):
                ids.append(space_id)
        ids.append(space_id)
        ids.append(eos_id)
        if len(ids) - 2 < 4:
            pad_n = 4 - (len(ids) - 2)
            ids = ids[:1] + [pad_id] * pad_n + ids[1:]
        return ids

    torch.manual_seed(42)
    enc = V2607TextEncoder.from_gguf(reader).eval()
    pf = V2607DecoderPrefill.from_gguf(reader).eval()
    ds = V2607DecoderStep.from_gguf(reader).eval()

    # Load speaker, audio embs, 2-layer LT
    t_ctx = int(reader.kvs["magpietts.baked_context_length"])
    d_model = int(reader.kvs["magpietts.embedding_dim"])
    spk_emb = reader.get_torch_f32("baked_context_embedding.weight")[speaker].reshape(1, t_ctx, d_model)
    audio_embs = [reader.get_torch_f32(f"audio_embeddings.{cb}.weight") for cb in range(16)]
    lt_pos = reader.get_torch_f32("local_transformer.position_embeddings.weight")
    lt_layers = []
    for l in range(2):
        p = f"local_transformer.layers.{l}."
        lt_layers.append((
            reader.get_torch_f32(p + "norm_self.weight"),
            reader.get_torch_f32(p + "self_attention.qkv_net.weight"),
            reader.get_torch_f32(p + "self_attention.o_net.weight"),
            reader.get_torch_f32(p + "norm_pos_ff.weight"),
            reader.get_torch_f32(p + "pos_ff.proj.conv.weight.k0"),
            reader.get_torch_f32(p + "pos_ff.o_net.conv.weight.k0"),
        ))
    lt_out_w = [reader.get_torch_f32(f"local_transformer_out_projections.{cb}.weight") for cb in range(16)]
    lt_out_b = [reader.get_torch_f32(f"local_transformer_out_projections.{cb}.bias") for cb in range(16)]

    def ln_nobias(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5) * w

    def lt_forward(seq: torch.Tensor) -> torch.Tensor:
        T = seq.shape[0]
        x = seq + lt_pos[:T]
        for norm1, qkv_w, o_w, norm2, f1, f2 in lt_layers:
            xn = ln_nobias(x, norm1)
            qkv = (xn @ qkv_w.T).view(T, 3, 12, 64)
            q, k, v = qkv[:, 0].transpose(0, 1), qkv[:, 1].transpose(0, 1), qkv[:, 2].transpose(0, 1)
            attn = (q @ k.transpose(-2, -1)) * (64 ** -0.5)
            causal = torch.tril(torch.ones(T, T))
            attn = F.softmax(attn.masked_fill(causal == 0, -1e9), dim=-1)
            out = (attn @ v).transpose(0, 1).reshape(T, 768) @ o_w.T
            x = x + out
            xn2 = ln_nobias(x, norm2)
            h = F.gelu(xn2 @ f1.T, approximate="tanh") @ f2.T
            x = x + h
        return x

    # Precompute uncond prefill
    with torch.no_grad():
        zero_ctx = torch.zeros((1, t_ctx, d_model))
        zero_enc = torch.zeros((1, 256, d_model))
        uncond_mask = torch.zeros((1, 256))
        uncond_mask[0, 0] = 1.0
        uncond_bias = torch.full((1, 256), MASK_NEG)
        uncond_bias[0, 0] = 0.0
        u_pf = pf(zero_ctx, zero_enc, uncond_mask)
        u_k0, u_v0 = list(u_pf[:12]), list(u_pf[12:])

        for phrase in ["por", "pero", "ya", "dado", "que", "Hola, esto es una prueba."]:
            toks = tokenize_es(phrase)
            t_len = len(toks)
            tok_t = torch.zeros((1, 256), dtype=torch.int32)
            tok_t[0, :t_len] = torch.tensor(toks, dtype=torch.int32)
            mask_t = torch.zeros((1, 256), dtype=torch.float32)
            mask_t[0, :t_len] = 1.0
            enc_out = enc(tok_t, mask_t)
            c_pf = pf(spk_emb, enc_out, mask_t)
            ck, cv = list(c_pf[:12]), list(c_pf[12:])
            uk, uv = [t.clone() for t in u_k0], [t.clone() for t in u_v0]

            # Attention prior state (exact NeMo v2607)
            last_attended = 1
            counts = [0] * t_len
            prior_bias = torch.full((1, 256), MASK_NEG)
            prior_bias[0, :t_len] = 0.0

            cur_codes = [2016] * 16
            frames_total = 0
            eos_src = "max"
            for step_idx in range(60):
                ae = sum(audio_embs[c][cur_codes[c]] for c in range(16)) / 16.0
                ae = ae.view(1, 1, 768)
                pos_t = torch.tensor([float(t_ctx + step_idx)])
                c_out = ds(ae, enc_out, mask_t, prior_bias, pos_t, *ck, *cv)
                c_logits, c_hid, align = c_out[0], c_out[1], c_out[2]
                ck, cv = list(c_out[3:15]), list(c_out[15:27])

                u_out = ds(ae, zero_enc, uncond_mask, uncond_bias, pos_t, *uk, *uv)
                u_logits, u_hid = u_out[0], u_out[1]
                uk, uv = list(u_out[3:15]), list(u_out[15:27])

                # Update attention prior (exact NeMo MagpieTTSModel.get_most_attended_text_timestep)
                scores = align[0, :t_len].tolist()
                prev = max(0, min(last_attended, t_len - 1))
                last = prev
                if counts[last] >= 8:
                    last = min(last + 1, t_len - 1)
                w_end = min(last + 6, t_len - 3)
                if w_end > last:
                    best_i = last + int(np.argmax(scores[last:w_end]))
                else:
                    best_i = t_len - 1
                last_attended = max(0, min(best_i, t_len - 1))
                counts[last_attended] += 1
                p_vals = [0.1] * t_len
                if t_len <= 5:
                    p_vals = [1.0] * t_len
                else:
                    p_vals[max(1, last_attended - 1)] = 1.0
                    p_vals[last_attended] = 1.0
                    for d in range(1, 7):
                        p_vals[min(last_attended + d, t_len - 1)] = 1.0
                for ti, cnt in enumerate(counts):
                    if cnt >= 10:
                        for j in range(min(ti + 1, last_attended)):
                            p_vals[j] = 0.1
                prior_bias.fill_(MASK_NEG)
                for ti, pv in enumerate(p_vals):
                    prior_bias[0, ti] = math.log(max(pv, 1e-6))

                forbid_eos = (step_idx * 2) < 4
                forbidden = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023] if forbid_eos else [2016, 2018, 2019, 2020, 2021, 2022, 2023]

                # Main decoder head greedy argmax (exact NeMo argmax_or_multinomial_any)
                cfg_main = (2.5 * c_logits - 1.5 * u_logits).view(16, 2024)
                cfg_main[:, forbidden] = -1e9
                greedy_main = cfg_main.argmax(dim=-1).tolist()

                # LocalTransformer autoregressive sampling (16 slots = 2 frames x 8 codebooks)
                c_seq = c_hid.view(1, 768)
                u_seq = u_hid.view(1, 768)
                sampled: List[int] = []
                greedy_lt: List[int] = []
                for cb in range(16):
                    cl = lt_forward(c_seq)[-1] @ lt_out_w[cb].T + lt_out_b[cb]
                    ul = lt_forward(u_seq)[-1] @ lt_out_w[cb].T + lt_out_b[cb]
                    logits = 2.5 * cl - 1.5 * ul
                    logits[forbidden] = -1e9
                    greedy_lt.append(int(logits.argmax().item()))
                    vals, idxs = torch.topk(logits, 80)
                    probs = F.softmax(vals / 0.6, dim=-1)
                    pick = int(idxs[torch.multinomial(probs, 1).item()].item())
                    sampled.append(pick)
                    if cb < 15:
                        nxt = audio_embs[cb][pick].view(1, 768)
                        c_seq = torch.cat([c_seq, nxt], dim=0)
                        u_seq = torch.cat([u_seq, nxt], dim=0)

                eos_lane = -1
                if not forbid_eos:
                    for lane in range(2):
                        s_lane = sampled[lane * 8 : (lane + 1) * 8]
                        gm_lane = greedy_main[lane * 8 : (lane + 1) * 8]
                        gl_lane = greedy_lt[lane * 8 : (lane + 1) * 8]
                        if 2017 in s_lane or 2017 in gm_lane or 2017 in gl_lane:
                            eos_lane = lane
                            eos_src = (
                                "lt_sample" if 2017 in s_lane else ("main_argmax" if 2017 in gm_lane else "lt_argmax")
                            )
                            break
                if eos_lane >= 0:
                    frames_total += eos_lane
                    break
                frames_total += 2
                cur_codes = sampled

            dur = frames_total * 1024 / 22050.0
            print(
                f"  [v2607 verify] {phrase!r:28s} -> tokens={t_len:2d}, "
                f"frames={frames_total:2d} ({dur:.2f}s), last_attended={last_attended}/{t_len - 1}, eos={eos_src}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NVIDIA Magpie TTS v2607 GGUF -> Core ML")
    parser.add_argument("--gguf", type=Path, default=None, help="Path to local v2607 F16 GGUF (auto-downloads if omitted)")
    parser.add_argument("--out", type=Path, default=Path("models/coreml/magpie_v2607"))
    parser.add_argument("--verify-pytorch", action="store_true", help="Run PyTorch Spanish end-to-end verification")
    parser.add_argument("--skip-coreml", action="store_true", help="Skip Core ML conversion (verify/constants only)")
    args = parser.parse_args()

    gguf_path = args.gguf or (args.out.parent / "magpie_tts_multilingual_357m.v2607.f16.gguf")
    download_file(V2607_GGUF_URL, gguf_path)

    print(f"Parsing GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)
    print(
        f"Loaded NVIDIA Magpie {reader.kvs.get('magpietts.tokenizer_profile')} "
        f"(frame_stacking={reader.kvs['magpietts.frame_stacking_factor']}, "
        f"vocab={reader.kvs['magpietts.text_vocab_size']}, "
        f"T_ctx={reader.kvs['magpietts.baked_context_length']}, "
        f"LT={reader.kvs['magpietts.local_transformer.layers']}L/"
        f"{reader.kvs['magpietts.local_transformer.heads']}H/"
        f"{reader.kvs['magpietts.local_transformer.hidden_dim']}D)"
    )

    export_constants_and_tokenizer(reader, args.out)
    if args.verify_pytorch:
        verify_pytorch_spanish(reader, args.out / "tokenizer")
    if not args.skip_coreml:
        convert_to_coreml(reader, args.out)
    print("All Magpie v2607 export steps complete!")


if __name__ == "__main__":
    main()
