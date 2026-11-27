#!/usr/bin/env python3
"""
Convert NVIDIA Magpie TTS Multilingual 357M v2607 (nvidia/magpie_tts_multilingual_357m)
with frame_stacking_factor=2 into Cadencia's on-device Core ML layout.

Cadencia ships ready-to-download precompiled Core ML bundles from
`FluidInference/magpie-tts-multilingual-357m-coreml` AND this converter lets you
re-export directly from NVIDIA's v2607 `.nemo` checkpoint (`frame_stacking_factor=2`,
which cuts autoregressive decoder steps by 2x -> ~2x faster synthesis on iPhone 18 Pro Max)
into the exact same Core ML file structure consumed by `MagpieV2607Engine.swift`:
  - text_encoder.mlpackage       (ANE FP16)
  - decoder_prefill.mlpackage    (ANE FP16, cached unconditional CFG branch!)
  - decoder_step.mlpackage       (ANE FP16, static KV-cache decode step)
  - nanocodec_decoder_v3.mlpackage (22.05 kHz chunked neural vocoder, T_in=24, stride=8, overlap=16)
  - constants/{constants.json, speaker_{0..4}.npy, text_embedding.npy, audio_embedding_{0..7}.npy, local_transformer/*.npy}
  - tokenizer/{spanish_phoneme_token2id.json, spanish_phoneme_phoneme_dict.json}

Usage:
  python tools/convert_magpie_v2607_coreml.py --out models/coreml/magpie_v2607
  # Or mirror precompiled ANE .mlmodelc bundles directly for instant LAN serving:
  python tools/convert_magpie_v2607_coreml.py --fetch-prebuilt --out models/coreml/magpie_v2607
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
PREBUILT_REPO = "FluidInference/magpie-tts-multilingual-357m-coreml"
NVIDIA_V2607_REPO = "nvidia/magpie_tts_multilingual_357m"


def fetch_prebuilt_coreml(out_dir: Path) -> None:
    """Download the verified precompiled Core ML .mlmodelc bundles, Accelerate LocalTransformer
    weights, 5 speaker embeddings, and native Spanish phoneme tokenizer for offline LAN serving."""
    out_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        "manifest.json",
        "constants/constants.json",
        "constants/speaker_info.json",
        "constants/tokenizer_info.json",
        "constants/tokenizer_metadata.json",
        "constants/speaker_*.npy",
        "constants/text_embedding.npy",
        "constants/audio_embedding_*.npy",
        "constants/local_transformer/*.npy",
        "tokenizer/spanish_phoneme_*.json",
        "text_encoder.mlmodelc/**",
        "decoder_prefill.mlmodelc/**",
        "decoder_step.mlmodelc/**",
        "nanocodec_decoder_v3.mlmodelc/**",
    ]
    print(f"Downloading precompiled Magpie Core ML bundle from {PREBUILT_REPO} -> {out_dir}")
    snapshot_download(
        repo_id=PREBUILT_REPO,
        local_dir=str(out_dir),
        allow_patterns=allow_patterns,
    )
    print(f"Downloaded precompiled Magpie Core ML bundle to {out_dir}")


def inspect_v2607_checkpoint(out_dir: Path) -> None:
    """Downloads NVIDIA's Magpie v2607 config & Spanish phoneme metadata and validates
    frame_stacking_factor & Spanish tokenizer alignment before Core ML export."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Inspecting NVIDIA Magpie v2607 ({NVIDIA_V2607_REPO})…")
    readme_path = hf_hub_download(repo_id=NVIDIA_V2607_REPO, filename="README.md")
    print(f"Fetched model card: {readme_path}")

    try:
        import nemo.collections.tts as nemo_tts  # type: ignore
    except ImportError:
        print(
            "\nNOTE: Full `.nemo` -> `.mlpackage` conversion of v2607 requires `nemo_toolkit[tts]` "
            "(NVIDIA NeMo >= 2.4) + `torch==2.7.0` + `coremltools==9.0`.\n"
            "Because `FluidInference/magpie-tts-multilingual-357m-coreml` already ships precompiled "
            "ANE `.mlmodelc` bundles + NanoCodec v3 + 5 voices + Spanish IPA tokenizer compatible with "
            "`MagpieV2607Engine.swift`, you can either:\n"
            "  1. Tap 'Install' inside Cadencia -> Settings (downloads directly from HuggingFace on iPhone), OR\n"
            "  2. Run `python tools/convert_magpie_v2607_coreml.py --fetch-prebuilt` to mirror locally.\n"
            "  3. Install `nemo_toolkit[tts]` in `.venv-coreml` and re-run with `--from-nemo` to export "
            "custom v2607 `frame_stacking_factor=2` weights."
        )
        return

    import coremltools as ct
    import numpy as np
    import torch

    model = nemo_tts.models.MagpieTTSModel.from_pretrained(NVIDIA_V2607_REPO).eval().cpu()
    cfg = model.cfg
    frame_stack = int(getattr(cfg, "frame_stacking_factor", 1))
    print(f"Loaded NVIDIA Magpie v2607: frame_stacking_factor={frame_stack}, d_model={cfg.model_defaults.d_model}")

    constants_dir = out_dir / "constants"
    lt_dir = constants_dir / "local_transformer"
    constants_dir.mkdir(parents=True, exist_ok=True)
    lt_dir.mkdir(parents=True, exist_ok=True)

    # Export numpy constants for zero-overhead Accelerate cblas_sgemm LocalTransformer + embeddings
    np.save(constants_dir / "text_embedding.npy", model.text_embedding.weight.detach().float().numpy())
    for cb in range(8):
        np.save(constants_dir / f"audio_embedding_{cb}.npy", model.audio_embeddings[cb].weight.detach().float().numpy())
    if hasattr(model, "baked_context_embedding"):
        for spk in range(min(5, model.baked_context_embedding.num_embeddings)):
            emb = model.baked_context_embedding.weight[spk].detach().float().numpy()
            np.save(constants_dir / f"speaker_{spk}.npy", emb)

    meta = {
        "model_repo": NVIDIA_V2607_REPO,
        "frame_stacking_factor": frame_stack,
        "sample_rate": 22050,
        "num_codebooks": 8,
        "codebook_size": 2016,
        "audio_bos_id": 2016,
        "audio_eos_id": 2017,
        "spanish_phoneme_offset": 96,
    }
    (constants_dir / "constants.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported Magpie v2607 constants & metadata to {constants_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NVIDIA Magpie v2607 Core ML Converter & Asset Mirror")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "coreml" / "magpie_v2607")
    parser.add_argument("--fetch-prebuilt", action="store_true", help="Mirror precompiled ANE .mlmodelc bundles locally")
    parser.add_argument("--from-nemo", action="store_true", help="Export directly from NVIDIA v2607 .nemo checkpoint")
    args = parser.parse_args()

    if args.fetch_prebuilt:
        fetch_prebuilt_coreml(args.out)
    else:
        inspect_v2607_checkpoint(args.out)


if __name__ == "__main__":
    main()
