//! Pure-Rust on-device Core ML exporter for `Phonon-40m-es` (and any Phonon TTS checkpoint).
//!
//! Exposed via `ptts_export_coreml(checkpoint_dir, voices_dir, out_dir)` in `PhononCore.xcframework`
//! so Cadencia on iPhone downloads the user-authenticated Hugging Face checkpoint directly on-device
//! and builds the Apple Neural Engine Core ML `.mlpackage` bundle locally with zero Python, zero
//! computer transfer, and zero redistribution of gated weights.

use ptts::checkpoint::{Checkpoint, ResolveOptions};
use ptts_coreml::Weights;
use ptts_coreml::mil::Gelu;
use ptts_coreml::package::write_mlpackage_with_weights;
use ptts_coreml::phonon::{flow_lm as fl, mimi};
use safetensors::tensor::{Dtype, TensorView};
use sha2::Digest;
use std::collections::HashMap;
use std::path::Path;

fn write_pkg(out: &Path, name: &str, built: fl::Built) -> Result<(), String> {
    let (model, blob) = built;
    write_mlpackage_with_weights(&out.join(format!("{name}.mlpackage")), &model, blob)
        .map_err(|e| format!("writing {name}.mlpackage: {e}"))
}

pub fn export_bundle(
    checkpoint_dir: &Path,
    voices_dir: Option<&Path>,
    out_dir: &Path,
) -> Result<(), String> {
    let ck = Checkpoint::resolve(checkpoint_dir, ResolveOptions::default())
        .map_err(|e| format!("resolving checkpoint {}: {e}", checkpoint_dir.display()))?;
    let cfg = &ck.config;
    let f = &cfg.flow_lm;

    let dims = fl::Dims {
        d: f.d_model,
        heads: f.num_heads,
        attn: f.attn_dim.unwrap_or(f.d_model),
        kv_heads: f.num_kv_heads.unwrap_or(f.num_heads),
        layers: f.num_layers,
        ff: f.dim_feedforward,
        ldim: f.ldim,
        z: f.z_dim,
        flow_d: f.flow_dim,
        flow_blocks: f.flow_depth,
    };
    if dims.heads == 0
        || dims.attn % dims.heads != 0
        || dims.kv_heads == 0
        || dims.heads % dims.kv_heads != 0
    {
        return Err("checkpoint attention heads do not divide width or KV heads".into());
    }

    let flow = match f.flow_type {
        ptts::flow_lm::FlowType::Lsd => fl::FlowType::Lsd,
        ptts::flow_lm::FlowType::Drifting => fl::FlowType::Drifting,
        ptts::flow_lm::FlowType::Pointwise => fl::FlowType::Pointwise,
    };
    if flow == fl::FlowType::Lsd && cfg.lsd_decode_steps != 1 {
        return Err(format!(
            "Core ML flow graph takes 1 step, checkpoint requests {}",
            cfg.lsd_decode_steps
        ));
    }

    let time_norm = match f.time_rms_norm {
        ptts::flow_lm::TimeRmsNorm::Rms => fl::TimeNorm::Rms,
        ptts::flow_lm::TimeRmsNorm::RmsVar => fl::TimeNorm::RmsVar,
    };
    let gelu = |g: Option<ptts::transformer::Gelu>| match g.unwrap_or_else(|| cfg.default_gelu()) {
        ptts::transformer::Gelu::Erf => Gelu::Erf,
        ptts::transformer::Gelu::Tanh => Gelu::Tanh,
    };

    let m = &cfg.mimi;
    let hop: usize = m.ratios.iter().product();
    let mimi_dims = mimi::Dims {
        ldim: m.quantizer_dimension,
        dim: m.dimension,
        d_model: m.transformer_d_model,
        heads: m.transformer_num_heads,
        layers: m.transformer_num_layers,
        ff: m.transformer_dim_feedforward,
        out_dim: m.decoder_dimension.unwrap_or(m.dimension),
        n_filters: m.n_filters,
        ratios: m.ratios.clone(),
        kernel_size: m.kernel_size,
        residual_kernel_size: m.residual_kernel_size,
        last_kernel_size: m.last_kernel_size,
        compress: m.compress,
        steps: (m.sample_rate as f64 / hop as f64 / m.frame_rate).round() as usize,
    };
    if m.channels != 1
        || m.n_residual_layers != 1
        || mimi_dims.steps <= 1
        || (m.quantizer_dimension, m.quantizer_output_dimension) != (f.ldim, m.dimension)
    {
        return Err("unsupported Mimi decoder layout for Core ML export".into());
    }
    let window = m.transformer_context;

    // Open weights via ptts_coreml (handles both model.safetensors and model.q8.gguf)
    let wt = if ck.weights.extension().is_some_and(|e| e == "gguf") {
        Weights::open_gguf(&ck.weights)
    } else {
        Weights::open(&ck.weights)
    }
    .map_err(|e| format!("opening weights {}: {e}", ck.weights.display()))?
    .renamed(ptts::loader::remap_key);

    // Load host weights via ptts loader for speaker projection & conditioners
    let vb = ptts::loader::load_weights::<xn::Unquantized<f32, xn::CpuDevice>>(
        &ck.weights,
        &xn::CpuDevice,
    )
    .map_err(|e| format!("loading weights for voice projection: {e}"))?;
    let speaker_proj = ptts::loader::load_speaker_proj(&vb, cfg)
        .map_err(|e| format!("loading speaker projection: {e}"))?;
    let embeddings_ext = cfg.embeddings_ext.as_deref();

    let given: HashMap<String, String> = HashMap::new();
    let load_conds = |values: &HashMap<String, String>| -> Result<Vec<f32>, String> {
        match ptts::loader::load_conditions::<f32, _>(&vb, cfg, values)
            .map_err(|e| format!("loading conditions: {e}"))?
        {
            Some(sum) => sum.to_vec().map_err(|e| format!("condition to_vec: {e}")),
            None => Ok(vec![0f32; dims.d]),
        }
    };

    // Project and export all native Spanish voices
    let mut voices = Vec::new();
    if cfg.voices.is_empty() {
        let files = match voices_dir {
            Some(dir) => ptts::loader::voices_in(dir),
            None => ck.voices.clone(),
        };
        for (name, path) in files {
            let emb = ptts::loader::load_voice_emb(
                &path,
                embeddings_ext,
                speaker_proj.as_ref(),
                &xn::CpuDevice,
            )
            .map_err(|e| format!("loading voice {name}: {e}"))?;
            voices.push((name, emb, None));
        }
    } else {
        let baked = ptts::loader::load_config_voices(&vb, cfg, speaker_proj.as_ref())
            .map_err(|e| format!("loading baked voices: {e}"))?;
        for (voice, (name, emb)) in cfg.voices.iter().zip(baked) {
            let mut values = given.clone();
            values.extend(voice.conditions.clone());
            voices.push((name, emb, Some(load_conds(&values)?)));
        }
    }
    if voices.is_empty() {
        return Err("no voices found for Core ML bundle export".into());
    }

    std::fs::create_dir_all(out_dir.join("voices"))
        .map_err(|e| format!("creating voices output dir: {e}"))?;

    let mut vlen = 0usize;
    for (name, emb, voice_conditions) in &voices {
        let shape = emb.dims().to_vec();
        if shape.len() != 3 || shape[2] != dims.d {
            return Err(format!("voice {name} has shape {shape:?}, expected [1, T, {}]", dims.d));
        }
        vlen = vlen.max(shape[1]);
        let vals = emb.to_vec().map_err(|e| format!("voice {name} to_vec: {e}"))?;
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut views = vec![(
            "emb",
            TensorView::new(Dtype::F32, shape, &bytes).map_err(|e| e.to_string())?,
        )];
        let cond_bytes: Vec<u8> =
            voice_conditions.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        if voice_conditions.is_some() {
            views.push((
                "conditions",
                TensorView::new(Dtype::F32, vec![dims.d], &cond_bytes)
                    .map_err(|e| e.to_string())?,
            ));
        }
        safetensors::serialize_to_file(
            views,
            None,
            &out_dir.join(format!("voices/{name}.safetensors")),
        )
        .map_err(|e| format!("writing voice {name}: {e}"))?;
    }

    // Cap per-chunk graph budget at 64 tokens (Cadencia synthesizes one sentence at a time,
    // typically 10-35 tokens, and ptts automatically splits longer utterances across chunks).
    // Shrinking max_tokens from 200 -> 64 cuts `ctx` from ~1,200 -> ~497 (~8x smaller prefill
    // attention matrix, drastically reducing on-device Core ML size, RAM, and ANE compile time).
    let max_tokens: usize = 64;
    let temperature: f32 = 0.3;
    let max_frames = ptts::plan::frame_budget(max_tokens, m.frame_rate);
    let ctx = vlen + max_tokens + max_frames + 16;
    let prefill_len = max_tokens;

    let step_flow = fl::build(&wt, &dims, ctx, 1, time_norm, gelu(f.gelu), flow)?;
    write_pkg(out_dir, &fl::package_name(ctx, 1), step_flow)?;

    let prefill_flow = fl::build(&wt, &dims, ctx, prefill_len, time_norm, gelu(f.gelu), flow)?;
    write_pkg(out_dir, &fl::package_name(ctx, prefill_len), prefill_flow)?;

    let mimi_built = mimi::build(&wt, &mimi_dims, mimi_dims.cache_len(window), gelu(m.gelu))?;
    write_pkg(out_dir, &mimi_dims.package_name(window), mimi_built)?;

    // Host-side tensors (f32)
    let mut defaults = given.clone();
    for (name, value) in cfg.voices.first().map(|v| &v.conditions).into_iter().flatten() {
        defaults.entry(name.clone()).or_insert_with(|| value.clone());
    }

    let mut host: Vec<(&str, Vec<usize>, Vec<u8>)> = Vec::new();
    let (shape, table) = wt.get("flow_lm.conditioner.embed.weight")?;
    let (shape, table) = match wt.get("flow_lm.conditioner.output_proj.weight") {
        Ok((proj_shape, proj)) => {
            let (rows, rank, d) = (shape[0], shape[1], proj_shape[0]);
            let mut out = vec![0f32; rows * d];
            for (row, o) in table.chunks(rank).zip(out.chunks_mut(d)) {
                for (v, p) in o.iter_mut().zip(proj.chunks(rank)) {
                    *v = row.iter().zip(p).map(|(a, b)| a * b).sum();
                }
            }
            (vec![rows, d], out)
        }
        Err(_) => (shape.to_vec(), table.to_vec()),
    };
    host.push((
        "flow_lm.conditioner.embed.weight",
        shape,
        table.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ));

    for n in ["flow_lm.input_linear.weight", "flow_lm.bos_emb"] {
        let (shape, data) = wt.get(n)?;
        host.push((n, shape.to_vec(), data.iter().flat_map(|v| v.to_le_bytes()).collect()));
    }
    host.push((
        "flow_lm.conditions",
        vec![dims.d],
        load_conds(&defaults)?.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ));

    let mut views: HashMap<&str, TensorView> = HashMap::new();
    for (n, s, b) in &host {
        views.insert(
            *n,
            TensorView::new(Dtype::F32, s.clone(), b).map_err(|e| e.to_string())?,
        );
    }
    safetensors::serialize_to_file(&views, None, &out_dir.join("host.safetensors"))
        .map_err(|e| format!("writing host.safetensors: {e}"))?;

    let tokenizer = ck
        .tokenizer
        .clone()
        .ok_or_else(|| "checkpoint is missing tokenizer.json".to_string())?;
    std::fs::copy(&tokenizer, out_dir.join("tokenizer.json"))
        .map_err(|e| format!("copying tokenizer.json: {e}"))?;

    let mut files = Vec::new();
    list_files(out_dir, out_dir, &mut files)?;
    files.sort_by(|a: &serde_json::Value, b| a["path"].as_str().cmp(&b["path"].as_str()));

    let built = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();

    let meta = serde_json::json!({
        "built": built,
        "ctx": ctx,
        "prefill_len": prefill_len,
        "max_frames": max_frames,
        "mimi": mimi_dims.to_json(),
        "mimi_window": window,
        "eos_threshold": cfg.eos_threshold,
        "temperature": temperature,
        "dims": {
            "d": dims.d, "heads": dims.heads, "attn": dims.attn, "kv_heads": dims.kv_heads,
            "layers": dims.layers, "ff": dims.ff, "ldim": dims.ldim, "z": dims.z,
            "flow_d": dims.flow_d, "flow_blocks": dims.flow_blocks,
        },
        "voices": voices.iter().map(|(n, _, _)| n).collect::<Vec<_>>(),
        "languages": cfg.languages,
        "conditions": given,
        "files": files,
    });
    let meta_bytes = serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?;
    std::fs::write(out_dir.join("bundle.json"), meta_bytes)
        .map_err(|e| format!("writing bundle.json: {e}"))?;
    Ok(())
}

fn list_files(root: &Path, dir: &Path, out: &mut Vec<serde_json::Value>) -> Result<(), String> {
    for e in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let p = e.map_err(|e| e.to_string())?.path();
        if p.is_dir() {
            list_files(root, &p, out)?;
        } else if p.file_name().is_some_and(|n| n != "bundle.json") {
            let bytes = std::fs::read(&p).map_err(|e| e.to_string())?;
            let rel: Vec<String> = p
                .strip_prefix(root)
                .map_err(|e| e.to_string())?
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into())
                .collect();
            out.push(serde_json::json!({
                "path": rel.join("/"),
                "size": bytes.len(),
                "sha256": format!("{:x}", sha2::Sha256::digest(&bytes)),
            }));
        }
    }
    Ok(())
}
