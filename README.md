# Cadencia — Signal-Starvation Hypothesis Spanish Acquisition System

Complete, end-to-end native iOS application implementing the **Signal-Starvation Hypothesis** test protocol for **iPhone 18 Pro Max** (iOS 26/27 Liquid Glass + 120Hz custom Metal shaders, backwards-compatible to iOS 18+).

## Architecture Overview

| Layer | Technology | Details |
| :--- | :--- | :--- |
| **Tutor TTS (On-Device)** | **NVIDIA Magpie v2607** (`Core ML`) | 5 voices (`Sofia`, `Aria`, `Jason`, `Leo`, `John`), 3 regional Spanish accents (Castilian `/θ/`, Latin American `/s/`, Rioplatense `/ʃ/` via native inline IPA), cached unconditional CFG prefill, Accelerate AMX `cblas_sgemm` LocalTransformer, chunked NanoCodec v3 (22.05 kHz), pitch-preserving 70–140 WPM `AVAudioUnitTimePitch`, SHA-256 disk WAV cache. |
| **Learner Phoneme STT (On-Device)** | **ZIPA CR Large 300M FP16** (`Core ML`) | iOS 18+ Multifunction `MLProgram` with 4 static Apple Neural Engine buckets (`256, 512, 1024, 1536` frames) sharing a single FP16 weight blob (~600 MB), Accelerate `vDSP`/`cblas_sgemm` Kaldi 80-bin Fbank, SIMD `vDSP_maxvi` CTC greedy decoder, articulatory IPA alignment. |
| **Learner Lexical STT (On-Device)** | **Parakeet Ultra 600M INT8** (`FluidAudio` v0.18.4 `Core ML`) | INT8 linear per-channel ANE encoder (>100× real-time) + Silero VAD silence trimming, executing **concurrently (`async let`)** with ZIPA on the same 16 kHz PCM capture. |
| **LLM Engine** | **OpenAI GPT-6.1 Terra** (`gpt-6.1-terra`) | OpenAI Responses API (`/v1/responses`) with strict `json_schema` structured outputs + configurable `reasoning.effort` for Contract-Enforced Passage Generation (3-rewrite + 25%-due-reduction loop), Isolated Stage B Verification, Closed-Vocab Speaking Partner, 13-min Writing Coach, and Weekly 60-Item Audit Grading. API key stored solely in device-local iOS Keychain (`kSecAttrAccessibleWhenUnlockedThisDeviceOnly`). |
| **Local Database** | **SQLite3 (WAL Mode)** | Bundled 4,900-item Spanish linguistic seed (`cadencia_seed.sqlite`: 4,000 active lemmas, 400 non-cognate hold-out lemmas, 200 multi-word chunks, 300 Instituto Cervantes A1–B2 grammar patterns, 64,397 inflections, 120 pseudowords, 30 Ortega Spanish EIT sentences, 4 native-speed listening tests, 3 serial story genres) + strict **append-only `evidence_events`** protected by `BEFORE UPDATE`/`BEFORE DELETE` abort triggers. |
| **Graphics & UX** | **Metal + SwiftUI Liquid Glass** | 120Hz `MediterraneanAuroraView`, raymarched iridescent `ResonanceOrbView`, GPU-instanced 4,500-star `MemoryConstellationMetalView`, iOS 26/27 `.glassEffect`/`GlassEffectContainer`, CoreHaptics choreography. |

---

## Quick Start: Build on GitHub Actions & Deploy to iPhone 18 Pro Max (Free Apple Account)

### 1. Push to GitHub to Build Unsigned `.ipa`, Precompile ZIPA FP16, & Run Tests
GitHub Actions workflow [`.github/workflows/build-ios.yml`](.github/workflows/build-ios.yml) runs on `macos-26` (Apple Silicon, Xcode 26+):
- Runs all unit tests in `CadenciaCoreTests` (`swift test --parallel`)
- Converts `anyspeech/zipa-large-crctc-ns-800k` to FP16 Multifunction Core ML and compiles it with `xcrun coremlcompiler compile` into `ZipaCRLarge300M_FP16.mlmodelc` (cached across runs!)
- Publishes `ZipaCRLarge300M_FP16.mlmodelc.zip` to the `models-zipa-fp16` GitHub Release **and** embeds the precompiled `.mlmodelc` directly inside `Cadencia.app` by default (zero download and zero computer transfer required on your iPhone!)
- Builds an unsigned `arm64` iOS app and uploads `Cadencia-unsigned-ipa` as a workflow artifact

### 2. One-Time Setup on Your Arch Linux Machine (`xtool`)
```bash
./scripts/setup-xtool.sh
xtool auth   # Sign in locally with your free Apple ID
```

### 3. Sideload onto Your USB-Connected iPhone 18 Pro Max
```bash
# Authenticate GitHub CLI once if downloading CI artifacts automatically:
gh auth login

# Pull latest CI unsigned .ipa, sign with your free Apple ID, and install via USB:
./scripts/deploy-iphone.sh
```
On your iPhone: **Settings -> General -> VPN & Device Management -> Trust** your Apple ID certificate.

### 4. Launch Cadencia on Your iPhone (Zero Computer Required)
1. Open **Cadencia** on your iPhone 18 Pro Max and enter your OpenAI API key during onboarding (or later in **Settings -> GPT-6.1 Terra Keychain Vault**). Your key is encrypted in the device Keychain and never leaves the device except in OpenAI API headers.
2. **ZIPA CR Large 300M (FP16)** is already built into the app bundle and shows **Built-in · ANE Ready** immediately (or downloads in 1 tap from your GitHub Release CDN if you choose a lightweight build).
3. In **Settings -> On-Device Neural Engine Models**, tap **Install** on **NVIDIA Magpie v2607 TTS** and **Parakeet Ultra 600M (INT8)** to download both directly onto your iPhone from HuggingFace.

---

## Reproducible Local Tooling

- **Rebuild 4,900-item Spanish SQLite seed**:
  ```bash
  .venv/bin/python tools/build_item_bank.py
  ```
- **Convert ZIPA CR Large 300M to Multifunction FP16 Core ML**:
  ```bash
  .venv-coreml/bin/python tools/convert_zipa_coreml.py
  ```
- **NVIDIA Magpie v2607 Core ML mirror / `.nemo` exporter**:
  ```bash
  .venv/bin/python tools/convert_magpie_v2607_coreml.py --fetch-prebuilt
  ```
- **Run Stage 0 Bench Test CLI** (up to 1,000 passages across simulated hours 2, 10, 20, 35, 50):
  ```bash
  export OPENAI_API_KEY="sk-..."
  .venv/bin/python tools/bench_test.py --passages-per-hour 10
  ```
- **Run full local system verification suite**:
  ```bash
  .venv-coreml/bin/python tools/verify_system.py
  ```
