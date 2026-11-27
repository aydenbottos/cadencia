import CadenciaCore
import CadenciaSpeech
import SwiftUI

/// System, API & Hugging Face Keychain Vaults, On-Device Core ML Model Hub
/// (Phonon-40m-es, Kokoro-82M English, Parakeet Ultra INT8, ZIPA FP16),
/// Hardware Latency Benchmark, Study Protocol Parameters, and Learner Profile Configuration.
public struct HardwareSettingsView: View {
    @Bindable var appState: AppState

    @State private var apiKeyInput: String = ""
    @State private var isTestingAPI: Bool = false
    @State private var apiTestResult: String?

    @State private var hfTokenInput: String = ""
    @State private var hfTokenStatus: String?

    @State private var isBenchmarking: Bool = false
    @State private var benchmarkOutput: String?
    @State private var lastSpanishBenchmarkClip: SynthesizedAudioClip?
    @State private var lastEnglishBenchmarkClip: SynthesizedAudioClip?
    @State private var isPlayingSpanishSample: Bool = false
    @State private var isPlayingEnglishSample: Bool = false

    @State private var selectedMode: StudyMode = .soloPilot
    @State private var selectedSerialId: String = "madrid_noir"
    @State private var topicsText: String = ""
    @State private var isParametersFrozen: Bool = false
    @State private var saveToast: String?

    public init(appState: AppState) {
        self.appState = appState
    }

    public var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerSection
                openAIKeychainVaultCard
                huggingFaceTokenVaultCard
                onDeviceCoreMLHubCard
                hardwareBenchmarkCard
                protocolParametersCard
                profileStudyModeCard
            }
            .padding(.horizontal, 18)
            .padding(.top, 16)
            .padding(.bottom, 24)
        }
        .onAppear {
            if let p = appState.profile {
                selectedMode = p.studyMode
                selectedSerialId = p.selectedSerialId
                topicsText = p.nonfictionTopics.joined(separator: ", ")
                isParametersFrozen = p.parameters.isFrozen
            }
        }
    }

    private var headerSection: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("SETTINGS & CORE ML")
                    .font(CadenciaFont.mono(11, weight: .bold))
                    .foregroundStyle(CadenciaPalette.saffronGold)
                    .tracking(1.4)
                Text("Hardware & Protocol Vault")
                    .font(CadenciaFont.displaySerif(30, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
            }
            Spacer()
        }
    }

    // MARK: - 1. OpenAI GPT-6.1 Terra Keychain Vault

    private var openAIKeychainVaultCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("GPT-6.1 Terra Keychain Vault", systemImage: "key.shield.fill")
                    .font(CadenciaFont.uiTitle(16, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                Spacer()
                Text(appState.hasOpenAIKey ? "KEY STORED" : "NOT SET")
                    .font(CadenciaFont.mono(10, weight: .bold))
                    .foregroundStyle(appState.hasOpenAIKey ? CadenciaPalette.mediterraneanJade : CadenciaPalette.terracotta)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background((appState.hasOpenAIKey ? CadenciaPalette.mediterraneanJade : CadenciaPalette.terracotta).opacity(0.18), in: Capsule())
            }

            Text("Stored exclusively in this device's hardware-encrypted iOS Keychain (`WhenUnlockedThisDeviceOnly`). Never synced to iCloud or written to SQLite.")
                .font(CadenciaFont.uiCaption(12))
                .foregroundStyle(CadenciaPalette.mutedSilver)

            SecureField("sk-proj-… (Enter OpenAI API Key)", text: $apiKeyInput)
                .textFieldStyle(.plain)
                .font(CadenciaFont.mono(13))
                .foregroundStyle(CadenciaPalette.ivoryText)
                .padding(12)
                .background(Color.black.opacity(0.32), in: RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Color.white.opacity(0.15)))

            HStack(spacing: 10) {
                Button {
                    let trimmed = apiKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { return }
                    _ = KeychainVault.saveAPIKey(trimmed)
                    Task {
                        await appState.terraClient.setAPIKey(trimmed)
                        appState.hasOpenAIKey = true
                        apiKeyInput = ""
                        apiTestResult = "Saved securely to iOS Keychain."
                    }
                    HapticDirector.shared.playStrongSuccess()
                } label: {
                    Label("Save Key", systemImage: "lock.fill")
                        .font(CadenciaFont.uiCaption(13, weight: .bold))
                        .foregroundStyle(CadenciaPalette.obsidian)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(CadenciaPalette.saffronGold, in: Capsule())
                }

                Button {
                    Task {
                        isTestingAPI = true
                        defer { isTestingAPI = false }
                        do {
                            let greeting = try await appState.terraClient.verifyConnection()
                            apiTestResult = "GPT-6.1 Terra online: «\(greeting)»"
                            HapticDirector.shared.playStrongSuccess()
                        } catch {
                            apiTestResult = error.localizedDescription
                            HapticDirector.shared.playFailure()
                        }
                    }
                } label: {
                    HStack(spacing: 6) {
                        if isTestingAPI { ProgressView().scaleEffect(0.8) }
                        Text(isTestingAPI ? "Testing…" : "Ping gpt-6.1-terra")
                            .font(CadenciaFont.uiCaption(13, weight: .semibold))
                    }
                    .foregroundStyle(CadenciaPalette.ivoryText)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .cadenciaGlassPill(tint: CadenciaPalette.bioluminescentCobalt)
                }
                .disabled(isTestingAPI)

                if appState.hasOpenAIKey {
                    Spacer()
                    Button(role: .destructive) {
                        _ = KeychainVault.deleteAPIKey()
                        Task {
                            await appState.terraClient.setAPIKey(nil)
                            appState.hasOpenAIKey = false
                            apiTestResult = "Key removed from Keychain."
                        }
                    } label: {
                        Image(systemName: "trash")
                            .foregroundStyle(CadenciaPalette.crimsonAlert)
                            .padding(10)
                            .cadenciaGlassPill(tint: CadenciaPalette.crimsonAlert)
                    }
                }
            }

            if let msg = apiTestResult {
                Text(msg)
                    .font(CadenciaFont.mono(11))
                    .foregroundStyle(msg.contains("online") || msg.contains("Saved") ? CadenciaPalette.mediterraneanJade : CadenciaPalette.terracotta)
            }
        }
        .padding(18)
        .cadenciaGlassCard(cornerRadius: 24)
    }

    // MARK: - 2. Hugging Face Token Vault (For Gated Gradium Phonon-40m-es On-Device Download)

    private var huggingFaceTokenVaultCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("Hugging Face Vault (Phonon-40m-es)", systemImage: "lock.shield.fill")
                    .font(CadenciaFont.uiTitle(16, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                Spacer()
                Text(appState.hasHuggingFaceToken ? "TOKEN STORED" : "NEEDED FOR PHONON")
                    .font(CadenciaFont.mono(10, weight: .bold))
                    .foregroundStyle(appState.hasHuggingFaceToken ? CadenciaPalette.mediterraneanJade : CadenciaPalette.saffronGold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background((appState.hasHuggingFaceToken ? CadenciaPalette.mediterraneanJade : CadenciaPalette.saffronGold).opacity(0.18), in: Capsule())
            }

            Text("Gradium Phonon-40m-es is license-gated on Hugging Face. Click Agree on the model card below, then paste your free Hugging Face Read token (`hf_…`) so your iPhone downloads and exports Phonon locally.")
                .font(CadenciaFont.uiCaption(12))
                .foregroundStyle(CadenciaPalette.mutedSilver)

            HStack(spacing: 10) {
                if let modelURL = URL(string: "https://huggingface.co/gradium/phonon-40m-es") {
                    Link(destination: modelURL) {
                        Label("1. Accept Phonon License", systemImage: "arrow.up.right.square")
                            .font(CadenciaFont.mono(11, weight: .semibold))
                            .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(CadenciaPalette.bioluminescentCobalt.opacity(0.14), in: Capsule())
                    }
                }
                if let tokenURL = URL(string: "https://huggingface.co/settings/tokens") {
                    Link(destination: tokenURL) {
                        Label("2. Get Read Token", systemImage: "key.fill")
                            .font(CadenciaFont.mono(11, weight: .semibold))
                            .foregroundStyle(CadenciaPalette.saffronGold)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(CadenciaPalette.saffronGold.opacity(0.14), in: Capsule())
                    }
                }
            }

            SecureField("hf_… (Hugging Face Read Token)", text: $hfTokenInput)
                .textFieldStyle(.plain)
                .font(CadenciaFont.mono(13))
                .foregroundStyle(CadenciaPalette.ivoryText)
                .padding(12)
                .background(Color.black.opacity(0.32), in: RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Color.white.opacity(0.15)))

            HStack(spacing: 10) {
                Button {
                    let trimmed = hfTokenInput.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { return }
                    _ = KeychainVault.saveHuggingFaceToken(trimmed)
                    appState.hasHuggingFaceToken = true
                    hfTokenInput = ""
                    hfTokenStatus = "Saved securely to iOS Keychain. You can now tap Install on Gradium Phonon-40m-es."
                    HapticDirector.shared.playStrongSuccess()
                } label: {
                    Label("Save HF Token", systemImage: "lock.fill")
                        .font(CadenciaFont.uiCaption(13, weight: .bold))
                        .foregroundStyle(CadenciaPalette.obsidian)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(CadenciaPalette.bioluminescentCobalt, in: Capsule())
                }

                if appState.hasHuggingFaceToken {
                    Spacer()
                    Button(role: .destructive) {
                        _ = KeychainVault.deleteHuggingFaceToken()
                        appState.hasHuggingFaceToken = false
                        hfTokenStatus = "Hugging Face token removed from Keychain."
                    } label: {
                        Image(systemName: "trash")
                            .foregroundStyle(CadenciaPalette.crimsonAlert)
                            .padding(10)
                            .cadenciaGlassPill(tint: CadenciaPalette.crimsonAlert)
                    }
                }
            }

            if let msg = hfTokenStatus {
                Text(msg)
                    .font(CadenciaFont.mono(11))
                    .foregroundStyle(msg.contains("Saved") ? CadenciaPalette.mediterraneanJade : CadenciaPalette.terracotta)
            }
        }
        .padding(18)
        .cadenciaGlassCard(cornerRadius: 24)
    }

    // MARK: - 3. On-Device Neural Engine Model Manager

    private var onDeviceCoreMLHubCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("On-Device Neural Engine Models", systemImage: "cpu.fill")
                    .font(CadenciaFont.uiTitle(16, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                Spacer()
                if appState.isWarmingModels {
                    HStack(spacing: 5) {
                        ProgressView().scaleEffect(0.7)
                        Text(appState.warmingLabel.isEmpty ? "WARMING" : appState.warmingLabel)
                            .font(CadenciaFont.mono(9, weight: .bold))
                            .foregroundStyle(CadenciaPalette.saffronGold)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(CadenciaPalette.saffronGold.opacity(0.14), in: Capsule())
                }
                Button {
                    Task {
                        await appState.refreshModelStatuses()
                        appState.startBackgroundModelWarmup()
                    }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                        .padding(8)
                        .cadenciaGlassPill(interactive: false)
                }
            }

            // 1. Gradium Phonon-40m-es (Native Spanish Voice)
            modelRow(kind: .phononSpanish) {
                Task { await appState.downloadPhononSpanish() }
            }

            Divider().overlay(Color.white.opacity(0.1))

            // 2. Kokoro-82M (English Conversation Tutor)
            modelRow(kind: .kokoroEnglish) {
                Task { await appState.downloadKokoroEnglish() }
            }

            Divider().overlay(Color.white.opacity(0.1))

            // 3. Parakeet Ultra 600M INT8 Core ML (FluidAudio)
            modelRow(kind: .parakeetUltra) {
                Task { await appState.downloadParakeetUltra() }
            }

            Divider().overlay(Color.white.opacity(0.1))

            // 4. ZIPA CR Large 300M FP16 Multifunction Core ML
            modelRow(kind: .zipaPhoneme) {
                Task { await appState.installZipaFromCloud() }
            }

            Divider().overlay(Color.white.opacity(0.1))

            // 5. Silero VAD status row
            let vadProg = appState.modelStatus[.sileroVAD]
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(ManagedModelKind.sileroVAD.displayName)
                        .font(CadenciaFont.uiTitle(14))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                    Text(ManagedModelKind.sileroVAD.roleDescription)
                        .font(CadenciaFont.uiCaption(11))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
                Spacer()
                Text(vadProg?.isReady == true ? "READY" : "AUTO-LOADS")
                    .font(CadenciaFont.mono(10, weight: .bold))
                    .foregroundStyle(vadProg?.isReady == true ? CadenciaPalette.mediterraneanJade : CadenciaPalette.mutedSilver)
            }
        }
        .padding(18)
        .cadenciaGlassCard(cornerRadius: 24)
    }

    private func modelRow(kind: ManagedModelKind, onInstall: @escaping () -> Void) -> some View {
        let prog = appState.modelStatus[kind]
        let isDownloading = (prog?.fractionCompleted ?? 0) > 0 && !(prog?.isReady ?? false) && prog?.errorMessage == nil
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(kind.displayName)
                        .font(CadenciaFont.uiTitle(14, weight: .semibold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                    Text(kind.roleDescription)
                        .font(CadenciaFont.uiCaption(11))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
                Spacer()
                if prog?.isReady == true {
                    let badge: String = {
                        if prog?.statusMessage.hasPrefix("Built-in") == true { return "BUILT-IN · GPU" }
                        if kind == .phononSpanish { return "ANE READY" }
                        if kind == .kokoroEnglish { return "FP32 READY" }
                        return "ANE READY"
                    }()
                    Label(badge, systemImage: "checkmark.circle.fill")
                        .font(CadenciaFont.mono(10, weight: .bold))
                        .foregroundStyle(CadenciaPalette.mediterraneanJade)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(CadenciaPalette.mediterraneanJade.opacity(0.16), in: Capsule())
                } else {
                    Button(action: onInstall) {
                        Text(isDownloading ? "\(Int((prog?.fractionCompleted ?? 0) * 100))%" : "Install (~\(kind.estimatedSizeMB)MB)")
                            .font(CadenciaFont.mono(11, weight: .bold))
                            .foregroundStyle(CadenciaPalette.obsidian)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 7)
                            .background(CadenciaPalette.bioluminescentCobalt, in: Capsule())
                    }
                    .disabled(isDownloading)
                }
            }

            if isDownloading, let prog {
                let isCompiling = prog.statusMessage.localizedCaseInsensitiveContains("compil")
                    || prog.statusMessage.localizedCaseInsensitiveContains("export")
                    || prog.statusMessage.localizedCaseInsensitiveContains("loading")
                    || prog.statusMessage.localizedCaseInsensitiveContains("optimizing")
                    || prog.statusMessage.localizedCaseInsensitiveContains("attaching")
                ProgressView(value: prog.fractionCompleted)
                    .tint(isCompiling ? CadenciaPalette.saffronGold : CadenciaPalette.bioluminescentCobalt)
                HStack(spacing: 6) {
                    if isCompiling {
                        ProgressView()
                            .scaleEffect(0.65)
                            .tint(CadenciaPalette.saffronGold)
                    }
                    Text(prog.statusMessage)
                        .font(CadenciaFont.mono(10))
                        .foregroundStyle(isCompiling ? CadenciaPalette.saffronGold : CadenciaPalette.mutedSilver)
                }
            } else if let err = prog?.errorMessage {
                Text(err)
                    .font(CadenciaFont.mono(10))
                    .foregroundStyle(CadenciaPalette.crimsonAlert)
            } else if let note = appState.engineRuntimeNote[kind] {
                Text(note)
                    .font(CadenciaFont.mono(10))
                    .foregroundStyle(note.hasPrefix("Load error") ? CadenciaPalette.crimsonAlert : CadenciaPalette.mediterraneanJade.opacity(0.9))
            }
        }
    }

    // MARK: - 4. Live On-Device ANE Hardware Benchmark

    private var hardwareBenchmarkCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("On-Device Latency Benchmark", systemImage: "gauge.with.dots.needle.67percent")
                    .font(CadenciaFont.uiTitle(16, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                Spacer()
                Button {
                    Task {
                        isBenchmarking = true
                        defer { isBenchmarking = false }
                        benchmarkOutput = "Warming installed Core ML engines…"
                        await appState.ensureModelsWarmed()

                        benchmarkOutput = "Running 3.0s concurrent Parakeet Ultra INT8 + ZIPA FP16…"
                        let stt = await appState.speechRecognizer.runHardwareBenchmark()

                        var phononSummary = "Phonon-40m-es (ES): not installed"
                        if await appState.assetManager.isModelInstalled(.phononSpanish) {
                            do {
                                benchmarkOutput = "Synthesizing native Spanish via Phonon-40m-es…"
                                try await appState.ensurePhononLoaded()
                                let t0 = Date()
                                let clip = try await appState.phononEngine.synthesizeSentence(
                                    text: "Buenos días, por qué pero ya dado suena tan natural en español.",
                                    voice: .celia,
                                    accent: .castilian
                                )
                                lastSpanishBenchmarkClip = clip
                                let ms = Date().timeIntervalSince(t0) * 1000.0
                                let rtfx = (ms > 0) ? (clip.durationSeconds / (ms / 1000.0)) : 0
                                phononSummary = String(
                                    format: "Phonon-40m-es (Celia es-ES · 24 kHz ANE): %.0f ms for %.1fs audio (%.1f× RT)",
                                    ms, clip.durationSeconds, rtfx
                                )
                            } catch {
                                phononSummary = "Phonon-40m-es error: \(error.localizedDescription)"
                            }
                        }

                        var kokoroSummary = "Kokoro-82M (EN): not installed"
                        if await appState.assetManager.isModelInstalled(.kokoroEnglish) {
                            do {
                                benchmarkOutput = "Synthesizing English tutor via Kokoro-82M…"
                                try await appState.ensureKokoroLoaded()
                                let t0 = Date()
                                let clip = try await appState.kokoroEngine.synthesizeEnglish(
                                    text: "Great job! Notice how naturally that Spanish sentence flows.",
                                    voice: KokoroEnglishEngine.defaultFemaleVoice
                                )
                                lastEnglishBenchmarkClip = clip
                                let ms = Date().timeIntervalSince(t0) * 1000.0
                                let rtfx = (ms > 0) ? (clip.durationSeconds / (ms / 1000.0)) : 0
                                kokoroSummary = String(
                                    format: "Kokoro-82M (af_heart EN · 24 kHz FP32): %.0f ms for %.1fs audio (%.1f× RT)",
                                    ms, clip.durationSeconds, rtfx
                                )
                            } catch {
                                kokoroSummary = "Kokoro-82M error: \(error.localizedDescription)"
                            }
                        }

                        benchmarkOutput = String(
                            format: "3.0s Concurrent Dual-STT -> Parakeet Ultra INT8: %.0f ms | ZIPA 300M FP16: %.0f ms\n%@\n%@",
                            stt.parakeetMs, stt.zipaMs, phononSummary, kokoroSummary
                        )
                    }
                } label: {
                    HStack(spacing: 6) {
                        if isBenchmarking { ProgressView().scaleEffect(0.75) }
                        Text(isBenchmarking ? "Running…" : "Benchmark ANE")
                            .font(CadenciaFont.mono(11, weight: .bold))
                    }
                    .foregroundStyle(CadenciaPalette.obsidian)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(CadenciaPalette.mediterraneanJade, in: Capsule())
                }
                .disabled(isBenchmarking)
            }

            Text("Measures concurrent Parakeet Ultra INT8 + ZIPA CR Large FP16 on 3.0s audio, native Spanish Phonon-40m-es speed, and English Kokoro-82M speed on your iPhone 18 Pro Max.")
                .font(CadenciaFont.uiCaption(12))
                .foregroundStyle(CadenciaPalette.mutedSilver)

            if let out = benchmarkOutput {
                Text(out)
                    .font(CadenciaFont.mono(11))
                    .foregroundStyle(CadenciaPalette.saffronGold)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
            }

            HStack(spacing: 10) {
                if let esClip = lastSpanishBenchmarkClip {
                    Button {
                        Task {
                            isPlayingSpanishSample = true
                            defer { isPlayingSpanishSample = false }
                            do {
                                try await appState.audioPlayer.play(clip: esClip, targetWPM: 140)
                            } catch {
                                benchmarkOutput = (benchmarkOutput ?? "") + "\nPlayback error: \(error.localizedDescription)"
                            }
                        }
                    } label: {
                        Label(isPlayingSpanishSample ? "Playing ES…" : "Play Phonon ES", systemImage: isPlayingSpanishSample ? "speaker.wave.3.fill" : "play.circle.fill")
                            .font(CadenciaFont.uiCaption(12, weight: .bold))
                            .foregroundStyle(CadenciaPalette.ivoryText)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 9)
                            .cadenciaGlassPill(tint: CadenciaPalette.saffronGold, interactive: false)
                    }
                    .disabled(isPlayingSpanishSample || isPlayingEnglishSample)
                }

                if let enClip = lastEnglishBenchmarkClip {
                    Button {
                        Task {
                            isPlayingEnglishSample = true
                            defer { isPlayingEnglishSample = false }
                            do {
                                try await appState.audioPlayer.play(clip: enClip, targetWPM: 140)
                            } catch {
                                benchmarkOutput = (benchmarkOutput ?? "") + "\nPlayback error: \(error.localizedDescription)"
                            }
                        }
                    } label: {
                        Label(isPlayingEnglishSample ? "Playing EN…" : "Play Kokoro EN", systemImage: isPlayingEnglishSample ? "speaker.wave.3.fill" : "play.circle.fill")
                            .font(CadenciaFont.uiCaption(12, weight: .bold))
                            .foregroundStyle(CadenciaPalette.ivoryText)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 9)
                            .cadenciaGlassPill(tint: CadenciaPalette.bioluminescentCobalt, interactive: false)
                    }
                    .disabled(isPlayingSpanishSample || isPlayingEnglishSample)
                }
            }
        }
        .padding(18)
        .cadenciaGlassCard(cornerRadius: 24)
    }

    // MARK: - 5. Protocol Parameters & Stage 1 Freeze Lock

    private var protocolParametersCard: some View {
        let p = appState.profile?.parameters ?? .protocolDefaults
        return VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Learner Model Parameters", systemImage: "slider.horizontal.3")
                    .font(CadenciaFont.uiTitle(16, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                Spacer()
                Text(p.isFrozen ? "STAGE 1 FROZEN" : "ADAPTIVE")
                    .font(CadenciaFont.mono(10, weight: .bold))
                    .foregroundStyle(p.isFrozen ? CadenciaPalette.bioluminescentCobalt : CadenciaPalette.saffronGold)
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                paramBadge("Due Threshold", "≤\(Int(p.dueRecallThreshold * 100))%")
                paramBadge("Scheduled Cap", "\(Int(p.scheduledShareFloor * 100))–\(Int(p.scheduledShareCap * 100))%")
                paramBadge("Weak Pass", "×\(String(format: "%.2f", p.weakSuccessMultiplier))")
                paramBadge("Strong Pass", "×\(String(format: "%.2f", p.strongSuccessMultiplier))")
                paramBadge("Failure", "×\(String(format: "%.2f", p.failureMultiplier))")
                paramBadge("Doubt Pace", ">\(String(format: "%.1f", p.doubtPaceMultiplier))× med.")
            }

            Toggle(isOn: $isParametersFrozen) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Freeze Protocol Constants for Stage 1 / RCT")
                        .font(CadenciaFont.uiCaption(13, weight: .semibold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                    Text("Locks multipliers & thresholds before formal capacity test (PDF p. 10).")
                        .font(CadenciaFont.uiCaption(11))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
            }
            .tint(CadenciaPalette.bioluminescentCobalt)
            .onChange(of: isParametersFrozen) { _, newVal in
                guard var prof = appState.profile else { return }
                prof.parameters.isFrozen = newVal
                appState.profile = prof
                Task { try? await appState.database.saveProfile(prof) }
            }
        }
        .padding(18)
        .cadenciaGlassCard(cornerRadius: 24)
    }

    private func paramBadge(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(CadenciaFont.uiCaption(11))
                .foregroundStyle(CadenciaPalette.mutedSilver)
            Spacer()
            Text(value)
                .font(CadenciaFont.mono(11, weight: .bold))
                .foregroundStyle(CadenciaPalette.ivoryText)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - 6. Study Mode & Serial Story Settings

    private var profileStudyModeCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Study Arm & Serial Preferences", systemImage: "book.closed.fill")
                .font(CadenciaFont.uiTitle(16, weight: .bold))
                .foregroundStyle(CadenciaPalette.ivoryText)

            Picker("Study Mode", selection: $selectedMode) {
                ForEach(StudyMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .tint(CadenciaPalette.saffronGold)

            Picker("Ongoing Serial", selection: $selectedSerialId) {
                ForEach(appState.serialGenres) { g in
                    Text("\(g.titleEs) (\(g.setting))").tag(g.id)
                }
            }
            .pickerStyle(.menu)
            .tint(CadenciaPalette.terracotta)

            VStack(alignment: .leading, spacing: 6) {
                Text("Non-Fiction Interest Topics (comma-separated)")
                    .font(CadenciaFont.uiCaption(11))
                    .foregroundStyle(CadenciaPalette.mutedSilver)
                TextField("Architecture, Science, History, Food", text: $topicsText)
                    .textFieldStyle(.plain)
                    .font(CadenciaFont.uiCaption(13))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                    .padding(10)
                    .background(Color.black.opacity(0.32), in: RoundedRectangle(cornerRadius: 10))
            }

            Button {
                guard var prof = appState.profile else { return }
                prof.studyMode = selectedMode
                prof.selectedSerialId = selectedSerialId
                prof.nonfictionTopics = topicsText
                    .split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                appState.profile = prof
                Task {
                    try? await appState.database.saveProfile(prof)
                    saveToast = "Preferences saved to local SQLite."
                }
                HapticDirector.shared.playStrongSuccess()
            } label: {
                Text("Save Study Preferences")
                    .font(CadenciaFont.uiCaption(13, weight: .bold))
                    .foregroundStyle(CadenciaPalette.obsidian)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(CadenciaPalette.ivoryText, in: Capsule())
            }

            if let toast = saveToast {
                Text(toast)
                    .font(CadenciaFont.mono(11))
                    .foregroundStyle(CadenciaPalette.mediterraneanJade)
            }
        }
        .padding(18)
        .cadenciaGlassCard(cornerRadius: 24)
    }
}
