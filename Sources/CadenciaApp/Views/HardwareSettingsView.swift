import CadenciaCore
import CadenciaSpeech
import SwiftUI

/// System, API Keychain Vault, On-Device Core ML Model Hub, Hardware Latency Benchmark,
/// Study Protocol Parameters (with Stage 1 Freeze Lock), and Learner Profile Configuration.
public struct HardwareSettingsView: View {
    @Bindable var appState: AppState

    @State private var apiKeyInput: String = ""
    @State private var isTestingAPI: Bool = false
    @State private var apiTestResult: String?

    @State private var isBenchmarking: Bool = false
    @State private var benchmarkOutput: String?

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
                onDeviceCoreMLHubCard
                hardwareBenchmarkCard
                protocolParametersCard
                profileStudyModeCard
            }
            .padding(.horizontal, 18)
            .padding(.top, 16)
            .padding(.bottom, 120)
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

    // MARK: - 2. On-Device Neural Engine Model Manager

    private var onDeviceCoreMLHubCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("On-Device Neural Engine Models", systemImage: "cpu.fill")
                    .font(CadenciaFont.uiTitle(16, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                Spacer()
                Button {
                    Task {
                        await appState.refreshModelStatuses()
                        await appState.warmInstalledModelsInBackground()
                    }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                        .padding(8)
                        .cadenciaGlassPill()
                }
            }

            // NVIDIA Magpie v2607 Core ML
            modelRow(kind: .magpieTTS) {
                Task { await appState.downloadMagpie() }
            }

            Divider().overlay(Color.white.opacity(0.1))

            // Parakeet Ultra 600M INT8 Core ML (FluidAudio)
            modelRow(kind: .parakeetUltra) {
                Task { await appState.downloadParakeetUltra() }
            }

            Divider().overlay(Color.white.opacity(0.1))

            // ZIPA CR Large 300M FP16 Multifunction Core ML (built into app bundle or one-tap cloud download)
            modelRow(kind: .zipaPhoneme) {
                Task { await appState.installZipaFromCloud() }
            }

            Divider().overlay(Color.white.opacity(0.1))

            // Silero VAD status row
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
                    let badge = (prog?.statusMessage.hasPrefix("Built-in") == true) ? "BUILT-IN · ANE" : "ANE READY"
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
                ProgressView(value: prog.fractionCompleted)
                    .tint(CadenciaPalette.bioluminescentCobalt)
                Text(prog.statusMessage)
                    .font(CadenciaFont.mono(10))
                    .foregroundStyle(CadenciaPalette.mutedSilver)
            } else if let err = prog?.errorMessage {
                Text(err)
                    .font(CadenciaFont.mono(10))
                    .foregroundStyle(CadenciaPalette.crimsonAlert)
            }
        }
    }

    // MARK: - 3. Live On-Device ANE Hardware Benchmark

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
                        let stt = await appState.speechRecognizer.runHardwareBenchmark()
                        var ttsSummary = "Magpie not loaded"
                        if await appState.magpieEngine.isLoaded {
                            let t0 = Date()
                            if let clip = try? await appState.magpieEngine.synthesizeSentence(
                                text: "Buenos días, el cielo de Madrid brilla hoy.",
                                voice: .sofia,
                                accent: .castilian
                            ) {
                                let ms = Date().timeIntervalSince(t0) * 1000.0
                                let rtf = (ms / 1000.0) / max(0.01, clip.durationSeconds)
                                ttsSummary = String(format: "Magpie v2607: %.0f ms (RTF %.2f×)", ms, rtf)
                            }
                        }
                        benchmarkOutput = String(
                            format: "3.0s Concurrent Dual-STT -> Parakeet Ultra INT8: %.0f ms | ZIPA 300M FP16: %.0f ms\n%@",
                            stt.parakeetMs, stt.zipaMs, ttsSummary
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

            Text("Measures concurrent `async let` execution of Parakeet Ultra INT8 + ZIPA CR Large FP16 on 3.0s audio, plus Magpie v2607 RTF on your iPhone 18 Pro Max.")
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
        }
        .padding(18)
        .cadenciaGlassCard(cornerRadius: 24)
    }

    // MARK: - 4. Protocol Parameters & Stage 1 Freeze Lock

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

    // MARK: - 5. Study Mode & Serial Story Settings

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
