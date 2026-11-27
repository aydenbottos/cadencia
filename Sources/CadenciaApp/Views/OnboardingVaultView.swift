import CadenciaCore
import CadenciaSpeech
import SwiftUI

/// Museum-grade Editorial Onboarding & Vault Setup:
///  Step 1: Study Protocol Mode (Solo Pilot / Group A Full / Group B Control / Stage 1 Capacity)
///  Step 2: Continuity World — pick 1 of 3 Spanish Serial Genres + Non-Fiction Topics
///  Step 3: Secure iOS Keychain OpenAI Key Vault (live GPT-6.1 Terra verification)
///  Step 4: On-Device Core ML Speech Engine Setup (Phonon-40m-es, Kokoro-82M, Parakeet Ultra INT8, ZIPA 300M FP16)
public struct OnboardingVaultView: View {
    @Bindable var appState: AppState

    @State private var step = 0
    @State private var displayName = ""
    @State private var studyMode: StudyMode = .soloPilot
    @State private var selectedSerialId = "madrid_noir"
    @State private var selectedTopics: Set<String> = ["Architecture & Cities", "Science & Neuroscience", "Food & History"]
    @State private var apiKeyInput = ""
    @State private var hfTokenInput = ""
    @State private var keyVerifyStatus: String?
    @State private var isVerifyingKey = false
    @State private var isFinishing = false

    private let availableTopics = [
        "Architecture & Cities", "Science & Neuroscience", "Food & History",
        "Film & Literature", "Technology & AI", "Nature & Oceans",
        "Music & Acoustics", "Economics & Society", "Travel & Geography",
    ]

    public var body: some View {
        ZStack {
            MediterraneanAuroraView(mood: .studio, audioEnergy: appState.liveAudioEnergy)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                headerProgress
                    .padding(.top, 16)
                    .padding(.horizontal, 24)

                ScrollView(showsIndicators: false) {
                    VStack(spacing: 24) {
                        switch step {
                        case 0: welcomeAndModeStep
                        case 1: storyAndTopicsStep
                        case 2: keyVaultStep
                        default: onDeviceModelsStep
                        }
                    }
                    .padding(.horizontal, 22)
                    .padding(.vertical, 24)
                }

                footerNavigation
                    .padding(.horizontal, 22)
                    .padding(.bottom, 20)
            }
        }
        .preferredColorScheme(.dark)
    }

    private var headerProgress: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("CADENCIA")
                    .font(CadenciaFont.mono(12, weight: .bold))
                    .tracking(3.5)
                    .foregroundStyle(CadenciaPalette.saffronGold)
                Spacer()
                Text("STEP \(step + 1) OF 4")
                    .font(CadenciaFont.mono(11, weight: .semibold))
                    .foregroundStyle(CadenciaPalette.mutedSilver)
            }
            HStack(spacing: 6) {
                ForEach(0..<4, id: \.self) { idx in
                    Capsule()
                        .fill(idx <= step ? CadenciaPalette.saffronGold : Color.white.opacity(0.14))
                        .frame(height: 3.5)
                        .animation(.spring(response: 0.4), value: step)
                }
            }
        }
    }

    // MARK: - Step 0: Study Mode

    private var welcomeAndModeStep: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Zero to B2 Spanish\nin 60 Hours.")
                    .font(CadenciaFont.displaySerif(36, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                    .lineSpacing(3)
                Text("Every passage is written dynamically from what you are about to forget—4,500 items, 15+ usable events per minute, powered on-device by Phonon-40m-es, Kokoro-82M, Parakeet Ultra INT8, ZIPA 300M FP16, and GPT-6.1 Terra.")
                    .font(.subheadline)
                    .foregroundStyle(CadenciaPalette.mutedSilver)
                    .lineSpacing(4)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("YOUR NAME")
                    .font(CadenciaFont.mono(11, weight: .semibold))
                    .foregroundStyle(CadenciaPalette.saffronGold)
                TextField("e.g. Ayden", text: $displayName)
                    .textFieldStyle(.plain)
                    .padding(14)
                    .foregroundStyle(CadenciaPalette.ivoryText)
                    .cadenciaGlassCard(cornerRadius: 14)
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("PROTOCOL MODE")
                    .font(CadenciaFont.mono(11, weight: .semibold))
                    .foregroundStyle(CadenciaPalette.saffronGold)

                ForEach(StudyMode.allCases, id: \.self) { mode in
                    Button {
                        HapticDirector.shared.playGlossTap()
                        studyMode = mode
                    } label: {
                        HStack(alignment: .top, spacing: 14) {
                            Image(systemName: studyMode == mode ? "checkmark.circle.fill" : "circle")
                                .font(.title3)
                                .foregroundStyle(studyMode == mode ? CadenciaPalette.saffronGold : CadenciaPalette.mutedSilver)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(mode.title)
                                    .font(CadenciaFont.uiTitle(16))
                                    .foregroundStyle(CadenciaPalette.ivoryText)
                                Text(mode.subtitle)
                                    .font(.caption)
                                    .foregroundStyle(CadenciaPalette.mutedSilver)
                                    .multilineTextAlignment(.leading)
                            }
                            Spacer()
                        }
                        .padding(16)
                        .cadenciaGlassCard(
                            cornerRadius: 18,
                            tint: studyMode == mode ? CadenciaPalette.saffronGold : nil,
                            interactive: true
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    // MARK: - Step 1: Continuity Serial & Topics

    private var storyAndTopicsStep: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Choose Your Serial")
                    .font(CadenciaFont.displaySerif(30, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                Text("Rule 7 Continuity: passages alternate between serialized episodes of your chosen story and vivid non-fiction on your favorite topics.")
                    .font(.subheadline)
                    .foregroundStyle(CadenciaPalette.mutedSilver)
            }

            ForEach(appState.serialGenres) { genre in
                let selected = selectedSerialId == genre.id
                Button {
                    HapticDirector.shared.playGlossTap()
                    selectedSerialId = genre.id
                } label: {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(genre.titleEn)
                                .font(CadenciaFont.displaySerif(20, weight: .semibold))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                            Spacer()
                            if selected {
                                Label("Selected", systemImage: "sparkles")
                                    .font(CadenciaFont.mono(11, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.saffronGold)
                            }
                        }
                        Text("\(genre.titleEs) · \(genre.setting)")
                            .font(CadenciaFont.mono(11))
                            .foregroundStyle(CadenciaPalette.terracotta)
                        Text(genre.premise)
                            .font(.caption)
                            .foregroundStyle(CadenciaPalette.mutedSilver)
                            .lineSpacing(3)
                            .multilineTextAlignment(.leading)
                    }
                    .padding(18)
                    .cadenciaGlassCard(cornerRadius: 20, tint: selected ? CadenciaPalette.terracotta : nil, interactive: true)
                }
                .buttonStyle(.plain)
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("NON-FICTION TOPICS (PICK 3+)")
                    .font(CadenciaFont.mono(11, weight: .semibold))
                    .foregroundStyle(CadenciaPalette.saffronGold)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 145), spacing: 10)], spacing: 10) {
                    ForEach(availableTopics, id: \.self) { topic in
                        let active = selectedTopics.contains(topic)
                        Button {
                            HapticDirector.shared.playGlossTap()
                            if active { selectedTopics.remove(topic) } else { selectedTopics.insert(topic) }
                        } label: {
                            Text(topic)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(active ? CadenciaPalette.obsidian : CadenciaPalette.ivoryText)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 10)
                                .padding(.horizontal, 12)
                                .background(active ? CadenciaPalette.saffronGold : Color.white.opacity(0.08), in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    // MARK: - Step 2: Keychain GPT-6.1 Terra Vault

    private var keyVaultStep: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Text("GPT-6.1 Terra Vault")
                    .font(CadenciaFont.displaySerif(30, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                Text("Your OpenAI API key is encrypted inside the iOS Hardware Keychain (`WhenUnlockedThisDeviceOnly`). It never leaves your device except to authenticate directly with OpenAI's `/v1/responses` endpoint.")
                    .font(.subheadline)
                    .foregroundStyle(CadenciaPalette.mutedSilver)
                    .lineSpacing(3)
            }

            VStack(alignment: .leading, spacing: 12) {
                Text("OPENAI API KEY")
                    .font(CadenciaFont.mono(11, weight: .semibold))
                    .foregroundStyle(CadenciaPalette.saffronGold)
                SecureField("sk-proj-...", text: $apiKeyInput)
                    .textFieldStyle(.plain)
                    .font(CadenciaFont.mono(14))
                    .padding(16)
                    .foregroundStyle(CadenciaPalette.ivoryText)
                    .cadenciaGlassCard(cornerRadius: 16)

                Button {
                    Task { await verifyKey() }
                } label: {
                    HStack {
                        if isVerifyingKey {
                            ProgressView().tint(CadenciaPalette.obsidian)
                        } else {
                            Image(systemName: "bolt.shield.fill")
                        }
                        Text(isVerifyingKey ? "Verifying GPT-6.1 Terra…" : "Save to Keychain & Verify Connection")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .foregroundStyle(CadenciaPalette.obsidian)
                    .background(CadenciaPalette.saffronGold, in: Capsule())
                }
                .disabled(apiKeyInput.trimmingCharacters(in: .whitespaces).isEmpty || isVerifyingKey)

                if let status = keyVerifyStatus {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(status.hasPrefix("✓") ? CadenciaPalette.mediterraneanJade : CadenciaPalette.crimsonAlert)
                        .padding(.top, 4)
                }
            }
            .padding(20)
            .cadenciaGlassCard(cornerRadius: 22)

            VStack(alignment: .leading, spacing: 10) {
                Text("HUGGING FACE READ TOKEN (FOR PHONON-40M-ES)")
                    .font(CadenciaFont.mono(11, weight: .semibold))
                    .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                Text("Optional now (you can also set this in Settings). Accept gradium/phonon-40m-es on huggingface.co and enter your Read token (`hf_…`) to download native Spanish voices on-device.")
                    .font(.caption)
                    .foregroundStyle(CadenciaPalette.mutedSilver)
                SecureField("hf_... (Hugging Face Read Token)", text: $hfTokenInput)
                    .textFieldStyle(.plain)
                    .font(CadenciaFont.mono(14))
                    .padding(14)
                    .foregroundStyle(CadenciaPalette.ivoryText)
                    .cadenciaGlassCard(cornerRadius: 14)
                    .onChange(of: hfTokenInput) { _, newVal in
                        let trimmed = newVal.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !trimmed.isEmpty {
                            _ = KeychainVault.saveHuggingFaceToken(trimmed)
                            appState.hasHuggingFaceToken = true
                        }
                    }
            }
            .padding(20)
            .cadenciaGlassCard(cornerRadius: 22)
        }
    }

    // MARK: - Step 3: On-Device Core ML Models

    private var onDeviceModelsStep: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 6) {
                Text("On-Device Speech Stack")
                    .font(CadenciaFont.displaySerif(30, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                Text("Download your neural models directly onto your iPhone 18 Pro Max Neural Engine. You can finish onboarding now and monitor downloads in Settings anytime.")
                    .font(.subheadline)
                    .foregroundStyle(CadenciaPalette.mutedSilver)
            }

            modelRow(.phononSpanish) {
                Task { await appState.downloadPhononSpanish() }
            }
            modelRow(.kokoroEnglish) {
                Task { await appState.downloadKokoroEnglish() }
            }
            modelRow(.parakeetUltra) {
                Task { await appState.downloadParakeetUltra() }
            }
            modelRow(.zipaPhoneme) {
                Task { await appState.installZipaFromCloud() }
            }
        }
    }

    private func modelRow(_ kind: ManagedModelKind, action: @escaping () -> Void) -> some View {
        let prog = appState.modelStatus[kind]
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(kind.displayName)
                        .font(CadenciaFont.uiTitle(15))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                    Text(kind.roleDescription)
                        .font(.caption2)
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
                Spacer()
                if prog?.isReady == true {
                    Label("Ready", systemImage: "checkmark.seal.fill")
                        .font(CadenciaFont.mono(11, weight: .bold))
                        .foregroundStyle(CadenciaPalette.mediterraneanJade)
                } else {
                    Button("Download (\(kind.estimatedSizeMB) MB)", action: action)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .foregroundStyle(CadenciaPalette.obsidian)
                        .background(CadenciaPalette.saffronGold, in: Capsule())
                }
            }
            if let p = prog, p.fractionCompleted > 0 && !p.isReady {
                ProgressView(value: p.fractionCompleted)
                    .tint(CadenciaPalette.saffronGold)
                Text(p.statusMessage)
                    .font(CadenciaFont.mono(10))
                    .foregroundStyle(CadenciaPalette.mutedSilver)
            }
            if let err = prog?.errorMessage {
                Text(err)
                    .font(.caption2)
                    .foregroundStyle(CadenciaPalette.crimsonAlert)
            }
        }
        .padding(16)
        .cadenciaGlassCard(cornerRadius: 18)
    }

    private var footerNavigation: some View {
        CadenciaGlassContainer(spacing: 14) {
            HStack(spacing: 12) {
                if step > 0 {
                    Button {
                        withAnimation(.spring(response: 0.45)) { step -= 1 }
                    } label: {
                        Label("Back", systemImage: "chevron.left")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(CadenciaPalette.ivoryText)
                            .padding(.horizontal, 20)
                            .padding(.vertical, 15)
                            .cadenciaGlassPill()
                    }
                }
                Button {
                    if step < 3 {
                        HapticDirector.shared.playGlossTap()
                        withAnimation(.spring(response: 0.45)) { step += 1 }
                    } else {
                        Task { await completeOnboarding() }
                    }
                } label: {
                    HStack {
                        if isFinishing { ProgressView().tint(CadenciaPalette.obsidian) }
                        Text(step < 3 ? "Continue" : "Enter Studio")
                            .font(.headline)
                        Image(systemName: step < 3 ? "arrow.right" : "sparkles")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .foregroundStyle(CadenciaPalette.obsidian)
                    .background(
                        LinearGradient(
                            colors: [CadenciaPalette.saffronGold, CadenciaPalette.terracotta],
                            startPoint: .leading,
                            endPoint: .trailing
                        ),
                        in: Capsule()
                    )
                }
                .disabled(isFinishing)
            }
        }
    }

    private func verifyKey() async {
        isVerifyingKey = true
        defer { isVerifyingKey = false }
        let trimmed = apiKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
        _ = KeychainVault.saveAPIKey(trimmed)
        await appState.terraClient.setAPIKey(trimmed)
        appState.hasOpenAIKey = true
        do {
            let greeting = try await appState.terraClient.verifyConnection()
            keyVerifyStatus = "✓ Connected to GPT-6.1 Terra: «\(greeting)»"
            HapticDirector.shared.playStrongSuccess()
        } catch {
            keyVerifyStatus = "✗ \(error.localizedDescription)"
            HapticDirector.shared.playFailure()
        }
    }

    private func completeOnboarding() async {
        isFinishing = true
        defer { isFinishing = false }
        do {
            try await appState.createProfile(
                displayName: displayName,
                studyMode: studyMode,
                serialId: selectedSerialId,
                nonfictionTopics: Array(selectedTopics),
                openAIKey: apiKeyInput,
                huggingFaceToken: hfTokenInput
            )
            HapticDirector.shared.playStatePromotion()
        } catch {
            appState.errorBanner = error.localizedDescription
        }
    }
}
