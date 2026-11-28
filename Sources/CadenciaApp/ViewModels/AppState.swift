import CadenciaCore
import CadenciaSpeech
import Foundation
import Observation
import SwiftUI

/// Top-level @Observable state container owning the local SQLite database, linguistic engines,
/// GPT-6.1 Terra client, Gradium Phonon-40m-es native Spanish TTS, Kokoro-82M English tutor TTS,
/// Parakeet Ultra + ZIPA dual STT recognizer, and GPU Memory Constellation coordinates.
@MainActor
@Observable
public final class AppState {
    // Core services
    public let database: CadenciaDatabase
    public let terraClient: GPT61TerraClient
    public let assetManager: ModelAssetManager
    public let audioPlayer: TutorAudioPlayer
    public let phononEngine: PhononSpanishEngine
    public let kokoroEngine: KokoroEnglishEngine
    public let zipaEngine: ZipaPhonemeEngine
    public let speechRecognizer: DualSpeechRecognizer
    public let analyticsEngine = ResearchAnalyticsEngine()

    // Cached linguistic datasets loaded from cadencia_seed.sqlite
    public private(set) var allSyllabusItems: [SyllabusItem] = []
    public private(set) var holdoutItems: [SyllabusItem] = []
    public private(set) var serialGenres: [SerialGenre] = []
    public private(set) var lemmatizer: SpanishLemmatizer = SpanishLemmatizer(inflections: [:])

    // Active learner state
    public var profile: LearnerProfile?
    public private(set) var learnerRecords: [String: LearnerItemRecord] = [:]
    public private(set) var sessions: [SessionSummary] = []
    public private(set) var audits: [WeeklyAuditResult] = []
    public private(set) var telemetry: HypothesisTelemetrySnapshot?
    public private(set) var constellationStars: [StarPointGPU] = []

    // Hardware & Model readiness
    private static let phononComputeModeKey = "cadencia.phonon.computeMode"
    public var hasOpenAIKey: Bool = false
    public var hasHuggingFaceToken: Bool = false
    public var phononComputeMode: PhononComputeMode = .fastCPU
    public var modelStatus: [ManagedModelKind: ModelDownloadProgress] = [:]
    public var engineRuntimeNote: [ManagedModelKind: String] = [:]
    public var liveAudioEnergy: Float = 0.0
    public var isBootstrapping: Bool = true
    public var isWarmingModels: Bool = false
    public var warmingLabel: String = ""
    public var errorBanner: String?

    // Background warmup tasks (never block app launch!)
    private var linguisticLoadTask: Task<Void, Never>?
    private var modelWarmupTask: Task<Void, Never>?

    // Active 75-minute session controller (presented full-screen when active)
    public var activeSessionController: SessionController?

    public init() {
        let db = try! CadenciaDatabase()
        self.database = db
        self.terraClient = GPT61TerraClient()
        self.assetManager = ModelAssetManager()
        self.audioPlayer = TutorAudioPlayer()
        self.phononEngine = PhononSpanishEngine()
        self.kokoroEngine = KokoroEnglishEngine()
        let zipa = ZipaPhonemeEngine()
        self.zipaEngine = zipa
        self.speechRecognizer = DualSpeechRecognizer(zipaEngine: zipa)
        self.hasOpenAIKey = KeychainVault.loadAPIKey() != nil
        self.hasHuggingFaceToken = KeychainVault.loadHuggingFaceToken() != nil
        if let savedMode = UserDefaults.standard.string(forKey: Self.phononComputeModeKey),
           let parsed = PhononComputeMode(rawValue: savedMode) {
            self.phononComputeMode = parsed
        }

        for kind in ManagedModelKind.allCases {
            modelStatus[kind] = ModelDownloadProgress(
                kind: kind,
                fractionCompleted: 0,
                statusMessage: "Checking local cache…",
                isReady: false,
                errorMessage: nil
            )
        }
    }

    /// Ultra-fast launch path (<150ms after first start): loads only what the home shell needs
    /// to paint immediately, then hydrates the 64k inflection lemmatizer and warms installed
    /// Core ML engines concurrently in background `.utility` tasks without blocking the UI.
    public func bootstrap() async {
        isBootstrapping = true

        do {
            async let itemsTask = database.loadAllSyllabusItems(includeHoldout: false)
            async let genresTask = database.loadSerialGenres()
            async let profileTask = database.loadActiveProfile()

            let (items, genres, loadedProfile) = try await (itemsTask, genresTask, profileTask)
            self.allSyllabusItems = items
            self.serialGenres = genres
            self.profile = loadedProfile

            if loadedProfile != nil {
                try await refreshLearnerData()
            }

            // Wire live audio level from unified TutorAudioPlayer (Phonon + Kokoro) & microphone into Metal shaders
            await audioPlayer.setAudioLevelHandler { [weak self] level in
                Task { @MainActor in self?.liveAudioEnergy = level }
            }
            await speechRecognizer.setTelemetryHandler { [weak self] tel in
                Task { @MainActor in self?.liveAudioEnergy = tel.rmsLevel }
            }

            await refreshModelStatuses()
        } catch {
            self.errorBanner = error.localizedDescription
        }

        // Dismiss splash IMMEDIATELY — never block initial screen on 64k inflections or Core ML ANE compilation!
        isBootstrapping = false

        // 1. Hydrate 64k Spanish lemmatizer & 400 holdout items silently in background
        let activeItems = self.allSyllabusItems
        let db = self.database
        linguisticLoadTask = Task(priority: .utility) { [weak self] in
            async let holdoutTask = try? db.loadHoldoutItems()
            async let inflectionsTask = try? db.loadInflectionDictionary()
            let (holdouts, inflections) = await (holdoutTask, inflectionsTask)
            let chunks = activeItems.filter { $0.itemType == .chunk }
            let builtLemmatizer = SpanishLemmatizer(inflections: inflections ?? [:], chunkItems: chunks)
            await MainActor.run {
                self?.holdoutItems = holdouts ?? []
                self?.lemmatizer = builtLemmatizer
            }
        }

        // 2. Warm installed Core ML engines sequentially in background
        startBackgroundModelWarmup()
    }

    /// Awaits background loading of the 64k Spanish lemmatizer & hold-out set.
    public func ensureLinguisticResourcesLoaded() async {
        await linguisticLoadTask?.value
    }

    /// Starts (or joins) fast high-priority warmup without blocking the UI or stalling on ANE compilation.
    /// Warms Phonon (CPU FP32, <0.5s) -> ZIPA (Metal GPU, ~0.5s) -> Parakeet Ultra (Metal GPU, ~1-2s).
    /// Kokoro-82M English loads on demand when entering bilingual conversations or running the benchmark.
    public func startBackgroundModelWarmup() {
        if modelWarmupTask != nil { return }
        isWarmingModels = true
        warmingLabel = "WARMING"
        modelWarmupTask = Task(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            await self.warmInstalledModelsInBackground(includeKokoro: false)
            await MainActor.run {
                self.isWarmingModels = false
                self.warmingLabel = ""
                self.modelWarmupTask = nil
            }
        }
    }

    /// Awaits any in-flight warmup and ensures all installed engines (including Kokoro) are loaded.
    public func ensureModelsWarmed(includeKokoro: Bool = true) async {
        if let task = modelWarmupTask {
            await task.value
        }
        await warmInstalledModelsInBackground(includeKokoro: includeKokoro)
    }

    /// Switches Phonon-40m-es between Fast CPU/AMX FP32 (<0.5s instant load) and ANE FP16.
    public func setPhononComputeMode(_ mode: PhononComputeMode) async {
        phononComputeMode = mode
        UserDefaults.standard.set(mode.rawValue, forKey: Self.phononComputeModeKey)
        guard await assetManager.isModelInstalled(.phononSpanish) else { return }
        let dir = await assetManager.directory(for: .phononSpanish)
        engineRuntimeNote[.phononSpanish] = "Switching Phonon to \(mode.shortHardwareLabel)…"
        let t0 = Date()
        do {
            try await phononEngine.loadModels(from: dir, computeMode: mode, forceReload: true)
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            let voices = await phononEngine.availableVoices.count
            engineRuntimeNote[.phononSpanish] = "Loaded on \(mode.shortHardwareLabel) · \(voices) voices · \(ms) ms"
        } catch {
            engineRuntimeNote[.phononSpanish] = "Load error: \(error.localizedDescription)"
        }
    }

    /// On-demand loader for Gradium Phonon-40m-es native Spanish TTS.
    public func ensurePhononLoaded() async throws {
        let mode = self.phononComputeMode
        if await phononEngine.isLoaded, await phononEngine.activeComputeMode == mode { return }
        guard await assetManager.isModelInstalled(.phononSpanish) else {
            throw NSError(domain: "PhononSpanishEngine", code: 503, userInfo: [
                NSLocalizedDescriptionKey: "Phonon-40m-es is not installed yet. Add your Hugging Face token and install it in Settings."
            ])
        }
        let dir = await assetManager.directory(for: .phononSpanish)
        let t0 = Date()
        try await phononEngine.loadModels(from: dir, computeMode: mode)
        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        let voiceCount = await phononEngine.availableVoices.count
        engineRuntimeNote[.phononSpanish] = "Loaded on \(mode.shortHardwareLabel) · \(voiceCount) voices · \(ms) ms"
    }

    /// On-demand loader for Kokoro-82M English conversation voice.
    public func ensureKokoroLoaded() async throws {
        if await kokoroEngine.isLoaded { return }
        guard await assetManager.isModelInstalled(.kokoroEnglish) else {
            throw NSError(domain: "KokoroEnglishEngine", code: 503, userInfo: [
                NSLocalizedDescriptionKey: "Kokoro-82M English Tutor is not installed yet. Install it in Settings."
            ])
        }
        let dir = await assetManager.directory(for: .kokoroEnglish)
        let t0 = Date()
        try await kokoroEngine.loadModels(from: dir)
        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        engineRuntimeNote[.kokoroEnglish] = "Loaded FP32 Vocoder (af_heart + am_michael) · \(ms) ms"
    }

    /// Loads installed local Core ML models cleanly at `.userInitiated` priority:
    /// 1. Phonon-40m-es (CPU/AMX FP32 default: <0.5s instant load, 10x+ RT, zero ANE compile wait!)
    /// 2. ZIPA CR Large 300M FP16 (Metal GPU: ~0.5s)
    /// 3. Parakeet Ultra 600M INT8 (`AsrModels.loadLocal` Metal GPU: ~1–2s, zero 5-branch ANE ladder!)
    /// 4. Kokoro-82M English (only when `includeKokoro == true`, otherwise lazy-loaded on demand)
    /// 5. Silero VAD (detached background)
    public func warmInstalledModelsInBackground(includeKokoro: Bool = false) async {
        let phonon = self.phononEngine
        let kokoro = self.kokoroEngine
        let speech = self.speechRecognizer
        let assets = self.assetManager
        let mode = self.phononComputeMode

        // 1. Gradium Phonon-40m-es (Native Spanish — default CPU/AMX FP32 loads in <0.5s!)
        if await assets.isModelInstalled(.phononSpanish),
           !(await phonon.isLoaded && (await phonon.activeComputeMode) == mode) {
            await MainActor.run {
                self.warmingLabel = "WARMING PHONON"
                self.engineRuntimeNote[.phononSpanish] = "Loading Phonon-40m-es on \(mode.shortHardwareLabel)…"
            }
            let dir = await assets.directory(for: .phononSpanish)
            let t0 = Date()
            do {
                try await phonon.loadModels(from: dir, computeMode: mode)
                let ms = Int(Date().timeIntervalSince(t0) * 1000)
                let voices = await phonon.availableVoices.count
                await MainActor.run {
                    self.engineRuntimeNote[.phononSpanish] = "Loaded on \(mode.shortHardwareLabel) · \(voices) voices · \(ms) ms"
                }
            } catch {
                await MainActor.run {
                    self.engineRuntimeNote[.phononSpanish] = "Load error: \(error.localizedDescription)"
                }
            }
        }

        // 2. ZIPA CR Large 300M FP16 (Metal GPU — fast ~0.5s load)
        if await assets.isModelInstalled(.zipaPhoneme), !(await speech.isZipaReady) {
            await MainActor.run {
                self.warmingLabel = "WARMING ZIPA"
                self.engineRuntimeNote[.zipaPhoneme] = "Loading FP16 on Metal GPU…"
            }
            let t0 = Date()
            do {
                try await speech.ensureZipaLoaded(assetManager: assets)
                let ms = Int(Date().timeIntervalSince(t0) * 1000)
                await MainActor.run { self.engineRuntimeNote[.zipaPhoneme] = "Loaded FP16 on Metal GPU · \(ms) ms" }
            } catch {
                await MainActor.run { self.engineRuntimeNote[.zipaPhoneme] = "Load error: \(error.localizedDescription)" }
            }
        }

        // 3. Parakeet Ultra 600M INT8 (Metal GPU via AsrModels.loadLocal — ~1–2s, zero ANE stall!)
        if await assets.isModelInstalled(.parakeetUltra), !(await speech.isParakeetReady) {
            await MainActor.run {
                self.warmingLabel = "WARMING PARAKEET"
                self.engineRuntimeNote[.parakeetUltra] = "Loading INT8 on Metal GPU…"
            }
            let t0 = Date()
            do {
                try await speech.ensureParakeetLoaded(assetManager: assets)
                let ms = Int(Date().timeIntervalSince(t0) * 1000)
                await MainActor.run { self.engineRuntimeNote[.parakeetUltra] = "Loaded INT8 on Metal GPU · \(ms) ms" }
            } catch {
                await MainActor.run { self.engineRuntimeNote[.parakeetUltra] = "Load error: \(error.localizedDescription)" }
            }
        }

        // 4. Kokoro-82M English Tutor (lazy-loaded on conversation/benchmark so startup stays instant)
        if includeKokoro, await assets.isModelInstalled(.kokoroEnglish), !(await kokoro.isLoaded) {
            await MainActor.run {
                self.warmingLabel = "WARMING KOKORO"
                self.engineRuntimeNote[.kokoroEnglish] = "Loading Kokoro-82M English (FP32 Vocoder)…"
            }
            let dir = await assets.directory(for: .kokoroEnglish)
            let t0 = Date()
            do {
                try await kokoro.loadModels(from: dir)
                let ms = Int(Date().timeIntervalSince(t0) * 1000)
                await MainActor.run {
                    self.engineRuntimeNote[.kokoroEnglish] = "Loaded FP32 Vocoder (af_heart + am_michael) · \(ms) ms"
                }
            } catch {
                await MainActor.run {
                    self.engineRuntimeNote[.kokoroEnglish] = "Load error: \(error.localizedDescription)"
                }
            }
        } else if !includeKokoro, await assets.isModelInstalled(.kokoroEnglish), !(await kokoro.isLoaded) {
            await MainActor.run {
                if self.engineRuntimeNote[.kokoroEnglish] == nil {
                    self.engineRuntimeNote[.kokoroEnglish] = "Ready on disk · loads on demand in conversations"
                }
            }
        }

        await refreshModelStatuses()

        // 5. Optional Silero VAD in a completely detached background task
        if !(await speech.isVADReady) {
            Task.detached(priority: .background) { [weak self] in
                await speech.ensureVadLoaded(assetManager: assets)
                await self?.refreshModelStatuses()
            }
        }
    }

    public func createProfile(
        displayName: String,
        studyMode: StudyMode,
        serialId: String,
        nonfictionTopics: [String],
        openAIKey: String,
        huggingFaceToken: String = ""
    ) async throws {
        if !openAIKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            _ = KeychainVault.saveAPIKey(openAIKey)
            await terraClient.setAPIKey(openAIKey)
            hasOpenAIKey = true
        }
        if !huggingFaceToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            _ = KeychainVault.saveHuggingFaceToken(huggingFaceToken)
            hasHuggingFaceToken = true
        }

        let groupPrefix: String = switch studyMode {
        case .groupAFullSystem: "GRP-A"
        case .groupBSchedulerOff: "GRP-B"
        case .stage1CapacityTest: "STG-1"
        case .soloPilot: "SOLO"
        }
        let codedLabel = "\(groupPrefix)-\(Int.random(in: 1000...9999))"

        let newProfile = LearnerProfile(
            learnerId: UUID().uuidString,
            codedGroupLabel: codedLabel,
            displayName: displayName.isEmpty ? "Learner" : displayName,
            studyMode: studyMode,
            selectedSerialId: serialId,
            nonfictionTopics: nonfictionTopics.isEmpty
                ? ["Architecture & Cities", "Science & Neuroscience", "Food & History", "Film & Music"]
                : nonfictionTopics,
            currentSessionNumber: 1,
            totalLoggedSeconds: 0,
            medianReadingWPM: 75,
            parameters: .protocolDefaults,
            createdAt: Date()
        )

        try await database.saveProfile(newProfile)
        try await database.initializeLearnerItems(
            learnerId: newProfile.learnerId,
            items: allSyllabusItems,
            parameters: newProfile.parameters,
            holdBackTenPercentForSoloPilot: studyMode == .soloPilot
        )
        self.profile = newProfile
        try await refreshLearnerData()
    }

    public func refreshLearnerData() async throws {
        guard let profile else { return }
        async let recsTask = database.loadLearnerItems(learnerId: profile.learnerId)
        async let sessTask = database.loadSessionSummaries(learnerId: profile.learnerId)
        async let auditTask = database.loadWeeklyAudits(learnerId: profile.learnerId)
        async let evTask = database.loadEvidenceEvents(learnerId: profile.learnerId)
        async let capTask = database.loadCapacityStats(learnerId: profile.learnerId)

        let (recs, sess, auds, events, caps) = try await (recsTask, sessTask, auditTask, evTask, capTask)
        self.learnerRecords = recs
        self.sessions = sess
        self.audits = auds
        self.telemetry = analyticsEngine.computeTelemetry(
            profile: profile,
            allItems: allSyllabusItems,
            records: recs,
            sessions: sess,
            evidenceEvents: events,
            audits: auds,
            capacityStats: caps
        )
        self.constellationStars = buildConstellationStars(items: allSyllabusItems, records: recs, params: profile.parameters)
    }

    public func refreshModelStatuses() async {
        let zipaBundled = await assetManager.isZipaBundledInApp()
        let phononVoices = await assetManager.installedPhononVoiceCount()
        for kind in ManagedModelKind.allCases {
            let installed = await assetManager.isModelInstalled(kind)
            let readyMsg: String
            switch kind {
            case .zipaPhoneme where zipaBundled:
                readyMsg = "Built-in · Metal GPU Ready"
            case .phononSpanish where installed:
                readyMsg = "Installed · \(max(1, phononVoices)) native Spanish voices"
            case .kokoroEnglish where installed:
                readyMsg = "Installed · Kokoro-82M English (FP32 Vocoder)"
            default:
                readyMsg = installed ? "Installed & ready on device" : "Tap Install to download on device"
            }
            modelStatus[kind] = ModelDownloadProgress(
                kind: kind,
                fractionCompleted: installed ? 1.0 : 0.0,
                statusMessage: readyMsg,
                isReady: installed,
                errorMessage: nil
            )
        }
    }

    private func applyDownloadProgress(_ prog: ModelDownloadProgress) {
        if let existing = modelStatus[prog.kind] {
            if existing.isReady && !prog.isReady && prog.errorMessage == nil { return }
            if !prog.isReady && prog.errorMessage == nil && prog.fractionCompleted + 0.001 < existing.fractionCompleted {
                return
            }
        }
        modelStatus[prog.kind] = prog
    }

    public func downloadPhononSpanish() async {
        guard let hfToken = KeychainVault.loadHuggingFaceToken(), !hfToken.isEmpty else {
            modelStatus[.phononSpanish] = ModelDownloadProgress(
                kind: .phononSpanish,
                fractionCompleted: 0,
                statusMessage: "Hugging Face token required",
                isReady: false,
                errorMessage: "Accept gradium/phonon-40m-es on huggingface.co and save your free Read token in the Hugging Face Vault above."
            )
            return
        }

        modelStatus[.phononSpanish] = ModelDownloadProgress(
            kind: .phononSpanish,
            fractionCompleted: 0.02,
            statusMessage: "Starting Phonon-40m-es download from Hugging Face…",
            isReady: false,
            errorMessage: nil
        )
        do {
            let bundleDir = try await assetManager.downloadPhononSpanish(hfToken: hfToken) { [weak self] prog in
                Task { @MainActor in self?.applyDownloadProgress(prog) }
            }
            let mode = self.phononComputeMode
            modelStatus[.phononSpanish] = ModelDownloadProgress(
                kind: .phononSpanish,
                fractionCompleted: 0.94,
                statusMessage: "Loading Phonon-40m-es on \(mode.shortHardwareLabel)…",
                isReady: false,
                errorMessage: nil
            )
            let t0 = Date()
            try await phononEngine.loadModels(from: bundleDir, computeMode: mode, forceReload: true)
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            let voiceCount = await phononEngine.availableVoices.count
            self.engineRuntimeNote[.phononSpanish] = "Loaded on \(mode.shortHardwareLabel) · \(voiceCount) voices · \(ms) ms"
            modelStatus[.phononSpanish] = ModelDownloadProgress(
                kind: .phononSpanish,
                fractionCompleted: 1.0,
                statusMessage: "Installed · \(voiceCount) native Spanish voices",
                isReady: true,
                errorMessage: nil
            )
        } catch {
            modelStatus[.phononSpanish] = ModelDownloadProgress(
                kind: .phononSpanish,
                fractionCompleted: 0,
                statusMessage: "Install failed",
                isReady: false,
                errorMessage: error.localizedDescription
            )
        }
    }

    public func downloadKokoroEnglish() async {
        modelStatus[.kokoroEnglish] = ModelDownloadProgress(
            kind: .kokoroEnglish,
            fractionCompleted: 0.02,
            statusMessage: "Starting Kokoro-82M English Tutor download…",
            isReady: false,
            errorMessage: nil
        )
        do {
            let dir = try await assetManager.downloadKokoroEnglish { [weak self] prog in
                Task { @MainActor in self?.applyDownloadProgress(prog) }
            }
            modelStatus[.kokoroEnglish] = ModelDownloadProgress(
                kind: .kokoroEnglish,
                fractionCompleted: 0.90,
                statusMessage: "Loading Kokoro-82M English (FP32 Vocoder + G2P + voices)…",
                isReady: false,
                errorMessage: nil
            )
            let t0 = Date()
            try await kokoroEngine.loadModels(from: dir, forceReload: true)
            await assetManager.markKokoroReady()
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            self.engineRuntimeNote[.kokoroEnglish] = "Loaded FP32 Vocoder (af_heart + am_michael) · \(ms) ms"
            modelStatus[.kokoroEnglish] = ModelDownloadProgress(
                kind: .kokoroEnglish,
                fractionCompleted: 1.0,
                statusMessage: "Installed · Kokoro-82M English (FP32 Vocoder)",
                isReady: true,
                errorMessage: nil
            )
        } catch {
            modelStatus[.kokoroEnglish] = ModelDownloadProgress(
                kind: .kokoroEnglish,
                fractionCompleted: 0,
                statusMessage: "Install failed",
                isReady: false,
                errorMessage: error.localizedDescription
            )
        }
    }

    public func downloadParakeetUltra() async {
        modelStatus[.parakeetUltra] = ModelDownloadProgress(
            kind: .parakeetUltra, fractionCompleted: 0.02,
            statusMessage: "Starting Parakeet Ultra 600M INT8 download…",
            isReady: false, errorMessage: nil
        )
        do {
            let models = try await assetManager.downloadParakeetUltra { [weak self] prog in
                Task { @MainActor in
                    let mapped = prog.isReady
                        ? ModelDownloadProgress(
                            kind: .parakeetUltra, fractionCompleted: 0.98,
                            statusMessage: "Attaching Parakeet Ultra INT8 to speech engine…",
                            isReady: false, errorMessage: nil
                        )
                        : prog
                    self?.applyDownloadProgress(mapped)
                }
            }
            try await speechRecognizer.attachParakeetModels(models)
            modelStatus[.parakeetUltra] = ModelDownloadProgress(
                kind: .parakeetUltra, fractionCompleted: 1.0,
                statusMessage: "Installed & ready on device",
                isReady: true, errorMessage: nil
            )
            let speech = self.speechRecognizer
            let assets = self.assetManager
            Task(priority: .utility) { [weak self] in
                await speech.ensureVadLoaded(assetManager: assets)
                await self?.refreshModelStatuses()
            }
        } catch {
            modelStatus[.parakeetUltra] = ModelDownloadProgress(
                kind: .parakeetUltra, fractionCompleted: 0, statusMessage: "Download failed",
                isReady: false, errorMessage: error.localizedDescription
            )
        }
    }

    public func installZipaFromCloud() async {
        modelStatus[.zipaPhoneme] = ModelDownloadProgress(
            kind: .zipaPhoneme, fractionCompleted: 0.02,
            statusMessage: "Checking ZIPA CR Large 300M FP16…",
            isReady: false, errorMessage: nil
        )
        do {
            _ = try await assetManager.installZipaFromCloud { [weak self] prog in
                Task { @MainActor in self?.applyDownloadProgress(prog) }
            }
            try await speechRecognizer.ensureZipaLoaded(assetManager: assetManager)
            await refreshModelStatuses()
        } catch {
            modelStatus[.zipaPhoneme] = ModelDownloadProgress(
                kind: .zipaPhoneme, fractionCompleted: 0, statusMessage: "Install failed",
                isReady: false, errorMessage: error.localizedDescription
            )
        }
    }

    public func startTodaySession() {
        guard let profile else { return }
        let controller = SessionController(appState: self, profile: profile)
        self.activeSessionController = controller
        controller.startSessionAsync()
    }

    public func dismissActiveSession() {
        activeSessionController?.cancelSession()
        activeSessionController = nil
        Task { try? await refreshLearnerData() }
    }

    public func finishActiveSession() async {
        activeSessionController?.cancelSession()
        activeSessionController = nil
        try? await refreshLearnerData()
    }

    public var dueNowCount: Int {
        guard let params = profile?.parameters else { return 0 }
        let eng = LearnerModelEngine(parameters: params)
        let now = Date()
        return learnerRecords.values.filter { eng.isDue(record: $0, at: now) }.count
    }

    public var currentSerialGenre: SerialGenre? {
        guard let sid = profile?.selectedSerialId else { return serialGenres.first }
        return serialGenres.first { $0.id == sid } ?? serialGenres.first
    }

    /// Maps all 4,500 active syllabus items onto a golden-angle Fibonacci spiral galaxy for the Metal Constellation shader.
    private func buildConstellationStars(
        items: [SyllabusItem],
        records: [String: LearnerItemRecord],
        params: ProtocolParameters
    ) -> [StarPointGPU] {
        let eng = LearnerModelEngine(parameters: params)
        let now = Date()
        let goldenAngle: Float = 2.39996323
        let count = Float(max(1, items.count))

        return items.enumerated().map { idx, item in
            let fi = Float(idx)
            let radius = 0.88 * sqrtf((fi + 0.5) / count)
            let theta = fi * goldenAngle
            let x = radius * cosf(theta)
            let y = radius * sinf(theta)

            let rec = records[item.id]
            let stateVal: Float = switch rec?.state ?? .new {
            case .new: 0
            case .learning: 1
            case .known: 2
            case .productive: 3
            }
            let recall = Float(rec?.predictedRecall(at: now) ?? 0.0)
            let due: Float = (rec.map { eng.isDue(record: $0, at: now) } ?? false) ? 1.0 : 0.0

            return StarPointGPU(
                pos: SIMD2<Float>(x, y),
                state: stateVal,
                recall: recall,
                isDue: due,
                phase: fi * 0.37
            )
        }
    }
}
