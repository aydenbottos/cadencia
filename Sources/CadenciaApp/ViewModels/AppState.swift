import CadenciaCore
import CadenciaSpeech
import Foundation
import Observation
import SwiftUI

/// Top-level @Observable state container owning the local SQLite database, linguistic engines,
/// GPT-6.1 Terra client, Magpie v2607 TTS engine, Parakeet Ultra + ZIPA dual STT recognizer,
/// and GPU Memory Constellation coordinates.
@MainActor
@Observable
public final class AppState {
    // Core services
    public let database: CadenciaDatabase
    public let terraClient: GPT61TerraClient
    public let assetManager: ModelAssetManager
    public let magpieEngine: MagpieV2607Engine
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
    public var hasOpenAIKey: Bool = false
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
        self.magpieEngine = MagpieV2607Engine()
        let zipa = ZipaPhonemeEngine()
        self.zipaEngine = zipa
        self.speechRecognizer = DualSpeechRecognizer(zipaEngine: zipa)
        self.hasOpenAIKey = KeychainVault.loadAPIKey() != nil

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

            // Wire live audio level from Magpie playback + microphone into Metal shaders
            await magpieEngine.setAudioLevelHandler { [weak self] level in
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

        // 2. Warm installed Core ML engines concurrently in background
        startBackgroundModelWarmup()
    }

    /// Awaits background loading of the 64k Spanish lemmatizer & hold-out set (called before sessions/bench tests).
    public func ensureLinguisticResourcesLoaded() async {
        await linguisticLoadTask?.value
    }

    /// Starts (or joins) clean sequential ANE warmup without blocking the UI or contending on `aned`.
    public func startBackgroundModelWarmup() {
        if modelWarmupTask != nil { return }
        isWarmingModels = true
        warmingLabel = "WARMING"
        modelWarmupTask = Task(priority: .utility) { [weak self] in
            guard let self else { return }
            await self.warmInstalledModelsInBackground()
            await MainActor.run {
                self.isWarmingModels = false
                self.warmingLabel = ""
                self.modelWarmupTask = nil
            }
        }
    }

    /// Awaits any in-flight background Core ML warmup (used before Benchmark ANE or voice sessions).
    public func ensureModelsWarmed() async {
        if let task = modelWarmupTask {
            await task.value
        } else {
            await warmInstalledModelsInBackground()
        }
    }

    /// On-demand loader for Magpie TTS so playback in Memory Ledger, Research Lab, or Sessions
    /// never silently skips audio if background warmup hasn't finished yet.
    public func ensureMagpieLoaded() async throws {
        if await magpieEngine.isLoaded { return }
        guard await assetManager.isModelInstalled(.magpieTTS) else {
            throw NSError(domain: "MagpieV2607Engine", code: 503, userInfo: [
                NSLocalizedDescriptionKey: "NVIDIA Magpie v2607 is not installed yet. Install it in Settings."
            ])
        }
        let dir = await assetManager.directory(for: .magpieTTS)
        let t0 = Date()
        try await magpieEngine.loadModels(from: dir)
        let ms = Int(Date().timeIntervalSince(t0) * 1000)
        let ver = await magpieEngine.checkpointVersion
        let fsf = await magpieEngine.frameStackingFactor
        engineRuntimeNote[.magpieTTS] = "Loaded \(ver) (\(fsf)× stack · ANE+GPU+FP32) · \(ms) ms"
    }

    /// Loads installed local Core ML models in fastest-first order (Magpie -> Parakeet ANE -> ZIPA Metal GPU)
    /// recording exact per-model load latencies or errors in `engineRuntimeNote` (never silently swallowed!),
    /// then warms optional Silero VAD in a completely detached background task.
    public func warmInstalledModelsInBackground() async {
        let magpie = self.magpieEngine
        let speech = self.speechRecognizer
        let assets = self.assetManager

        // 1. NVIDIA Magpie v2607 TTS (ANE + Metal GPU step + CPU FP32 NanoCodec) — first so tutor voice is ready immediately!
        if await assets.isModelInstalled(.magpieTTS), !(await magpie.isLoaded) {
            await MainActor.run {
                self.warmingLabel = "WARMING MAGPIE"
                self.engineRuntimeNote[.magpieTTS] = "Loading on ANE + Metal GPU + CPU…"
            }
            let dir = await assets.directory(for: .magpieTTS)
            let t0 = Date()
            do {
                try await magpie.loadModels(from: dir)
                let ms = Int(Date().timeIntervalSince(t0) * 1000)
                let ver = await magpie.checkpointVersion
                let fsf = await magpie.frameStackingFactor
                await MainActor.run { self.engineRuntimeNote[.magpieTTS] = "Loaded \(ver) (\(fsf)× stack · ANE+GPU+FP32) · \(ms) ms" }
            } catch {
                await MainActor.run { self.engineRuntimeNote[.magpieTTS] = "Load error: \(error.localizedDescription)" }
            }
        }

        // 2. Parakeet Ultra 600M INT8 (ANE)
        if await assets.isModelInstalled(.parakeetUltra), !(await speech.isParakeetReady) {
            await MainActor.run {
                self.warmingLabel = "WARMING PARAKEET"
                self.engineRuntimeNote[.parakeetUltra] = "Loading INT8 on ANE…"
            }
            let t0 = Date()
            do {
                try await speech.ensureParakeetLoaded(assetManager: assets)
                let ms = Int(Date().timeIntervalSince(t0) * 1000)
                await MainActor.run { self.engineRuntimeNote[.parakeetUltra] = "Loaded INT8 on ANE · \(ms) ms" }
            } catch {
                await MainActor.run { self.engineRuntimeNote[.parakeetUltra] = "Load error: \(error.localizedDescription)" }
            }
        }

        // 3. ZIPA CR Large 300M FP16 (Metal GPU — avoids 7,344-op ANE compiler stalls & runs parallel with Parakeet!)
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

        await refreshModelStatuses()

        // 4. Optional Silero VAD in a completely detached background task (never blocks `isWarmingModels`!)
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
        openAIKey: String
    ) async throws {
        if !openAIKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            _ = KeychainVault.saveAPIKey(openAIKey)
            await terraClient.setAPIKey(openAIKey)
            hasOpenAIKey = true
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

    public var installedMagpieCheckpointVersion: String? = nil

    public func refreshModelStatuses() async {
        let zipaBundled = await assetManager.isZipaBundledInApp()
        let magpieVer = await assetManager.installedMagpieVersion()
        self.installedMagpieCheckpointVersion = magpieVer
        for kind in ManagedModelKind.allCases {
            let installed = await assetManager.isModelInstalled(kind)
            let readyMsg: String
            if kind == .zipaPhoneme && zipaBundled {
                readyMsg = "Built-in · Metal GPU Ready"
            } else if kind == .magpieTTS, let magpieVer {
                readyMsg = magpieVer == "v2607"
                    ? "Installed · v2607 GRPO (2× stack)"
                    : "v2512 installed · Upgrade to v2607 available"
            } else {
                readyMsg = "Installed & ready on device"
            }
            modelStatus[kind] = ModelDownloadProgress(
                kind: kind,
                fractionCompleted: installed ? 1.0 : 0.0,
                statusMessage: installed ? readyMsg : "Tap Install to download on device",
                isReady: installed,
                errorMessage: nil
            )
        }
    }

    private func applyDownloadProgress(_ prog: ModelDownloadProgress) {
        if let existing = modelStatus[prog.kind] {
            // Guard against out-of-order async callbacks reverting a finished download or moving progress backwards
            if existing.isReady && !prog.isReady && prog.errorMessage == nil { return }
            if !prog.isReady && prog.errorMessage == nil && prog.fractionCompleted + 0.001 < existing.fractionCompleted {
                return
            }
        }
        modelStatus[prog.kind] = prog
    }

    public func downloadMagpie() async {
        modelStatus[.magpieTTS] = ModelDownloadProgress(
            kind: .magpieTTS, fractionCompleted: 0.01,
            statusMessage: "Starting NVIDIA Magpie v2607 download…",
            isReady: false, errorMessage: nil
        )
        do {
            try await assetManager.downloadMagpieAssets { [weak self] prog in
                Task { @MainActor in
                    let clamped = ModelDownloadProgress(
                        kind: prog.kind,
                        fractionCompleted: min(0.92, prog.fractionCompleted * 0.92),
                        statusMessage: prog.isReady ? "Loading Magpie v2607 into Neural Engine…" : prog.statusMessage,
                        isReady: false,
                        errorMessage: prog.errorMessage
                    )
                    self?.applyDownloadProgress(clamped)
                }
            }
            modelStatus[.magpieTTS] = ModelDownloadProgress(
                kind: .magpieTTS, fractionCompleted: 0.95,
                statusMessage: "Loading Magpie v2607 (ANE + Metal GPU + FP32 NanoCodec)…",
                isReady: false, errorMessage: nil
            )
            let dir = await assetManager.directory(for: .magpieTTS)
            let t0 = Date()
            try await magpieEngine.loadModels(from: dir, forceReload: true)
            let ms = Int(Date().timeIntervalSince(t0) * 1000)
            let ver = await magpieEngine.checkpointVersion
            let fsf = await magpieEngine.frameStackingFactor
            self.installedMagpieCheckpointVersion = ver
            self.engineRuntimeNote[.magpieTTS] = "Loaded \(ver) (\(fsf)× stack · ANE+GPU+FP32) · \(ms) ms"
            modelStatus[.magpieTTS] = ModelDownloadProgress(
                kind: .magpieTTS, fractionCompleted: 1.0,
                statusMessage: ver == "v2607" ? "Installed · v2607 GRPO (2× stack)" : "Installed & ready on device",
                isReady: true, errorMessage: nil
            )
        } catch {
            await refreshModelStatuses()
            modelStatus[.magpieTTS] = ModelDownloadProgress(
                kind: .magpieTTS,
                fractionCompleted: installedMagpieCheckpointVersion != nil ? 1.0 : 0.0,
                statusMessage: "Download / upgrade failed",
                isReady: installedMagpieCheckpointVersion != nil,
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
                    // Keep final ready flag gated until attached to DualSpeechRecognizer
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
            // Fast single-engine attach (~50ms) — never blocks on Silero VAD network download or ZIPA reload!
            try await speechRecognizer.attachParakeetModels(models)
            modelStatus[.parakeetUltra] = ModelDownloadProgress(
                kind: .parakeetUltra, fractionCompleted: 1.0,
                statusMessage: "Installed & ready on device",
                isReady: true, errorMessage: nil
            )
            // Quietly ensure Silero VAD is warmed in background without blocking Parakeet's UI completion
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
        Task { await controller.beginSession() }
    }

    public func finishActiveSession() async {
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
