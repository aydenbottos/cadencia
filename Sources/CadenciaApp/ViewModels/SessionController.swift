import CadenciaCore
import CadenciaSpeech
import Foundation
import Observation
import SwiftUI

/// Orchestrates the exact 75-minute Signal-Starvation session across all 5 blocks + Closing Ritual:
///   0–3 min:   Warm-Up (20 rapid meaning checks on shakiest items from last session)
///   3–28 min:  Input 1 (Audio + synced text together, tap-to-gloss, replay, pace doubt detector, 2–3 load-bearing probes)
///   28–40 min: Output 1 (Voice Retell + 2–3 due-item follow-up questions with Parakeet Ultra + ZIPA + Phonon-40m-es + Kokoro-82M)
///   40–62 min: Input 2 (Audio-First Theater -> Probes by ear -> Curtain Reveal of synced text)
///   62–75 min: Output 2 (Open Talk / Role-Play / Week 5+ SIELE Speaking OR 13-min Writing twice/week from Week 3+)
///   Closing:   1–7 Fatigue & Enjoyment + Daily "No Other Spanish" Diary + Session Telemetry Recap!
@MainActor
@Observable
public final class SessionController {
    public let appState: AppState
    public var profile: LearnerProfile
    public let sessionId: String = UUID().uuidString
    public let startedAt: Date = Date()

    // Session timing
    public var currentBlock: SessionBlockType = .warmup
    public var blockElapsedSeconds: TimeInterval = 0
    public var totalInputSeconds: TimeInterval = 0
    public var totalOutputSeconds: TimeInterval = 0
    public var isTimerPaused: Bool = false
    private var timerTask: Task<Void, Never>?

    // Scheduler budget
    public let targetNewItemsTotal: Int
    public var newItemsIntroducedCount: Int = 0
    public let isWritingDay: Bool

    // Warm-up state (0–3 min)
    public var warmupItems: [SyllabusItem] = []
    public var warmupIndex: Int = 0
    public var warmupShowingAnswer: Bool = false

    // Input 1 & Input 2 state
    public var currentPassage: GeneratedPassage?
    public var prefetchedNextPassage: GeneratedPassage?
    public var currentAudioClip: SynthesizedAudioClip?
    public var isGeneratingPassage: Bool = false
    public var generationStatusText: String = ""
    public var isPlayingAudio: Bool = false
    public var highlightedWordIndex: Int = -1
    public var activeGrammarNote: GrammarMicroNote?
    public var showingProbes: Bool = false
    public var audioFirstTextRevealed: Bool = false
    public var selectedGlossItem: (item: SyllabusItem, contextClue: String?, surface: String)?
    public var tappedItemIdsInCurrentPassage: Set<String> = []
    public var replayedSentenceIndices: Set<Int> = []
    public var probeAnswers: [String: Int] = [:] // probeId -> selected option
    public var passageReadStartTime: Date = Date()
    public var currentWPMOverride: Int

    // Output 1 (Retell) & Output 2 (Open Talk / Writing) state
    public var orbMode: ResonanceOrbMode = .idle
    public var speakingTurns: [SpokenTurnEvaluation] = []
    public var currentPartnerPromptEs: String = ""
    public var currentPartnerReplyEn: String?
    public var currentPartnerHintEn: String?
    public var conversationTutorMode: ConversationTutorMode = .bilingualCoach
    public var activeSpeakingEngineLabel: String = "PHONON-40M-ES"
    public var isRecordingVoice: Bool = false
    public var isEvaluatingVoice: Bool = false
    public var writingPromptEs: String = ""
    public var writingDraftText: String = ""
    public var writingFeedback: WritingSubmissionFeedback?
    public var isEvaluatingWriting: Bool = false

    // Session counters & Closing ritual
    public var passagesCompleted: Int = 0
    public var runningWordsProcessed: Int = 0
    public var usableEventsThisSession: Int = 0
    public var probesCorrect: Int = 0
    public var probesTotal: Int = 0
    public var promotionsToKnown: Int = 0
    public var promotionsToProductive: Int = 0
    public var fatigueRating: Int = 3
    public var enjoymentRating: Int = 6
    public var noOtherSpanishConfirmed: Bool = true
    public var completedSummary: SessionSummary?
    public var errorMessage: String?

    private var passageCounterInSession: Int = 0
    private var introducedItemIdsThisSession: [String] = []
    private var activeBlockTask: Task<Void, Never>?
    private var playbackHighlightTask: Task<Void, Never>?
    private var prefetchTask: Task<Void, Never>?
    private var recentPassageSummaries: [String] = []
    private var isCancelledSession: Bool = false

    public init(appState: AppState, profile: LearnerProfile) {
        self.appState = appState
        self.profile = profile
        let scheduler = SessionScheduler(parameters: profile.parameters)
        self.targetNewItemsTotal = scheduler.targetNewItemsForSession(
            sessionNumber: profile.currentSessionNumber,
            studyMode: profile.studyMode,
            learnerSeed: abs(profile.learnerId.hashValue)
        )
        self.isWritingDay = scheduler.isWritingSession(sessionNumber: profile.currentSessionNumber)
        self.currentWPMOverride = WeeklyPlanSpec.forWeek(profile.currentWeek).paceWPM
    }

    /// Starts the session inside a tracked, cancellable task so tapping X during drafting cancels cleanly.
    public func startSessionAsync() {
        activeBlockTask?.cancel()
        activeBlockTask = Task { [weak self] in
            await self?.beginSession()
        }
    }

    /// Cancels all in-flight passage generation, prefetch, timers, and audio playback immediately.
    public func cancelSession() {
        isCancelledSession = true
        activeBlockTask?.cancel()
        activeBlockTask = nil
        prefetchTask?.cancel()
        prefetchTask = nil
        timerTask?.cancel()
        timerTask = nil
        playbackHighlightTask?.cancel()
        playbackHighlightTask = nil
        isGeneratingPassage = false
        isPlayingAudio = false
        orbMode = .idle
        let player = appState.audioPlayer
        Task { await player.stopPlayback() }
    }

    public func beginSession() async {
        await appState.ensureLinguisticResourcesLoaded()
        guard !Task.isCancelled && !isCancelledSession else { return }
        // Log immutable sessionStarted event
        try? await appState.database.appendEvidence(EvidenceEvent(
            learnerId: profile.learnerId,
            sessionId: sessionId,
            sessionNumber: profile.currentSessionNumber,
            eventType: .sessionStarted,
            payloadJSON: "{\"studyMode\":\"\(profile.studyMode.rawValue)\",\"targetNew\":\(targetNewItemsTotal)}"
        ))

        startPrecisionTimer()

        let scheduler = SessionScheduler(parameters: profile.parameters)
        let warm = scheduler.selectWarmupItems(
            allItems: appState.allSyllabusItems,
            records: appState.learnerRecords,
            currentSession: profile.currentSessionNumber
        )
        self.warmupItems = warm
        if warm.isEmpty {
            // Very first session (Session 1): no previous items to warm up -> go straight to Input 1!
            await transitionToBlock(.input1)
        } else {
            currentBlock = .warmup
            blockElapsedSeconds = 0
            // Start generating Passage #1 in background while the learner does 3-min Warm-Up!
            prefetchNextPassage(for: .input1)
        }
    }

    private func startPrecisionTimer() {
        timerTask?.cancel()
        timerTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard let self, !self.isTimerPaused, self.currentBlock != .closing else { continue }
                self.blockElapsedSeconds += 1
                if self.currentBlock.isInputBlock {
                    self.totalInputSeconds += 1
                } else if self.currentBlock.isOutputBlock {
                    self.totalOutputSeconds += 1
                }
            }
        }
    }

    public func transitionToBlock(_ next: SessionBlockType) async {
        await stopAudioPlayback()
        currentBlock = next
        blockElapsedSeconds = 0
        errorMessage = nil

        switch next {
        case .warmup:
            break
        case .input1, .input2:
            await loadOrGeneratePassage(for: next)
        case .output1:
            orbMode = .idle
            speakingTurns.removeAll()
            let lastTitle = currentPassage?.titleEs ?? "la historia"
            currentPartnerPromptEs = "Cuéntame con tus propias palabras qué ocurrió en «\(lastTitle)». ¿Cuál fue el momento más importante?"
            currentPartnerReplyEn = conversationTutorMode.allowsEnglish
                ? "Feel free to retell the passage in Spanish—and if you get stuck, ask me anything in English!"
                : nil
            currentPartnerHintEn = "Retell the passage you just heard in Spanish; your tutor can also coach you in English."
            await speakOpeningConversationPrompt(
                spanish: currentPartnerPromptEs,
                english: currentPartnerReplyEn,
                voice: .celia,
                accent: .castilian
            )
        case .output2:
            orbMode = .idle
            if isWritingDay {
                writingDraftText = ""
                writingFeedback = nil
                let week = profile.currentWeek
                writingPromptEs = week >= 5
                    ? "Tarea SIELE (13 min): Escribe un mensaje o texto de opinión (120–160 palabras) sobre un cambio importante en tu ciudad, explicando ventajas, inconvenientes y tu propuesta."
                    : "Práctica escrita (13 min): Escribe un relato breve o correo a un amigo describiendo lo que ocurrió hoy en la historia y qué crees que sucederá después."
            } else {
                speakingTurns.removeAll()
                let week = profile.currentWeek
                currentPartnerPromptEs = week >= 5
                    ? "Vamos a practicar formato SIELE: imagina que acabas de llegar a un piso nuevo y hay un problema con la calefacción. Explícame la situación y propón una solución."
                    : "Hablemos un rato sobre tu semana y los temas de hoy. ¿Qué es lo que más te ha llamado la atención últimamente?"
                currentPartnerReplyEn = conversationTutorMode.allowsEnglish
                    ? (week >= 5
                        ? "This is a SIELE speaking scenario. Respond in Spanish, or ask me in English if you need a phrase!"
                        : "Let's chat naturally in Spanish—and I can jump in with English hints whenever you'd like.")
                    : nil
                currentPartnerHintEn = week >= 5
                    ? "Week 5+ SIELE Oral Interaction task — speak naturally!"
                    : "Open conversation steered toward your top-2,000 productive vocabulary."
                await speakOpeningConversationPrompt(
                    spanish: currentPartnerPromptEs,
                    english: currentPartnerReplyEn,
                    voice: .leonel,
                    accent: .latinAmerican
                )
            }
        case .closing:
            timerTask?.cancel()
            prefetchTask?.cancel()
        }
    }

    // MARK: - Block 0: Warm-Up Actions (20 quick meaning checks)

    public func submitWarmupSelfCheck(knewIt: Bool) async {
        guard warmupIndex < warmupItems.count else {
            await transitionToBlock(.input1)
            return
        }
        let item = warmupItems[warmupIndex]
        await recordItemOutcome(
            item: item,
            outcome: knewIt ? .passedNormalPace : .tappedGloss,
            passageId: nil,
            wasEarOnly: false,
            isFirstMeetingGlossed: false,
            passageUnderstood: true
        )
        if knewIt { HapticDirector.shared.playStrongSuccess() } else { HapticDirector.shared.playDoubtPulse() }

        warmupShowingAnswer = false
        warmupIndex += 1
        if warmupIndex >= warmupItems.count {
            await transitionToBlock(.input1)
        }
    }

    // MARK: - Block 1 & 3: Passage Generation, Prefetch, Audio Playback & Interaction

    private func loadOrGeneratePassage(for block: SessionBlockType) async {
        selectedGlossItem = nil
        tappedItemIdsInCurrentPassage.removeAll()
        replayedSentenceIndices.removeAll()
        probeAnswers.removeAll()
        showingProbes = false
        audioFirstTextRevealed = false
        highlightedWordIndex = -1

        if let ready = prefetchedNextPassage, ready.blockType == block {
            prefetchedNextPassage = nil
            await presentPassage(ready)
            prefetchNextPassage(for: block)
            return
        }

        isGeneratingPassage = true
        generationStatusText = "Scheduling due & new items…"
        defer { isGeneratingPassage = false }

        do {
            let passage = try await generateSingleVerifiedPassage(for: block) { [weak self] status in
                Task { @MainActor in
                    guard let self, !self.isCancelledSession else { return }
                    self.generationStatusText = status
                }
            }
            guard !Task.isCancelled && !isCancelledSession else { return }
            await presentPassage(passage)
            prefetchNextPassage(for: block)
        } catch {
            guard !Task.isCancelled && !isCancelledSession else { return }
            errorMessage = error.localizedDescription
        }
    }

    private func prefetchNextPassage(for block: SessionBlockType) {
        prefetchTask?.cancel()
        prefetchTask = Task { [weak self] in
            guard let self else { return }
            if let next = try? await self.generateSingleVerifiedPassage(for: block, onProgress: nil) {
                await MainActor.run {
                    if !Task.isCancelled {
                        self.prefetchedNextPassage = next
                    }
                }
            }
        }
    }

    private func generateSingleVerifiedPassage(
        for block: SessionBlockType,
        onProgress: (@Sendable (String) -> Void)?
    ) async throws -> GeneratedPassage {
        let scheduler = SessionScheduler(parameters: profile.parameters)
        let verifier = ContractVerifier(lemmatizer: appState.lemmatizer, parameters: profile.parameters)
        let serial = appState.currentSerialGenre ?? SerialGenre(
            id: "madrid_noir", titleEs: "La Sombra del Retiro", titleEn: "The Shadow of Retiro",
            tagline: "", setting: "Madrid", premise: "", accentPalette: "", protagonist: "Clara"
        )
        let serialProg = try await appState.database.loadSerialProgress(learnerId: profile.learnerId, serialId: serial.id)
        let remainingNew = max(0, targetNewItemsTotal - newItemsIntroducedCount)

        // Load prior neighbor contexts for all currently due candidate items (Rule 4)
        let dueIds = appState.learnerRecords.values.filter { $0.meetingsCount > 0 }.map(\.itemId)
        let priorContext = try await appState.database.loadPriorContextContentLemmas(
            learnerId: profile.learnerId,
            itemIds: Array(dueIds.prefix(400))
        )

        let plan = scheduler.planNextPassage(
            profile: profile,
            blockType: block,
            passageIndexInSession: passageCounterInSession,
            newItemsRemainingInSession: remainingNew,
            allItems: appState.allSyllabusItems,
            holdoutItems: appState.holdoutItems,
            records: appState.learnerRecords,
            priorContextByItem: priorContext,
            serialGenre: serial,
            serialEpisodeNumber: serialProg.episode,
            storySummarySoFar: serialProg.summary
        )

        let (passage, _) = try await appState.terraClient.generateVerifiedPassage(
            plan: plan,
            verifier: verifier,
            priorContextByItem: priorContext,
            onProgress: onProgress
        )

        passageCounterInSession += 1
        try await appState.database.savePassage(passage, learnerId: profile.learnerId)
        if passage.format == .serialEpisode {
            try await appState.database.updateSerialProgress(
                learnerId: profile.learnerId,
                serialId: serial.id,
                episode: serialProg.episode + 1,
                summary: passage.episodeSummaryForNext
            )
        }
        return passage
    }

    private func presentPassage(_ passage: GeneratedPassage) async {
        self.currentPassage = passage
        self.currentAudioClip = nil
        self.passageReadStartTime = Date()
        self.currentWPMOverride = passage.targetWPM
        self.audioFirstTextRevealed = !passage.isAudioFirst

        // Show 20-second Grammar Micro-Note first if any assigned grammar pattern is brand-new (Rule 6)!
        let firstTimeGrammar = passage.grammarNotes.first { note in
            (appState.learnerRecords[note.grammarItemId]?.meetingsCount ?? 0) == 0
        }
        self.activeGrammarNote = firstTimeGrammar

        try? await appState.database.appendEvidence(EvidenceEvent(
            learnerId: profile.learnerId,
            sessionId: sessionId,
            sessionNumber: profile.currentSessionNumber,
            passageId: passage.id,
            eventType: .passageShown,
            payloadJSON: "{\"wordCount\":\(passage.wordCount),\"audioFirst\":\(passage.isAudioFirst),\"voice\":\"\(passage.voice.rawValue)\",\"accent\":\"\(passage.accent.rawValue)\"}"
        ))

        // Synthesize passage audio natively via on-device Gradium Phonon-40m-es!
        await synthesizeAndOptionallyPlay(passage: passage, autoPlay: firstTimeGrammar == nil)
    }

    public func dismissGrammarNoteAndPlay() async {
        if let note = activeGrammarNote {
            try? await appState.database.appendEvidence(EvidenceEvent(
                learnerId: profile.learnerId,
                sessionId: sessionId,
                sessionNumber: profile.currentSessionNumber,
                passageId: currentPassage?.id,
                itemId: note.grammarItemId,
                eventType: .grammarNoteViewed
            ))
        }
        activeGrammarNote = nil
        passageReadStartTime = Date()
        if let clip = currentAudioClip {
            await playCurrentClip(clip)
        } else if let passage = currentPassage {
            await synthesizeAndOptionallyPlay(passage: passage, autoPlay: true)
        }
    }

    private func synthesizeAndOptionallyPlay(passage: GeneratedPassage, autoPlay: Bool) async {
        try? await appState.ensurePhononLoaded()
        guard await appState.phononEngine.isLoaded else { return }
        do {
            let clip = try await appState.phononEngine.synthesizePassage(
                sentences: passage.sentences.map(\.spanish),
                voice: passage.voice,
                accent: passage.accent
            )
            self.currentAudioClip = clip
            if autoPlay {
                await playCurrentClip(clip)
            }
        } catch {
            // Keep passage readable even if audio assets are not yet downloaded
            self.errorMessage = "Audio synthesis notice: \(error.localizedDescription)"
        }
    }

    public func togglePlayPause() async {
        if isPlayingAudio {
            await stopAudioPlayback()
        } else if let clip = currentAudioClip {
            await playCurrentClip(clip)
        } else if let passage = currentPassage {
            await synthesizeAndOptionallyPlay(passage: passage, autoPlay: true)
        }
    }

    private func playCurrentClip(_ clip: SynthesizedAudioClip) async {
        await stopAudioPlayback()
        isPlayingAudio = true
        orbMode = .speaking

        let speedRatio = max(0.68, min(1.30, Double(currentWPMOverride) / 140.0))
        let timings = clip.wordTimings

        // Synchronized 60Hz Karaoke word highlighter
        playbackHighlightTask = Task { [weak self] in
            let start = Date()
            while !Task.isCancelled {
                let elapsedAudio = Date().timeIntervalSince(start) * speedRatio
                if let match = timings.last(where: { elapsedAudio >= $0.startTime }) {
                    await MainActor.run { self?.highlightedWordIndex = match.wordIndex }
                }
                if elapsedAudio > clip.durationSeconds + 0.15 { break }
                try? await Task.sleep(nanoseconds: 16_000_000)
            }
        }

        do {
            try await appState.audioPlayer.play(clip: clip, targetWPM: currentWPMOverride)
        } catch {}

        playbackHighlightTask?.cancel()
        isPlayingAudio = false
        orbMode = .idle
        highlightedWordIndex = -1
    }

    public func stopAudioPlayback() async {
        playbackHighlightTask?.cancel()
        await appState.audioPlayer.stopPlayback()
        isPlayingAudio = false
        orbMode = .idle
    }

    /// Handles tap-to-gloss on ANY Spanish word in the passage -> displays English gloss, visual cue,
    /// IPA, pronounces word natively via Phonon-40m-es, AND logs a Failure (x0.5) in the Evidence Log (PDF p. 6)!
    public func handleWordTapped(surface: String) async {
        HapticDirector.shared.playGlossTap()
        let lower = surface.lowercased().trimmingCharacters(in: CharacterSet.letters.inverted)
        let (lemma, _, _) = appState.lemmatizer.lookupLemma(lower: lower)

        if let item = appState.allSyllabusItems.first(where: { $0.spanish.lowercased() == lemma || $0.spanish.lowercased() == lower }) {
            let clue = currentPassage?.newItemGlosses.first(where: { $0.itemId == item.id })?.contextClueNote
            selectedGlossItem = (item, clue, surface)

            // Tapping a brand-new item on its first meeting is expected (meaning supplied); tapping a due/known item is a Failure (x0.5)!
            let isFirstNew = currentPassage?.newItemIds.contains(item.id) == true && (appState.learnerRecords[item.id]?.meetingsCount ?? 0) == 0
            if !isFirstNew {
                tappedItemIdsInCurrentPassage.insert(item.id)
                try? await appState.database.appendEvidence(EvidenceEvent(
                    learnerId: profile.learnerId,
                    sessionId: sessionId,
                    sessionNumber: profile.currentSessionNumber,
                    passageId: currentPassage?.id,
                    itemId: item.id,
                    eventType: .wordGlossTapped,
                    payloadJSON: "{\"surface\":\"\(surface)\",\"lemma\":\"\(lemma)\"}"
                ))
            }

            // Speak single word pronunciation immediately via Phonon-40m-es (cached -> instant!)
            if let passage = currentPassage {
                Task {
                    try? await appState.ensurePhononLoaded()
                    guard await appState.phononEngine.isLoaded else { return }
                    if let wordClip = try? await appState.phononEngine.synthesizeSentence(
                        text: surface, voice: passage.voice, accent: passage.accent
                    ) {
                        try? await appState.audioPlayer.play(clip: wordClip, targetWPM: 140)
                    }
                }
            }
        }
    }

    /// Handles sentence replay -> plays that exact sentence natively via Phonon-40m-es AND flags Doubt (PDF p. 6)!
    public func replaySentence(_ sentence: PassageSentence) async {
        HapticDirector.shared.playDoubtPulse()
        replayedSentenceIndices.insert(sentence.id)

        try? await appState.database.appendEvidence(EvidenceEvent(
            learnerId: profile.learnerId,
            sessionId: sessionId,
            sessionNumber: profile.currentSessionNumber,
            passageId: currentPassage?.id,
            eventType: .sentenceReplayed,
            payloadJSON: "{\"sentenceIndex\":\(sentence.id)}"
        ))

        if let passage = currentPassage {
            try? await appState.ensurePhononLoaded()
            guard await appState.phononEngine.isLoaded else { return }
            if let clip = try? await appState.phononEngine.synthesizeSentence(
                text: sentence.spanish, voice: passage.voice, accent: passage.accent
            ) {
                await playCurrentClip(clip)
            }
        }
    }

    public func proceedToProbes() {
        showingProbes = true
    }

    public func selectProbeOption(probe: PassageProbe, optionIndex: Int) async {
        guard probeAnswers[probe.id] == nil else { return }
        probeAnswers[probe.id] = optionIndex
        let isCorrect = optionIndex == probe.correctOptionIndex
        probesTotal += 1
        if isCorrect {
            probesCorrect += 1
            HapticDirector.shared.playStrongSuccess()
        } else {
            HapticDirector.shared.playFailure()
        }

        try? await appState.database.appendEvidence(EvidenceEvent(
            learnerId: profile.learnerId,
            sessionId: sessionId,
            sessionNumber: profile.currentSessionNumber,
            passageId: currentPassage?.id,
            itemId: probe.targetDueItemId,
            eventType: .probeAnswered,
            numericValue: isCorrect ? 1.0 : 0.0,
            payloadJSON: "{\"selected\":\(optionIndex),\"correct\":\(probe.correctOptionIndex)}"
        ))

        // In Input 2 Audio-First: once all probes are answered, reveal the full synced Spanish text!
        if let p = currentPassage, probeAnswers.count >= p.probes.count, p.isAudioFirst {
            withAnimation(.spring(response: 0.65, dampingFraction: 0.82)) {
                audioFirstTextRevealed = true
            }
        }
    }

    /// Commits all half-life updates, Rule 4 fresh contexts, reading-pace Doubt checks, and usable-event
    /// counts after completing a passage, then advances seamlessly to the next passage or block!
    public func completeCurrentPassageAndAdvance() async {
        guard let passage = currentPassage else { return }
        await stopAudioPlayback()

        let readSeconds = max(5.0, Date().timeIntervalSince(passageReadStartTime))
        let observedWPM = Double(passage.wordCount) / (readSeconds / 60.0)
        // PDF p. 6: reading at > 2.5× median pace (i.e. taking > 2.5× longer -> WPM < median / 2.5) = Doubt!
        let slowDoubt = observedWPM < (profile.medianReadingWPM / profile.parameters.doubtPaceMultiplier)
        // Update rolling median reading WPM smoothly
        profile.medianReadingWPM = 0.85 * profile.medianReadingWPM + 0.15 * min(220, max(50, observedWPM))

        let correctProbesInPassage = passage.probes.filter { probeAnswers[$0.id] == $0.correctOptionIndex }.count
        let passageUnderstood = passage.probes.isEmpty || (Double(correctProbesInPassage) / Double(passage.probes.count) >= 0.66)

        // Identify items inside any replayed sentence -> Doubt
        var replayedItemIds: Set<String> = []
        let itemByLemma = Dictionary(
            appState.allSyllabusItems.map { ($0.spanish.lowercased(), $0) },
            uniquingKeysWith: { first, _ in first }
        )
        for sIdx in replayedSentenceIndices where sIdx < passage.sentences.count {
            for tok in appState.lemmatizer.analyze(passage.sentences[sIdx].spanish) {
                if let it = itemByLemma[tok.lemma.lowercased()] { replayedItemIds.insert(it.id) }
            }
        }

        let probeTargetMap = Dictionary(
            passage.probes.map { ($0.targetDueItemId, probeAnswers[$0.id] == $0.correctOptionIndex) },
            uniquingKeysWith: { a, b in a && b }
        )

        // Apply half-life updates to every scheduled item (new + due + grammar)
        let scheduledIds = Array(Set(passage.newItemIds + passage.dueItemIds + passage.grammarItemIds))
        let itemMap = Dictionary(uniqueKeysWithValues: appState.allSyllabusItems.map { ($0.id, $0) })

        for itemId in scheduledIds {
            guard let item = itemMap[itemId] else { continue }
            let isFirstNew = passage.newItemIds.contains(itemId) && (appState.learnerRecords[itemId]?.meetingsCount ?? 0) == 0
            if isFirstNew {
                newItemsIntroducedCount += 1
                introducedItemIdsThisSession.append(itemId)
            }

            let outcome: LearningOutcome
            if let probeHit = probeTargetMap[itemId] {
                outcome = probeHit ? .probeCorrect : .probeWrong
            } else if tappedItemIdsInCurrentPassage.contains(itemId) {
                outcome = .tappedGloss
            } else if replayedItemIds.contains(itemId) || slowDoubt {
                outcome = .doubtReplayOrSlow
            } else {
                outcome = .passedNormalPace
            }

            await recordItemOutcome(
                item: item,
                outcome: outcome,
                passageId: passage.id,
                wasEarOnly: passage.isAudioFirst,
                isFirstMeetingGlossed: isFirstNew,
                passageUnderstood: passageUnderstood
            )
        }

        // Persist Rule 4 Fresh Context neighbor words for all tracked items in this passage
        let verifier = ContractVerifier(lemmatizer: appState.lemmatizer, parameters: profile.parameters)
        let tracked = scheduledIds.compactMap { itemMap[$0] }
        for ctx in verifier.extractSentenceContexts(passage: passage, trackedItems: tracked) {
            try? await appState.database.recordItemSentenceContext(
                learnerId: profile.learnerId,
                itemId: ctx.itemId,
                passageId: passage.id,
                sentenceText: ctx.sentenceText,
                neighborContentLemmas: ctx.neighbors
            )
        }

        passagesCompleted += 1
        runningWordsProcessed += passage.wordCount
        recentPassageSummaries.append("\(passage.titleEs): \(passage.episodeSummaryForNext)")

        try? await appState.database.appendEvidence(EvidenceEvent(
            learnerId: profile.learnerId,
            sessionId: sessionId,
            sessionNumber: profile.currentSessionNumber,
            passageId: passage.id,
            eventType: .passageCompleted,
            passageUnderstood: passageUnderstood,
            numericValue: observedWPM
        ))

        // Check block time budget (Input 1 = 25 min, Input 2 = 22 min)
        if blockElapsedSeconds >= currentBlock.durationSeconds {
            let nextBlock: SessionBlockType = (currentBlock == .input1) ? .output1 : .output2
            await transitionToBlock(nextBlock)
        } else {
            await loadOrGeneratePassage(for: currentBlock)
        }
    }

    // MARK: - Block 2 & 4: Bilingual Voice Tutor (Parakeet Ultra + ZIPA + GPT-6.1 Terra + Phonon-40m-es + Kokoro-82M)

    public func toggleVoiceRecording() async {
        if isRecordingVoice {
            await finishVoiceTurn()
        } else {
            await stopAudioPlayback()
            do {
                try await appState.speechRecognizer.startListening()
                isRecordingVoice = true
                orbMode = .listening
                HapticDirector.shared.playMicStart()
            } catch {
                errorMessage = "Microphone error: \(error.localizedDescription)"
            }
        }
    }

    private func finishVoiceTurn() async {
        isRecordingVoice = false
        isEvaluatingVoice = true
        orbMode = .thinking
        HapticDirector.shared.playMicStop()
        defer { isEvaluatingVoice = false }

        do {
            let accent = currentPassage?.accent ?? (currentBlock == .output2 ? .latinAmerican : .castilian)
            let tutorVoice: TutorVoice = currentBlock == .output2 ? .leonel : .celia

            // Runs Parakeet Ultra 600M INT8 AND ZIPA CR Large 300M FP16 CONCURRENTLY on Core ML!
            let recognition = try await appState.speechRecognizer.stopAndRecognize(
                expectedSpanishText: nil,
                accent: accent,
                assetManager: appState.assetManager
            )
            guard !recognition.transcriptEs.isEmpty else {
                orbMode = .idle
                errorMessage = "Didn't catch speech—tap the orb and speak clearly."
                return
            }

            let scheduler = SessionScheduler(parameters: profile.parameters)
            let dueItems = (currentPassage?.dueItemIds ?? []).compactMap { id in
                appState.allSyllabusItems.first(where: { $0.id == id })
            }
            let steering = scheduler.selectProductiveSteeringTargets(
                allItems: appState.allSyllabusItems,
                records: appState.learnerRecords,
                count: 8
            )
            let schedGrammar = appState.allSyllabusItems.filter {
                $0.itemType == .grammar && ($0.grammarWeek ?? 1) <= profile.currentWeek
            }.prefix(8)

            let allowedVocab = appState.allSyllabusItems.filter { item in
                guard let rec = appState.learnerRecords[item.id] else { return false }
                return rec.meetingsCount > 0
            }.map(\.spanish)

            let history: [(role: String, text: String)] = speakingTurns.flatMap { t -> [(role: String, text: String)] in
                let partnerJoined = [t.partnerReplyEn, t.partnerReplyEs].compactMap { $0 }.joined(separator: " ")
                return [("learner", t.learnerTranscript), ("partner", partnerJoined)]
            }

            let allowEnglish = conversationTutorMode.allowsEnglish
            let partnerResp = try await appState.terraClient.evaluateSpeakingTurn(
                mode: currentBlock,
                week: profile.currentWeek,
                learnerTranscript: recognition.transcriptEs,
                learnerIPA: recognition.learnerIPA,
                phonemeAccuracy: recognition.phonemeAccuracy,
                recentPassageSummary: recentPassageSummaries.last,
                targetDueItems: dueItems,
                steeringProductiveItems: steering,
                scheduledGrammarPatterns: Array(schedGrammar),
                allowedVocabularySample: allowedVocab,
                conversationHistory: history,
                allowEnglishTutor: allowEnglish
            )

            // Resolve item IDs for strong productive successes & failed grammar patterns
            let itemByLemma = Dictionary(
                appState.allSyllabusItems.map { ($0.spanish.lowercased(), $0) },
                uniquingKeysWith: { first, _ in first }
            )
            var usedIds: [String] = []
            for lem in partnerResp.correctlyUsedLemmas {
                if let item = itemByLemma[lem.lowercased()] {
                    usedIds.append(item.id)
                    await recordItemOutcome(
                        item: item,
                        outcome: .productiveCorrect,
                        passageId: currentPassage?.id,
                        wasEarOnly: true,
                        isFirstMeetingGlossed: false,
                        passageUnderstood: true
                    )
                }
            }
            for gramId in partnerResp.failedGrammarPatternIds {
                if let gItem = appState.allSyllabusItems.first(where: { $0.id == gramId }) {
                    await recordItemOutcome(
                        item: gItem,
                        outcome: .speakingPatternError,
                        passageId: currentPassage?.id,
                        wasEarOnly: true,
                        isFirstMeetingGlossed: false,
                        passageUnderstood: true
                    )
                }
            }

            let resolvedSegments = partnerResp.resolvedSpokenSegments(allowEnglish: allowEnglish)
            let eval = SpokenTurnEvaluation(
                id: UUID().uuidString,
                timestamp: Date(),
                learnerTranscript: recognition.transcriptEs,
                learnerIPA: recognition.learnerIPA,
                canonicalSpanishIPA: recognition.canonicalSpanishIPA,
                phonemeAccuracy: recognition.phonemeAccuracy,
                alignments: recognition.alignments,
                partnerRecastEs: partnerResp.recastCorrectionEs,
                partnerReplyEs: partnerResp.partnerReplyEs,
                partnerReplyEn: allowEnglish ? partnerResp.partnerReplyEn : nil,
                spokenSegments: resolvedSegments,
                correctlyUsedItemIds: usedIds,
                failedGrammarPatternIds: partnerResp.failedGrammarPatternIds,
                parakeetLatencyMs: recognition.parakeetLatencyMs,
                zipaLatencyMs: recognition.zipaLatencyMs
            )
            speakingTurns.append(eval)
            currentPartnerPromptEs = partnerResp.partnerReplyEs
            currentPartnerReplyEn = allowEnglish ? partnerResp.partnerReplyEn : nil
            currentPartnerHintEn = partnerResp.followUpPromptEnHint

            try? await appState.database.appendEvidence(EvidenceEvent(
                learnerId: profile.learnerId,
                sessionId: sessionId,
                sessionNumber: profile.currentSessionNumber,
                passageId: currentPassage?.id,
                eventType: .spokenUtterance,
                numericValue: recognition.phonemeAccuracy,
                payloadJSON: "{\"transcript\":\(jsonEscaped(recognition.transcriptEs)),\"ipa\":\(jsonEscaped(recognition.learnerIPA)),\"parakeetMs\":\(Int(recognition.parakeetLatencyMs)),\"zipaMs\":\(Int(recognition.zipaLatencyMs))}"
            ))

            await speakBilingualPartnerTurn(segments: resolvedSegments, spanishVoice: tutorVoice, accent: accent)
        } catch {
            orbMode = .idle
            errorMessage = error.localizedDescription
        }
    }

    private func speakOpeningConversationPrompt(
        spanish: String,
        english: String?,
        voice: TutorVoice,
        accent: SpanishRegionalAccent
    ) async {
        var segments: [TutorSpeechSegment] = [TutorSpeechSegment(language: .spanish, text: spanish)]
        if conversationTutorMode.allowsEnglish, let en = english, !en.isEmpty {
            segments.append(TutorSpeechSegment(language: .english, text: en))
        }
        await speakBilingualPartnerTurn(segments: segments, spanishVoice: voice, accent: accent)
    }

    public func speakBilingualPartnerTurn(
        segments: [TutorSpeechSegment],
        spanishVoice: TutorVoice,
        accent: SpanishRegionalAccent
    ) async {
        let hasSpanish = segments.contains { $0.language == .spanish }
        let hasEnglish = segments.contains { $0.language == .english }

        if hasSpanish { try? await appState.ensurePhononLoaded() }
        if hasEnglish { try? await appState.ensureKokoroLoaded() }

        let phononReady = await appState.phononEngine.isLoaded
        let kokoroReady = await appState.kokoroEngine.isLoaded
        guard (hasSpanish && phononReady) || (hasEnglish && kokoroReady) else {
            orbMode = .idle
            return
        }

        if hasSpanish && phononReady && hasEnglish && kokoroReady {
            activeSpeakingEngineLabel = "PHONON-40M-ES + KOKORO-82M"
        } else if hasEnglish && kokoroReady && !hasSpanish {
            activeSpeakingEngineLabel = "KOKORO-82M"
        } else {
            activeSpeakingEngineLabel = "PHONON-40M-ES"
        }

        orbMode = .speaking
        do {
            let clip = try await KokoroEnglishEngine.synthesizeBilingualTurn(
                segments: segments,
                spanishVoice: spanishVoice,
                accent: accent,
                spanishEngine: appState.phononEngine,
                englishEngine: appState.kokoroEngine
            )
            if !clip.samples.isEmpty {
                let pace = hasEnglish ? 140 : WeeklyPlanSpec.forWeek(profile.currentWeek).paceWPM
                try await appState.audioPlayer.play(clip: clip, targetWPM: pace)
            }
        } catch {
            errorMessage = "Tutor voice notice: \(error.localizedDescription)"
        }
        orbMode = .idle
    }

    public func speakSpanishLine(
        _ text: String,
        voice: TutorVoice = .celia,
        accent: SpanishRegionalAccent = .castilian
    ) async {
        await speakBilingualPartnerTurn(
            segments: [TutorSpeechSegment(language: .spanish, text: text)],
            spanishVoice: voice,
            accent: accent
        )
    }

    public func speakEnglishCoachLine(
        _ text: String,
        voice: String = KokoroEnglishEngine.defaultFemaleVoice
    ) async {
        try? await appState.ensureKokoroLoaded()
        guard await appState.kokoroEngine.isLoaded else {
            errorMessage = "Install Kokoro-82M (English Tutor) in Settings to hear English tutor speech."
            return
        }
        activeSpeakingEngineLabel = "KOKORO-82M"
        orbMode = .speaking
        do {
            let clip = try await appState.kokoroEngine.synthesizeEnglish(text: text, voice: voice)
            try await appState.audioPlayer.play(clip: clip, targetWPM: 140)
        } catch {
            errorMessage = "Kokoro-82M notice: \(error.localizedDescription)"
        }
        orbMode = .idle
    }

    public func replayConversationTurn(_ turn: SpokenTurnEvaluation) async {
        let accent = currentPassage?.accent ?? (currentBlock == .output2 ? .latinAmerican : .castilian)
        let tutorVoice: TutorVoice = currentBlock == .output2 ? .leonel : .celia
        let segments: [TutorSpeechSegment]
        if !turn.spokenSegments.isEmpty {
            segments = turn.spokenSegments
        } else {
            var built: [TutorSpeechSegment] = []
            if let en = turn.partnerReplyEn, !en.isEmpty {
                built.append(TutorSpeechSegment(language: .english, text: en))
            }
            built.append(TutorSpeechSegment(language: .spanish, text: turn.partnerReplyEs))
            segments = built
        }
        await speakBilingualPartnerTurn(segments: segments, spanishVoice: tutorVoice, accent: accent)
    }

    // MARK: - Block 4 (Alt): 13-Minute Writing Submission (Week 3+, Twice Weekly)

    public func submitWritingDraft() async {
        let trimmed = writingDraftText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        isEvaluatingWriting = true
        defer { isEvaluatingWriting = false }

        do {
            let scheduler = SessionScheduler(parameters: profile.parameters)
            let targets = scheduler.selectProductiveSteeringTargets(
                allItems: appState.allSyllabusItems,
                records: appState.learnerRecords,
                count: 25
            )
            let schedGrammar = appState.allSyllabusItems.filter {
                $0.itemType == .grammar && ($0.grammarWeek ?? 1) <= profile.currentWeek
            }.prefix(12)

            let fb = try await appState.terraClient.evaluateWritingSubmission(
                promptEs: writingPromptEs,
                learnerText: trimmed,
                targetItems: targets,
                scheduledGrammar: Array(schedGrammar),
                elapsedSeconds: blockElapsedSeconds
            )
            self.writingFeedback = fb

            let itemMap = Dictionary(uniqueKeysWithValues: appState.allSyllabusItems.map { ($0.id, $0) })
            for id in fb.correctlyUsedItemIds {
                if let it = itemMap[id] {
                    await recordItemOutcome(item: it, outcome: .productiveCorrect, passageId: nil, wasEarOnly: false, isFirstMeetingGlossed: false, passageUnderstood: true)
                }
            }
            for gid in fb.failedGrammarPatternIds {
                if let git = itemMap[gid] {
                    await recordItemOutcome(item: git, outcome: .speakingPatternError, passageId: nil, wasEarOnly: false, isFirstMeetingGlossed: false, passageUnderstood: true)
                }
            }

            try? await appState.database.appendEvidence(EvidenceEvent(
                learnerId: profile.learnerId,
                sessionId: sessionId,
                sessionNumber: profile.currentSessionNumber,
                eventType: .writingSubmitted,
                numericValue: Double(fb.wordCount)
            ))
            HapticDirector.shared.playStrongSuccess()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Session Closing Ritual & Persistence

    public func finalizeSession() async {
        timerTask?.cancel()
        await stopAudioPlayback()

        let inputMins = max(totalInputSeconds / 60.0, 0.01)
        let summary = SessionSummary(
            id: sessionId,
            sessionNumber: profile.currentSessionNumber,
            week: profile.currentWeek,
            studyMode: profile.studyMode,
            startedAt: startedAt,
            endedAt: Date(),
            activeInputSeconds: totalInputSeconds,
            activeOutputSeconds: totalOutputSeconds,
            passagesCompleted: passagesCompleted,
            runningWordsProcessed: runningWordsProcessed,
            usableEventsCount: usableEventsThisSession,
            usableEventsPerInputMinute: Double(usableEventsThisSession) / inputMins,
            probesCorrect: probesCorrect,
            probesTotal: probesTotal,
            newItemsIntroduced: newItemsIntroducedCount,
            itemsPromotedToKnown: promotionsToKnown,
            itemsPromotedToProductive: promotionsToProductive,
            fatigueRating: fatigueRating,
            enjoymentRating: enjoymentRating,
            noOtherSpanishConfirmed: noOtherSpanishConfirmed
        )

        do {
            try await appState.database.saveSessionSummary(summary, learnerId: profile.learnerId, targetNewItems: targetNewItemsTotal)

            // Schedule 7-day delayed 30-item retention test in Stage 1 Capacity / Solo Pilot weeks 1–2!
            if (profile.studyMode == .stage1CapacityTest || (profile.studyMode == .soloPilot && profile.currentSessionNumber <= 12)),
               !introducedItemIdsThisSession.isEmpty {
                let bucket = targetNewItemsTotal <= 45 ? 30 : (targetNewItemsTotal <= 90 ? 60 : 120)
                try? await appState.database.scheduleDelayedCapacityTest(
                    learnerId: profile.learnerId,
                    sourceSessionNumber: profile.currentSessionNumber,
                    intakeBucket: bucket,
                    itemIds: introducedItemIdsThisSession
                )
            }

            try await appState.database.appendEvidence(EvidenceEvent(
                learnerId: profile.learnerId,
                sessionId: sessionId,
                sessionNumber: profile.currentSessionNumber,
                eventType: .dailyDiaryLogged,
                numericValue: noOtherSpanishConfirmed ? 1.0 : 0.0,
                payloadJSON: "{\"fatigue\":\(fatigueRating),\"enjoyment\":\(enjoymentRating),\"noOtherSpanish\":\(noOtherSpanishConfirmed)}"
            ))

            try await appState.database.appendEvidence(EvidenceEvent(
                learnerId: profile.learnerId,
                sessionId: sessionId,
                sessionNumber: profile.currentSessionNumber,
                eventType: .sessionCompleted,
                numericValue: summary.usableEventsPerInputMinute
            ))

            profile.currentSessionNumber = min(48, profile.currentSessionNumber + 1)
            profile.totalLoggedSeconds += (totalInputSeconds + totalOutputSeconds)
            try await appState.database.saveProfile(profile)
            appState.profile = profile

            self.completedSummary = summary
            HapticDirector.shared.playStatePromotion()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Learner Model Update + Append-Only Evidence Logging

    private func recordItemOutcome(
        item: SyllabusItem,
        outcome: LearningOutcome,
        passageId: String?,
        wasEarOnly: Bool,
        isFirstMeetingGlossed: Bool,
        passageUnderstood: Bool
    ) async {
        let engine = LearnerModelEngine(parameters: profile.parameters)
        let existing = appState.learnerRecords[item.id] ?? LearnerItemRecord(
            learnerId: profile.learnerId,
            itemId: item.id,
            halfLifeDays: engine.initialHalfLifeDays(isTransparentCognate: item.isTransparentCognate)
        )
        let (updated, prevState, predBefore) = engine.apply(
            outcome: outcome,
            to: existing,
            item: item,
            sessionNumber: profile.currentSessionNumber,
            wasEarOnly: wasEarOnly
        )

        let usable = engine.isUsableEvent(
            predictedRecallBefore: predBefore,
            isFirstMeetingGlossed: isFirstMeetingGlossed,
            passageUnderstood: passageUnderstood,
            isUnseenSentence: true
        )
        if usable { usableEventsThisSession += 1 }

        if prevState != updated.state {
            if updated.state == .known {
                promotionsToKnown += 1
                HapticDirector.shared.playStatePromotion()
            } else if updated.state == .productive {
                promotionsToProductive += 1
                HapticDirector.shared.playStatePromotion()
            }
        }

        try? await appState.database.upsertLearnerItem(updated)
        try? await appState.database.appendEvidence(EvidenceEvent(
            learnerId: profile.learnerId,
            sessionId: sessionId,
            sessionNumber: profile.currentSessionNumber,
            passageId: passageId,
            itemId: item.id,
            eventType: .modelHalfLifeUpdated,
            predictedRecallBefore: predBefore,
            halfLifeBefore: existing.halfLifeDays,
            halfLifeAfter: updated.halfLifeDays,
            isFirstMeetingGlossed: isFirstMeetingGlossed,
            isUnseenSentence: true,
            passageUnderstood: passageUnderstood,
            isUsableEvent: usable,
            payloadJSON: "{\"outcome\":\"\(outcome.rawValue)\",\"state\":\"\(updated.state.rawValue)\"}"
        ))
    }

    private func jsonEscaped(_ s: String) -> String {
        let data = try? JSONEncoder().encode(s)
        return data.flatMap { String(data: $0, encoding: .utf8) } ?? "\"\""
    }
}
