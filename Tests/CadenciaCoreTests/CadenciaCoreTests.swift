import CadenciaCore
import CadenciaSpeech
import Foundation
import XCTest

final class CadenciaCoreTests: XCTestCase {

    // MARK: - 1. Bundled 4,900-Item SQLite Seed & Append-Only Trigger Integrity

    func testSeedDatabaseCountsAndAppendOnlyTriggers() async throws {
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cadencia_test_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let db = try CadenciaDatabase(databaseURL: tmpURL)
        let activeItems = try await db.loadAllSyllabusItems(includeHoldout: false)
        let holdoutItems = try await db.loadHoldoutItems()
        let inflections = try await db.loadInflectionDictionary()
        let pseudowords = try await db.loadPseudowords()
        let eitPrompts = try await db.loadEITPrompts()
        let listening = try await db.loadNativeListeningPassages()
        let genres = try await db.loadSerialGenres()

        let lemmas = activeItems.filter { $0.itemType == .lemma }
        let chunks = activeItems.filter { $0.itemType == .chunk }
        let grammar = activeItems.filter { $0.itemType == .grammar }

        XCTAssertEqual(lemmas.count, 4000, "Must contain exactly 4,000 active lemmas")
        XCTAssertEqual(chunks.count, 200, "Must contain exactly 200 multi-word chunks")
        XCTAssertEqual(grammar.count, 300, "Must contain exactly 300 Instituto Cervantes A1–B2 grammar patterns")
        XCTAssertEqual(activeItems.count, 4500, "Active syllabus must total 4,500 items")
        XCTAssertEqual(holdoutItems.count, 400, "Hold-out control bank must contain 400 non-cognate lemmas")
        XCTAssertGreaterThan(inflections.count, 50_000, "Inflection dictionary must contain >50k Spanish forms")
        XCTAssertEqual(pseudowords.count, 120, "Must contain 120 Spanish-like pseudowords")
        XCTAssertEqual(eitPrompts.count, 30, "Must contain 30 rising-length Ortega EIT sentences")
        XCTAssertEqual(listening.count, 4, "Must contain 4 native-speed listening passages (Hours 30 & 60)")
        XCTAssertEqual(genres.count, 3, "Must contain 3 serial story genres")

        // Verify zero overlap between active lemmas and hold-out lemmas
        let activeSet = Set(lemmas.map { $0.spanish.lowercased() })
        let holdoutSet = Set(holdoutItems.map { $0.spanish.lowercased() })
        XCTAssertTrue(activeSet.isDisjoint(with: holdoutSet), "Active and hold-out sets must be strictly disjoint")

        // Verify append-only evidence_events SQLite BEFORE DELETE / BEFORE UPDATE abort triggers
        try await db.appendEvidence(EvidenceEvent(
            learnerId: "test_learner",
            eventType: .sessionStarted,
            payloadJSON: "{\"test\":true}"
        ))
        let blocked = await db.attemptForbiddenEvidenceMutationForTest()
        XCTAssertTrue(blocked, "SQLite BEFORE DELETE trigger MUST reject mutating evidence_events")
    }

    // MARK: - 2. Duolingo Half-Life Curve, State Machine, Usable Event & Online MLE Refitter

    func testHalfLifeCurveAndStatePromotions() {
        let engine = LearnerModelEngine(parameters: .protocolDefaults)

        // Exact p = 2^(-\Delta / h)
        XCTAssertEqual(LearnerModelEngine.predictedRecall(deltaDays: 0, halfLifeDays: 5.0), 1.0, accuracy: 1e-6)
        XCTAssertEqual(LearnerModelEngine.predictedRecall(deltaDays: 5.0, halfLifeDays: 5.0), 0.5, accuracy: 1e-6)
        XCTAssertEqual(LearnerModelEngine.predictedRecall(deltaDays: 10.0, halfLifeDays: 5.0), 0.25, accuracy: 1e-6)

        // Cognate vs Regular initial half-lives (1 day vs 15 min)
        XCTAssertEqual(engine.initialHalfLifeDays(isTransparentCognate: true), 1.0, accuracy: 1e-6)
        XCTAssertEqual(engine.initialHalfLifeDays(isTransparentCognate: false), 15.0 / 1440.0, accuracy: 1e-6)

        let cogItem = SyllabusItem(
            id: "lem_test_cog", itemType: .lemma, rankIndex: 100, spanish: "música", englishGloss: "music",
            pos: "NOUN", cefr: .a1, ipaEs: "ˈmusika", ipaLatam: "ˈmusika",
            isTransparentCognate: true, cognateScore: 0.92, isBootstrap: true, isHoldout: false,
            isTop2000ProductiveTarget: true, zipfFrequency: 5.2, visualSymbol: "music.note"
        )
        var rec = LearnerItemRecord(learnerId: "u1", itemId: cogItem.id, halfLifeDays: 1.0)

        // 4 successes without ear-only -> must stay Learning even if h >= 7 days!
        for _ in 0..<4 {
            let (updated, _, _) = engine.apply(outcome: .probeCorrect, to: rec, item: cogItem, sessionNumber: 1, wasEarOnly: false)
            rec = updated
        }
        XCTAssertGreaterThanOrEqual(rec.halfLifeDays, 7.0)
        XCTAssertEqual(rec.state, .learning, "Item cannot become Known without >= 1 ear-alone success")

        // 1 ear-only success -> promotes to Known!
        let (knownRec, prev1, _) = engine.apply(outcome: .passedNormalPace, to: rec, item: cogItem, sessionNumber: 2, wasEarOnly: true)
        XCTAssertEqual(prev1, .learning)
        XCTAssertEqual(knownRec.state, .known)

        // 2 productive uses -> promotes to Productive!
        let (p1, _, _) = engine.apply(outcome: .productiveCorrect, to: knownRec, item: cogItem, sessionNumber: 3, wasEarOnly: true)
        XCTAssertEqual(p1.state, .known)
        let (p2, prev2, _) = engine.apply(outcome: .productiveCorrect, to: p1, item: cogItem, sessionNumber: 3, wasEarOnly: true)
        XCTAssertEqual(prev2, .known)
        XCTAssertEqual(p2.state, .productive)

        // Failure cuts half-life by x0.5 and schedules extra meeting
        let hBeforeFail = p2.halfLifeDays
        let (failed, _, _) = engine.apply(outcome: .tappedGloss, to: p2, item: cogItem, sessionNumber: 4, wasEarOnly: false)
        XCTAssertEqual(failed.halfLifeDays, hBeforeFail * 0.5, accuracy: 1e-4)
        XCTAssertTrue(failed.needsExtraMeetingThisSession)

        // 3-condition Usable Event test
        XCTAssertTrue(engine.isUsableEvent(predictedRecallBefore: 0.80, isFirstMeetingGlossed: false, passageUnderstood: true, isUnseenSentence: true))
        XCTAssertFalse(engine.isUsableEvent(predictedRecallBefore: 0.98, isFirstMeetingGlossed: false, passageUnderstood: true, isUnseenSentence: true))
        XCTAssertTrue(engine.isUsableEvent(predictedRecallBefore: 0.0, isFirstMeetingGlossed: true, passageUnderstood: true, isUnseenSentence: true))
        XCTAssertFalse(engine.isUsableEvent(predictedRecallBefore: 0.80, isFirstMeetingGlossed: false, passageUnderstood: false, isUnseenSentence: true))
        XCTAssertFalse(engine.isUsableEvent(predictedRecallBefore: 0.80, isFirstMeetingGlossed: false, passageUnderstood: true, isUnseenSentence: false))
    }

    func testOnlineMultiplierRefitter() {
        let engine = LearnerModelEngine(parameters: .protocolDefaults)
        // Generate synthetic observations where learner actually retains much faster than defaults
        var obs: [LearnerModelEngine.CalibrationObservation] = []
        for i in 0..<40 {
            obs.append(.init(
                itemId: "item_\(i)",
                deltaDays: 8.0,
                strongCount: 2,
                weakCount: 1,
                failCount: 0,
                isCognate: true,
                observedCorrect: i % 10 != 0 // 90% observed recall at 8 days
            ))
        }
        let refit = engine.refitMultipliers(observations: obs)
        XCTAssertGreaterThan(refit.strongSuccessMultiplier, 2.2, "Refitter should raise multiplier when observed recall exceeds predicted")

        // Frozen parameters must never mutate
        var frozenParams = ProtocolParameters.protocolDefaults
        frozenParams.isFrozen = true
        let frozenEngine = LearnerModelEngine(parameters: frozenParams)
        XCTAssertEqual(frozenEngine.refitMultipliers(observations: obs), frozenParams)
    }

    // MARK: - 3. Session Scheduler Across All 4 Study Modes

    func testSchedulerStudyModesAndWritingDays() {
        let sched = SessionScheduler(parameters: .protocolDefaults)

        // Sessions 1–4 are Bootstrap (60 items/session) in Group A / Group B / Solo Pilot
        for s in 1...4 {
            XCTAssertEqual(sched.targetNewItemsForSession(sessionNumber: s, studyMode: .groupAFullSystem), 60)
            XCTAssertEqual(sched.targetNewItemsForSession(sessionNumber: s, studyMode: .soloPilot), 60)
        }

        // Stage 1 Capacity Test: 12 sessions contain exactly 4x30, 4x60, 4x120
        let stage1Targets = (1...12).map { sched.targetNewItemsForSession(sessionNumber: $0, studyMode: .stage1CapacityTest, learnerSeed: 99) }
        XCTAssertEqual(stage1Targets.filter { $0 == 30 }.count, 4)
        XCTAssertEqual(stage1Targets.filter { $0 == 60 }.count, 4)
        XCTAssertEqual(stage1Targets.filter { $0 == 120 }.count, 4)

        // Writing days: never in weeks 1–2, twice weekly (day 3 & 6) from week 3+
        XCTAssertFalse(sched.isWritingSession(sessionNumber: 3))
        XCTAssertFalse(sched.isWritingSession(sessionNumber: 12))
        XCTAssertTrue(sched.isWritingSession(sessionNumber: 15)) // Week 3 day 3
        XCTAssertTrue(sched.isWritingSession(sessionNumber: 18)) // Week 3 day 6
        XCTAssertFalse(sched.isWritingSession(sessionNumber: 14))
    }

    // MARK: - 4. Spanish Lemmatizer, Regional IPA & Contract Verifier (Rules 1–4 + Holdout Ban)

    func testSpanishLemmatizerAndHoldoutBanVerifier() {
        let dict: [String: (lemma: String, pos: String)] = [
            "camino": ("camino", "NOUN"),
            "caminos": ("camino", "NOUN"),
            "puerta": ("puerta", "NOUN"),
            "ventana": ("ventana", "NOUN"),
            "hablar": ("hablar", "VERB"),
            "hablando": ("hablar", "VERB"),
            "alcornoque": ("alcornoque", "NOUN"),
        ]
        let lemmatizer = SpanishLemmatizer(inflections: dict)
        XCTAssertEqual(lemmatizer.lookupLemma(lower: "caminos").lemma, "camino")
        XCTAssertEqual(lemmatizer.lookupLemma(lower: "hablándome").lemma, "hablar")

        // Regional accent IPA contrasts (Castilian /θ/ vs LatAm /s/ vs Rioplatense /ʃ/)
        let castilian = SpanishLemmatizer.spanishToIPA("calle zapato", accent: .castilian)
        let latam = SpanishLemmatizer.spanishToIPA("calle zapato", accent: .latinAmerican)
        let rio = SpanishLemmatizer.spanishToIPA("calle zapato", accent: .rioplatense)
        XCTAssertTrue(castilian.contains("θ"), "Castilian must produce /θ/ for z/ce/ci")
        XCTAssertFalse(latam.contains("θ"), "Latin American seseo must use /s/")
        XCTAssertTrue(rio.contains("ʃ"), "Rioplatense sheísmo must produce /ʃ/ for ll/y")

        // Stage A Verifier must immediately reject any holdout word leak
        let verifier = ContractVerifier(lemmatizer: lemmatizer, parameters: .protocolDefaults)
        let newItem = SyllabusItem(
            id: "lem_puerta", itemType: .lemma, rankIndex: 20, spanish: "puerta", englishGloss: "door",
            pos: "NOUN", cefr: .a1, ipaEs: "ˈpweɾta", ipaLatam: "ˈpweɾta",
            isTransparentCognate: false, cognateScore: 0.1, isBootstrap: false, isHoldout: false,
            isTop2000ProductiveTarget: true, zipfFrequency: 5.5, visualSymbol: "door.left.hand.open"
        )
        let plan = PassagePlan(
            passageId: "p1", sessionNumber: 5, passageIndexInSession: 0, blockType: .input1,
            isBootstrap: false, isAudioFirst: false, targetWPM: 90, format: .serialEpisode,
            topicOrEpisodeTitle: "Test", storySummarySoFar: "",
            dueItems: [], newItems: [newItem], grammarPatterns: [],
            allowedKnownLemmas: ["camino", "puerta", "ventana", "hablar"],
            holdoutBannedLemmas: ["alcornoque"],
            voice: .celia, accent: .castilian, priorContextWordsByItem: [:]
        )
        let sentence = PassageSentence(id: 0, spanish: "El alcornoque está cerca de la puerta.", englishLine: "The cork oak is near the door.", wordTokens: ["El", "alcornoque", "está", "cerca", "de", "la", "puerta"])
        let report = verifier.verifyStageA(passageText: sentence.spanish, sentences: [sentence], plan: plan, priorContextByItem: [:])
        XCTAssertFalse(report.passed)
        XCTAssertTrue(report.violations.contains(where: { $0.detail.lowercased().contains("hold") || $0.offendingTokens.contains("alcornoque") }))
    }

    // MARK: - 5. Kaldi 80-Bin Fbank & Articulatory IPA Aligner (CadenciaSpeech DSP)

    func testKaldiFbankAndArticulatoryPhonemeAligner() {
        let extractor = KaldiFbankExtractor()
        // 1.0 second 440 Hz tone at 16 kHz -> expects ~100 frames x 80 bins
        let samples = (0..<16_000).map { i in Float(0.3 * sin(2.0 * .pi * 440.0 * Double(i) / 16_000.0)) }
        let (feats, numFrames) = extractor.extract(samples: samples)
        XCTAssertEqual(numFrames, 100, "1.0s at 10ms hop with snip_edges=false should yield 100 frames")
        XCTAssertEqual(feats.count, numFrames * 80)
        XCTAssertTrue(feats.allSatisfy { $0.isFinite }, "All log-mel filterbank values must be finite")

        let aligner = ArticulatoryPhonemeAligner()
        let (exactAcc, _) = aligner.align(expectedIPA: "peɾo", observedIPA: "peɾo")
        XCTAssertEqual(exactAcc, 1.0, accuracy: 1e-6)

        // Single trill vs tap substitution (/r/ vs /ɾ/) should give partial articulatory credit & coaching hint
        let (trillAcc, cells) = aligner.align(expectedIPA: "pero", observedIPA: "peɾo")
        XCTAssertLessThan(trillAcc, 1.0)
        XCTAssertGreaterThan(trillAcc, 0.7)
        XCTAssertTrue(cells.contains(where: { !$0.isExactMatch }))
    }

    // MARK: - 6. Vocabulary Size Guessing Correction & Hold-Out >15% Guard

    func testVocabSizeHoldoutGuardAndEITScoring() {
        let analytics = ResearchAnalyticsEngine()

        // Clean learner: 80% real, 5% false alarms, 5% hold-out -> NOT inflated
        let clean = analytics.scoreVocabSizeTest(
            checkpointLabel: "Hour 40",
            realWordsTested: 60, realWordsClaimed: 48,
            pseudowordsTested: 30, pseudowordsFalseAlarms: 1,
            holdoutWordsTested: 30, holdoutWordsClaimed: 2,
            typedRecall100Score: 0.78
        )
        XCTAssertFalse(clean.isInflatedWarning)
        XCTAssertGreaterThan(clean.correctedEstimatedVocabSize, 3500)

        // Inflated learner: claims 40% of never-taught hold-out words -> MUST trigger >15% guard!
        let inflated = analytics.scoreVocabSizeTest(
            checkpointLabel: "Hour 40",
            realWordsTested: 60, realWordsClaimed: 50,
            pseudowordsTested: 30, pseudowordsFalseAlarms: 1,
            holdoutWordsTested: 30, holdoutWordsClaimed: 12,
            typedRecall100Score: nil
        )
        XCTAssertTrue(inflated.isInflatedWarning)

        // Ortega EIT 0–4 scoring
        let prompt = EITPrompt(id: 1, syllables: 7, cefr: .a1, spanish: "Quiero cortarme el pelo", english: "I want to cut my hair", ipaEs: "", ipaLatam: "")
        let exact = analytics.scoreEITResponse(prompt: prompt, learnerTranscript: "Quiero cortarme el pelo", learnerIPA: "kjeɾo koɾtaɾme el pelo", phonemeAccuracy: 0.96)
        XCTAssertEqual(exact.score0To4, 4)
        let empty = analytics.scoreEITResponse(prompt: prompt, learnerTranscript: "", learnerIPA: "", phonemeAccuracy: 0.0)
        XCTAssertEqual(empty.score0To4, 0)
    }

    // MARK: - 7. Zero-Dependency MiniZipExtractor (Stored ZIP Extraction for Cloud ZIPA Release)

    func testMiniZipExtractorStoredArchive() throws {
        // Construct a minimal valid Stored (method 0) ZIP archive in memory containing "ZipaCRLarge300M_FP16.mlmodelc/test.txt"
        let entryName = Data("ZipaCRLarge300M_FP16.mlmodelc/test.txt".utf8)
        let payload = Data("cadencia_zipa_fp16_ok".utf8)
        var zip = Data()

        func appendU16(_ v: UInt16) {
            zip.append(UInt8(v & 0xFF)); zip.append(UInt8((v >> 8) & 0xFF))
        }
        func appendU32(_ v: UInt32) {
            zip.append(UInt8(v & 0xFF)); zip.append(UInt8((v >> 8) & 0xFF))
            zip.append(UInt8((v >> 16) & 0xFF)); zip.append(UInt8((v >> 24) & 0xFF))
        }

        // Local File Header
        appendU32(0x04034b50); appendU16(20); appendU16(0); appendU16(0); appendU16(0); appendU16(0)
        appendU32(0); appendU32(UInt32(payload.count)); appendU32(UInt32(payload.count))
        appendU16(UInt16(entryName.count)); appendU16(0)
        zip.append(entryName)
        zip.append(payload)

        // Central Directory Header
        let cdOffset = UInt32(zip.count)
        appendU32(0x02014b50); appendU16(20); appendU16(20); appendU16(0); appendU16(0); appendU16(0); appendU16(0)
        appendU32(0); appendU32(UInt32(payload.count)); appendU32(UInt32(payload.count))
        appendU16(UInt16(entryName.count)); appendU16(0); appendU16(0); appendU16(0); appendU16(0)
        appendU32(0); appendU32(0)
        zip.append(entryName)
        let cdSize = UInt32(zip.count) - cdOffset

        // End of Central Directory
        appendU32(0x06054b50); appendU16(0); appendU16(0); appendU16(1); appendU16(1)
        appendU32(cdSize); appendU32(cdOffset); appendU16(0)

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("zip_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let zipFile = tmpDir.appendingPathComponent("bundle.zip")
        try zip.write(to: zipFile)

        let outDir = tmpDir.appendingPathComponent("extracted")
        try MiniZipExtractor.extract(zipURL: zipFile, to: outDir)

        let extracted = try String(contentsOf: outDir.appendingPathComponent("ZipaCRLarge300M_FP16.mlmodelc/test.txt"), encoding: .utf8)
        XCTAssertEqual(extracted, "cadencia_zipa_fp16_ok")
    }

    // MARK: - 8. Phonon-40m-es Native Voice Mapping, Bilingual Conversation Segments & 24 kHz Stitching

    func testPhononVoicesBilingualTurnsAndAudioStitching() throws {
        // Legacy SQLite voice names decode transparently to native Spanish Phonon voices
        XCTAssertEqual(TutorVoice.fromStoredRawValue("Sofia"), .celia)
        XCTAssertEqual(TutorVoice.fromStoredRawValue("Aria"), .nerea)
        XCTAssertEqual(TutorVoice.fromStoredRawValue("Jason"), .iker)
        XCTAssertEqual(TutorVoice.fromStoredRawValue("Leo"), .leonel)
        XCTAssertEqual(TutorVoice.fromStoredRawValue("John"), .marcos)
        XCTAssertEqual(TutorVoice.fromStoredRawValue("Regina"), .regina)

        // JSON Codable backwards compatibility for persisted passages
        let legacyJSON = Data("\"Sofia\"".utf8)
        let decodedLegacy = try JSONDecoder().decode(TutorVoice.self, from: legacyJSON)
        XCTAssertEqual(decodedLegacy, .celia)

        // Accent-matched voice resolution (voice carries regional accent in Phonon-40m-es)
        XCTAssertEqual(TutorVoice.voice(for: .castilian, preferred: .regina), .celia)
        XCTAssertEqual(TutorVoice.voice(for: .latinAmerican, preferred: .celia), .regina)
        XCTAssertEqual(TutorVoice.voice(for: .latinAmerican, preferred: .iker), .leonel)
        XCTAssertEqual(TutorVoice.voice(for: .castilian, preferred: .leonel), .iker)

        // Paired English Kokoro-82M voices preserve speaker gender
        XCTAssertEqual(TutorVoice.celia.pairedKokoroEnglishVoice, "af_heart")
        XCTAssertEqual(TutorVoice.leonel.pairedKokoroEnglishVoice, "am_michael")

        // Bilingual GPT-6.1 Terra turn resolution: ES+EN mode keeps both; Spanish-only strips English
        let resp = GPT61TerraClient.SpeakingPartnerResponse(
            recastCorrectionEs: "Fui al mercado ayer.",
            partnerReplyEs: "¡Genial! ¿Qué compraste en el mercado?",
            partnerReplyEn: "Nice job! Use 'fui' for 'I went' in the past tense.",
            spokenSegments: [
                TutorSpeechSegment(language: .english, text: "Nice job! Use 'fui' for 'I went' in the past tense."),
                TutorSpeechSegment(language: .spanish, text: "¡Genial! ¿Qué compraste en el mercado?"),
            ],
            followUpPromptEnHint: "Tell your tutor what you bought.",
            correctlyUsedLemmas: ["mercado"],
            failedGrammarPatternIds: []
        )
        let bilingualSegs = resp.resolvedSpokenSegments(allowEnglish: true)
        XCTAssertEqual(bilingualSegs.count, 2)
        XCTAssertEqual(bilingualSegs[0].language, .english)
        XCTAssertEqual(bilingualSegs[1].language, .spanish)

        let spanishOnlySegs = resp.resolvedSpokenSegments(allowEnglish: false)
        XCTAssertEqual(spanishOnlySegs.count, 1)
        XCTAssertEqual(spanishOnlySegs[0].language, .spanish)

        // Seamless 24 kHz stitching between Phonon Spanish and Kokoro English clips
        let clipEn = SynthesizedAudioClip(
            samples: [Float](repeating: 0.2, count: 24_000),
            sampleRate: 24_000.0,
            durationSeconds: 1.0,
            wordTimings: [WordTimingSpan(wordIndex: 0, word: "Great", startTime: 0, endTime: 1.0)],
            synthesisLatencyMs: 60,
            realTimeFactor: 0.06,
            cachedFileURL: nil
        )
        let clipEs = SynthesizedAudioClip(
            samples: [Float](repeating: 0.3, count: 12_000),
            sampleRate: 24_000.0,
            durationSeconds: 0.5,
            wordTimings: [WordTimingSpan(wordIndex: 0, word: "Hola", startTime: 0, endTime: 0.5)],
            synthesisLatencyMs: 35,
            realTimeFactor: 0.07,
            cachedFileURL: nil
        )
        let stitched = TutorAudioCodec.stitchClips([clipEn, clipEs], pauseSeconds: 0.10)
        XCTAssertEqual(stitched.sampleRate, 24_000.0, accuracy: 1e-6)
        XCTAssertEqual(stitched.samples.count, 24_000 + 2_400 + 12_000)
        XCTAssertEqual(stitched.durationSeconds, 1.6, accuracy: 1e-3)
        XCTAssertEqual(stitched.wordTimings.count, 2)
        XCTAssertEqual(stitched.wordTimings[1].wordIndex, 1)
        XCTAssertEqual(stitched.wordTimings[1].startTime, 1.1, accuracy: 1e-3)
    }

    // MARK: - 9. Revised 16-Page Protocol Compliance (Weak Cap, Feasibility, Bootstrap, Full Contract & Reserve Fallback)

    func testRevisedProtocolWeakCapFeasibilityBootstrapAndFullContract() async throws {
        // 1. Weak-success cap: at most 3 updates per item per session (Revised Protocol p. 6)
        let modelEngine = LearnerModelEngine(parameters: .protocolDefaults)
        let funcItem = SyllabusItem(
            id: "lem_de", itemType: .lemma, rankIndex: 2, spanish: "de", englishGloss: "of",
            pos: "ADP", cefr: .a1, ipaEs: "de", ipaLatam: "de",
            isTransparentCognate: false, cognateScore: 0.0, isBootstrap: true, isHoldout: false,
            isTop2000ProductiveTarget: true, zipfFrequency: 7.5, visualSymbol: "link"
        )
        var rec = LearnerItemRecord(learnerId: "u_cap", itemId: funcItem.id, halfLifeDays: modelEngine.initialHalfLifeDays(isTransparentCognate: false))
        for i in 0..<3 {
            rec = modelEngine.apply(outcome: .passedNormalPace, to: rec, item: funcItem, sessionNumber: 1, wasEarOnly: false, weakSuccessesThisSession: i).updated
        }
        let hAfterThree = rec.halfLifeDays
        let successesAfterThree = rec.successesCount
        XCTAssertEqual(successesAfterThree, 3)

        // 4th & 5th weak successes in the SAME session MUST NOT grow halfLifeDays or successesCount!
        let fourth = modelEngine.apply(outcome: .passedNormalPace, to: rec, item: funcItem, sessionNumber: 1, wasEarOnly: true, weakSuccessesThisSession: 3).updated
        XCTAssertEqual(fourth.halfLifeDays, hAfterThree, accuracy: 1e-6, "4th weak success in same session must be capped")
        XCTAssertEqual(fourth.successesCount, 3, "Capped weak success must not increment successesCount")
        XCTAssertEqual(fourth.earSuccessesCount, 0, "Capped weak success must not grant ear credit")

        // Strong success (probeCorrect) is NOT blocked by the weak-success cap!
        let strong = modelEngine.apply(outcome: .probeCorrect, to: fourth, item: funcItem, sessionNumber: 1, wasEarOnly: false, weakSuccessesThisSession: 3).updated
        XCTAssertGreaterThan(strong.halfLifeDays, hAfterThree)

        // 2. Step 1 Zero-Cost Feasibility Check: 3*new + due <= floor(0.23 * 215) = 49
        let dummyNew = (0..<6).map { i in
            SyllabusItem(id: "n\(i)", itemType: .lemma, rankIndex: i, spanish: "nuevo\(i)", englishGloss: "new", pos: "NOUN", cefr: .a1, ipaEs: "", ipaLatam: "", isTransparentCognate: false, cognateScore: 0, isBootstrap: false, isHoldout: false, isTop2000ProductiveTarget: false, zipfFrequency: 4, visualSymbol: "star")
        }
        let dummyDue = (0..<40).map { i in
            SyllabusItem(id: "d\(i)", itemType: .lemma, rankIndex: 100 + i, spanish: "due\(i)", englishGloss: "due", pos: "NOUN", cefr: .a1, ipaEs: "", ipaLatam: "", isTransparentCognate: false, cognateScore: 0, isBootstrap: false, isHoldout: false, isTop2000ProductiveTarget: false, zipfFrequency: 4, visualSymbol: "star")
        }
        let feasible = SessionScheduler.enforceFeasibility(newItems: dummyNew, dueItems: dummyDue, targetWords: 215, scheduledShareCap: 0.23, minNewOccurrences: 3)
        XCTAssertTrue(feasible.didSplit, "6*3 + 40 = 58 > 49 must trigger Step 1 split")
        XCTAssertEqual(feasible.newItems.count, 6)
        XCTAssertEqual(feasible.dueItems.count, 31, "49 - 18 = 31 due items fit inside 23% of 215 words")

        // 3. Greedy Rule 2 spaced-subsequence helper (extra uses are free and never fail)
        XCTAssertEqual(ContractVerifier.maxSpacedOccurrences(positions: [3, 35, 68, 72, 110], minSpacing: 30), 4)
        XCTAssertEqual(ContractVerifier.maxSpacedOccurrences(positions: [3, 15, 28], minSpacing: 30), 1)

        // 4. Official 8-Kit Verifier Acceptance Test (17-Page Protocol pp. 8–9):
        //    "A hand-written passage for each kit passes a script check of this contract.
        //     Those eight passages are the verifier's acceptance test. A verifier that rejects
        //     any of them has the fault, not the generator."
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("cadencia_boot_test_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let db = try CadenciaDatabase(databaseURL: tmpURL)
        let activeItems = try await db.loadAllSyllabusItems(includeHoldout: false)
        let holdoutItems = try await db.loadHoldoutItems()
        let inflections = try await db.loadInflectionDictionary()
        let genres = try await db.loadSerialGenres()
        let lemmatizer = SpanishLemmatizer(inflections: inflections, chunkItems: activeItems.filter { $0.itemType == .chunk })
        let verifier = ContractVerifier(lemmatizer: lemmatizer, parameters: .protocolDefaults)

        let analytics = ResearchAnalyticsEngine()
        var (h0Profile, h0Records) = analytics.buildSimulatedLearnerState(atHour: 0, allItems: activeItems)
        XCTAssertEqual(h0Profile.currentSessionNumber, 1)
        XCTAssertEqual(h0Records.values.filter { $0.meetingsCount > 0 }.count, 0, "Hour 0 must have 0 pre-loaded Known/met items")

        let scheduler = SessionScheduler(parameters: .protocolDefaults)
        var newRemaining = 60

        for kitIdx in 0..<8 {
            let plan = scheduler.planNextPassage(
                profile: h0Profile,
                blockType: .input1,
                passageIndexInSession: kitIdx,
                newItemsRemainingInSession: newRemaining,
                allItems: activeItems,
                holdoutItems: holdoutItems,
                records: h0Records,
                priorContextByItem: [:],
                serialGenre: genres.first!,
                serialEpisodeNumber: 0,
                storySummarySoFar: ""
            )

            XCTAssertTrue(plan.isBootstrap)
            XCTAssertEqual(plan.bootstrapKitNumber, kitIdx + 1)
            XCTAssertTrue(BootstrapKitLibrary.checkKitFeasibility(items: plan.newItems), "Kit \(kitIdx + 1) must hold a verb and a noun")
            XCTAssertEqual(plan.bootstrapCastNames, ["Ana", "Luis", "Carla", "Perú", "México"])
            if kitIdx == 0 {
                XCTAssertEqual(plan.newItems.map(\.spanish), ["ser", "el", "un", "de", "y", "doctor", "hotel", "famoso"])
                XCTAssertEqual(plan.allowedKnownLemmas.count, 8, "Nothing pre-loaded as Known on Kit 1")
            }

            let handWritten = BootstrapKitLibrary.session1AcceptancePassages[kitIdx]
            let bootSentences = handWritten.sentences.enumerated().map { idx, s in
                PassageSentence(id: idx, spanish: s.es, englishLine: s.en, wordTokens: lemmatizer.extractWordTokens(s.es), visualSymbol: s.symbol)
            }
            let report = verifier.verifyStageA(
                passageText: handWritten.sentences.map(\.es).joined(separator: " "),
                sentences: bootSentences,
                plan: plan,
                priorContextByItem: [:]
            )
            XCTAssertTrue(
                report.passed,
                "ACCEPTANCE TEST FAILED on Kit \(kitIdx + 1)! Violations: \(report.violations.map { "\($0.ruleName): \($0.detail) [\($0.offendingTokens.joined(separator: ", "))]" }.joined(separator: " | "))"
            )
            XCTAssertGreaterThanOrEqual(report.closedVocabCompliance, 0.99, "Hand-written Kit \(kitIdx + 1) should achieve 100% core compliance via inflected-forms table + al/del split")

            // Verify whole-sentence picture-matching probe maps to multiple new items in the sentence
            if let firstProbe = handWritten.probes.first {
                let matched = verifier.matchedItemIds(in: firstProbe.sentenceEs, from: plan.newItems)
                XCTAssertGreaterThanOrEqual(matched.count, 2, "Kit \(kitIdx + 1) whole-sentence probe should match >=2 kit items")
            }

            // Mark this kit's items as met so subsequent kits accumulate metSoFar cleanly!
            for item in plan.newItems {
                var existing = h0Records[item.id] ?? LearnerItemRecord(learnerId: h0Profile.learnerId, itemId: item.id, halfLifeDays: 1.0)
                existing.meetingsCount += 1
                existing.firstMeetingSession = 1
                existing.lastMeetingSession = 1
                existing.lastMeetingAt = Date()
                existing.state = .learning
                h0Records[item.id] = existing
            }
            newRemaining = max(0, newRemaining - plan.newItems.count)
        }

        // Passage #9 (after all 8 kits = 60 items) MUST be a recombination passage adding 0 new items!
        let recombPlan = scheduler.planNextPassage(
            profile: h0Profile,
            blockType: .input1,
            passageIndexInSession: 8,
            newItemsRemainingInSession: newRemaining,
            allItems: activeItems,
            holdoutItems: holdoutItems,
            records: h0Records,
            priorContextByItem: [:],
            serialGenre: genres.first!,
            serialEpisodeNumber: 0,
            storySummarySoFar: ""
        )
        XCTAssertTrue(recombPlan.isBootstrapRecombination)
        XCTAssertTrue(recombPlan.newItems.isEmpty, "Post-kit recombination adds zero new items")
        XCTAssertGreaterThanOrEqual(recombPlan.allowedKnownLemmas.count, 60, "All 60 Session 1 kit items are now met and available")

        // Verify `al` and `del` expand as TWO items (a+el, de+el)
        XCTAssertEqual(BootstrapKitLibrary.expandTokenToCandidateItemSets(normalizedSurface: "al", lemmatizerLemma: "al").count, 2)
        XCTAssertEqual(BootstrapKitLibrary.expandTokenToCandidateItemSets(normalizedSurface: "del", lemmatizerLemma: "del").count, 2)

        // Verify Step 6 Reserve Passages (including all 8 Session 1 Kit reserves) contain ZERO hold-out leaks
        let holdoutSet = Set(holdoutItems.map { $0.spanish.lowercased() })
        for kitNum in 1...8 {
            let kit = BootstrapKitLibrary.resolveSession1Kit(kitNumber: kitNum, allItems: activeItems)!
            let resPlan = PassagePlan(
                passageId: "res_kit_\(kitNum)", sessionNumber: 1, passageIndexInSession: kitNum - 1,
                blockType: .input1, isBootstrap: true, isAudioFirst: false, targetWPM: 75,
                format: .bootstrapMicroStory, topicOrEpisodeTitle: "Kit \(kitNum)",
                storySummarySoFar: "", dueItems: [], newItems: kit.items, grammarPatterns: [],
                allowedKnownLemmas: kit.items.map(\.spanish), holdoutBannedLemmas: Array(holdoutSet),
                voice: .celia, accent: .castilian, priorContextWordsByItem: [:],
                bootstrapKitNumber: kitNum, bootstrapPatternDescription: kit.spec.pattern,
                bootstrapCastNames: BootstrapKitLibrary.fixedCastDisplayNames
            )
            let reserve = ReservePassageBank.selectReserve(for: resPlan, lemmatizer: lemmatizer, generatorCallsUsed: 3)
            XCTAssertTrue(reserve.isReserveFallback)
            XCTAssertEqual(reserve.generatorCallsUsed, 3)
            XCTAssertFalse(reserve.probes.isEmpty)
            XCTAssertTrue(reserve.probes.allSatisfy { $0.isPictureMatch && ($0.targetSentenceSpanish?.isEmpty == false) })
            let toks = lemmatizer.analyze(reserve.fullText)
            let leaks = toks.filter { holdoutSet.contains($0.lemma.lowercased()) }.map(\.surface)
            XCTAssertTrue(leaks.isEmpty, "Reserve Kit \(kitNum) leaked hold-out words: \(leaks)")
        }

        XCTAssertEqual(PhononComputeMode.fastCPU.rawValue, "cpu")
    }
}
