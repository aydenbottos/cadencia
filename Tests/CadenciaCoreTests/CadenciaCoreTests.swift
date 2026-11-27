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
            voice: .sofia, accent: .castilian, priorContextWordsByItem: [:]
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

    // MARK: - 8. NVIDIA Magpie v2607 Spanish Tokenizer, Attention Prior & argmax_or_multinomial_any EOS

    func testMagpieV2607TokenizerAttentionPriorAndEOSDetector() {
        let token2Id: [String: Int32] = [
            ".": 7, " ": 100, "<pad>": 101,
            "D": 15, "Í": 69, "A": 12, "S": 30, "Y": 36,
            "p": 53, "ˈ": 91, "o": 52, "ɾ": 85, "e": 43, "k": 48,
            "ʝ": 90, "a": 40, "d": 42, "ð": 77, "θ": 97, "s": 55, "ʎ": 88, "ʃ": 86,
        ]
        let dict: [String: [String]] = [
            "POR": ["p", "ˈ", "o", "ɾ"],
            "PERO": ["p", "ˈ", "e", "ɾ", "o"],
            "YA": ["ʝ", "ˈ", "a"],
            "DADO": ["d", "ˈ", "a", "ð", "o"],
            "QUE": ["k", "ˈ", "e"],
            "CALLE": ["k", "ˈ", "a", "ʎ", "e"],
        ]

        // v2607 Spanish offset=480, eos=3358: bare word "por" must auto-append "." and terminate with 3358
        let v2607Tok = MagpieSpanishTokenizer(token2Id: token2Id, phonemeDict: dict, spanishOffset: 480, eosTokenId: 3358)
        let porIds = v2607Tok.tokenize(text: "por", accent: .castilian)
        XCTAssertEqual(porIds.first, 480 + 100, "Leading space must use v2607 offset 480")
        XCTAssertEqual(porIds.last, 3358, "Must terminate with v2607 EOS 3358")
        XCTAssertTrue(porIds.contains(480 + 7), "Bare word must auto-append terminal period '.' to prevent trailing hallucination")
        XCTAssertTrue(porIds.contains(480 + 85), "Must emit Spanish tap /ɾ/ from es_ES dictionary")

        // OOV word "días" must use NeMo IpaG2p uppercase grapheme fallback (D, Í, A, S)
        let diasIds = v2607Tok.tokenize(text: "días", accent: .castilian)
        XCTAssertTrue(diasIds.contains(480 + 15) && diasIds.contains(480 + 69) && diasIds.contains(480 + 12) && diasIds.contains(480 + 30))

        // Ultra-short OOV word "y" must be padded to >= 4 inner tokens before EOS
        let yIds = v2607Tok.tokenize(text: "y", accent: .castilian)
        XCTAssertGreaterThanOrEqual(yIds.count - 2, 4)

        // Regional accent overrides: LatAm yeísmo (ʎ -> ʝ) and Rioplatense sheísmo (ʎ -> ʃ)
        let calleLatam = v2607Tok.tokenize(text: "calle.", accent: .latinAmerican)
        let calleRio = v2607Tok.tokenize(text: "calle.", accent: .rioplatense)
        XCTAssertTrue(calleLatam.contains(480 + 90))
        XCTAssertTrue(calleRio.contains(480 + 86))

        // Legacy v2512 offset=96, eos=2361 compatibility
        let v2512Tok = MagpieSpanishTokenizer(token2Id: token2Id, phonemeDict: dict, spanishOffset: 96, eosTokenId: 2361)
        let queLegacy = v2512Tok.tokenize(text: "que", accent: .castilian)
        XCTAssertEqual(queLegacy.first, 196)
        XCTAssertEqual(queLegacy.last, 2361)

        // Monotonic attention prior advances cleanly and suppresses past attention sinks
        var prior = MagpieAttentionPriorState(textLen: 12, epsilon: 0.1, lookaheadWindow: 6, advanceThreshold: 8, decayThreshold: 10)
        var scores = [Float](repeating: 0.01, count: 12)
        scores[4] = 0.9
        prior.update(alignmentScores: scores)
        XCTAssertEqual(prior.lastAttended, 4)
        XCTAssertEqual(prior.prior[3], 1.0, accuracy: 1e-5)
        XCTAssertEqual(prior.prior[4], 1.0, accuracy: 1e-5)
        XCTAssertEqual(prior.prior[10], 1.0, accuracy: 1e-5)
        XCTAssertEqual(prior.prior[0], 0.1, accuracy: 1e-5)

        var biasBuf = [Float](repeating: 0, count: 16)
        biasBuf.withUnsafeMutableBufferPointer { ptr in
            prior.fillLogPriorBias(into: ptr.baseAddress!, maxTextLen: 16)
        }
        XCTAssertEqual(biasBuf[4], 0.0, accuracy: 1e-5)
        XCTAssertEqual(biasBuf[0], logf(0.1), accuracy: 1e-4)
        XCTAssertEqual(biasBuf[12], -30_000.0, accuracy: 1e-3)

        // Exact NeMo argmax_or_multinomial_any EOS detector on 2x stacked frames (16 slots)
        let nonEOS = [Int32](repeating: 42, count: 16)
        var mainEOSLane1 = nonEOS
        mainEOSLane1[10] = 2017 // EOS in frame lane 1 predicted by main decoder head even when multinomial missed it!
        XCTAssertEqual(
            MagpieEOSDetector.firstEOSLane(
                sampled: nonEOS,
                greedyLT: nonEOS,
                greedyMain: mainEOSLane1,
                frameStackingFactor: 2,
                framesEmittedSoFar: 6,
                minFrames: 4
            ),
            1,
            "Must stop at lane 1 (emitting lane 0) when main head argmax hits EOS!"
        )
        // Respects minFrames guard
        var ltEOSLane0 = nonEOS
        ltEOSLane0[0] = 2017
        XCTAssertNil(
            MagpieEOSDetector.firstEOSLane(
                sampled: ltEOSLane0,
                greedyLT: ltEOSLane0,
                greedyMain: ltEOSLane0,
                frameStackingFactor: 2,
                framesEmittedSoFar: 0,
                minFrames: 4
            )
        )
    }
}
