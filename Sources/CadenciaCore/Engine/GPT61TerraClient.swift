import Foundation
#if canImport(Security)
import Security
#endif

// MARK: - Keychain API Key Vault

public enum KeychainVault {
    private static let service = "com.aydenbottos.cadencia.openai"
    private static let account = "openai_api_key"

    public static func saveAPIKey(_ key: String) -> Bool {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return deleteAPIKey() }
        #if canImport(Security)
        let data = Data(trimmed.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
        var addQuery = query
        addQuery[kSecValueData as String] = data
        addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        return SecItemAdd(addQuery as CFDictionary, nil) == errSecSuccess
        #else
        UserDefaults.standard.set(trimmed, forKey: account)
        return true
        #endif
    }

    public static func loadAPIKey() -> String? {
        if let envKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"], !envKey.isEmpty {
            return envKey
        }
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
        #else
        return UserDefaults.standard.string(forKey: account)
        #endif
    }

    @discardableResult
    public static func deleteAPIKey() -> Bool {
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
        #else
        UserDefaults.standard.removeObject(forKey: account)
        return true
        #endif
    }
}

// MARK: - OpenAI Responses API Client for GPT-6.1 Terra

public enum TerraReasoningEffort: String, Codable, Sendable {
    case none
    case low
    case medium
    case high
    case xhigh
}

public enum TerraAPIError: Error, LocalizedError {
    case missingAPIKey
    case httpError(Int, String)
    case emptyOutput
    case decodingFailed(String)
    case contractExhausted(String)

    public var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "OpenAI API key is not configured. Add your key in Settings -> API Vault."
        case .httpError(let code, let body):
            return "GPT-6.1 Terra HTTP \(code): \(body.prefix(300))"
        case .emptyOutput:
            return "GPT-6.1 Terra returned no text output."
        case .decodingFailed(let detail):
            return "Failed to parse structured JSON from GPT-6.1 Terra: \(detail)"
        case .contractExhausted(let detail):
            return "Passage generation could not satisfy all contract rules: \(detail)"
        }
    }
}

public struct TerraTokenUsage: Sendable, Codable {
    public let inputTokens: Int
    public let outputTokens: Int
    public let reasoningTokens: Int
}

/// Complete OpenAI Responses API (`/v1/responses`) client exclusively targeting `gpt-6.1-terra`
/// with strict JSON Schema structured outputs (`text.format.json_schema`) and configurable
/// `reasoning.effort` for:
///  1. Contract-bound Passage Generator (with up to 3 rewrites + 25% due-list reduction fallback)
///  2. Isolated Second-Model Verifier (Rules 5 & 6 + Spanish grammatical naturalness)
///  3. Closed-Vocabulary Voice Speaking Partner (Retell, Open Talk, SIELE Speaking, implicit recast corrections)
///  4. 13-Minute Writing Coach (recast pairs + pattern failure detection + productive item extraction)
///  5. Semantic Meaning Audit Grader (for the Weekly 60-Item Calibration Audit)
public actor GPT61TerraClient {
    public static let modelID = "gpt-6.1-terra"
    private static let responsesURL = URL(string: "https://api.openai.com/v1/responses")!

    private var explicitAPIKey: String?
    private let session: URLSession
    public private(set) var cumulativeInputTokens: Int = 0
    public private(set) var cumulativeOutputTokens: Int = 0

    public init(apiKey: String? = nil, session: URLSession = .shared) {
        self.explicitAPIKey = apiKey
        self.session = session
    }

    public func setAPIKey(_ key: String?) {
        self.explicitAPIKey = key
    }

    private func resolveAPIKey() throws -> String {
        if let explicit = explicitAPIKey, !explicit.isEmpty { return explicit }
        if let stored = KeychainVault.loadAPIKey(), !stored.isEmpty { return stored }
        throw TerraAPIError.missingAPIKey
    }

    /// Lightweight connectivity & key validation against `gpt-6.1-terra`.
    public func verifyConnection() async throws -> String {
        struct PingResp: Decodable { let status: String; let greeting: String }
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "status": ["type": "string"],
                "greeting": ["type": "string"],
            ],
            "required": ["status", "greeting"],
            "additionalProperties": false,
        ]
        let (parsed, _): (PingResp, TerraTokenUsage) = try await callStructuredJSON(
            instructions: "Respond with status 'ok' and a short warm Spanish welcome greeting under 8 words.",
            userInput: "Verify connection for Cadencia Spanish immersion.",
            schemaName: "connection_ping",
            jsonSchema: schema,
            effort: .none
        )
        return parsed.greeting
    }

    // MARK: - 1. Contract-Enforced Passage Generation + 2-Stage Verifier Loop

    /// Runs the full Generator -> Stage A Local Verifier -> Stage B Second-Model Verifier loop
    /// with up to 3 rewrites, followed by cutting the due list by 25% and regenerating if needed (PDF p. 7).
    public func generateVerifiedPassage(
        plan initialPlan: PassagePlan,
        verifier: ContractVerifier,
        priorContextByItem: [String: Set<String>],
        onProgress: (@Sendable (String) -> Void)? = nil
    ) async throws -> (passage: GeneratedPassage, report: VerificationReport) {
        var currentPlan = initialPlan
        var dueWasReduced = false
        var lastReport: VerificationReport?
        var totalAttempts = 0

        // Outer loop: round 0 = full due list (up to 3 rewrites); round 1 = due list cut by 25% (up to 3 rewrites)
        for reductionRound in 0...1 {
            if reductionRound == 1 {
                let reducedCount = max(3, Int((Double(currentPlan.dueItems.count) * 0.75).rounded(.down)))
                let reducedDue = Array(currentPlan.dueItems.prefix(reducedCount))
                currentPlan = PassagePlan(
                    passageId: currentPlan.passageId,
                    sessionNumber: currentPlan.sessionNumber,
                    passageIndexInSession: currentPlan.passageIndexInSession,
                    blockType: currentPlan.blockType,
                    isBootstrap: currentPlan.isBootstrap,
                    isAudioFirst: currentPlan.isAudioFirst,
                    targetWPM: currentPlan.targetWPM,
                    format: currentPlan.format,
                    topicOrEpisodeTitle: currentPlan.topicOrEpisodeTitle,
                    storySummarySoFar: currentPlan.storySummarySoFar,
                    dueItems: reducedDue,
                    newItems: currentPlan.newItems,
                    grammarPatterns: currentPlan.grammarPatterns,
                    allowedKnownLemmas: currentPlan.allowedKnownLemmas,
                    holdoutBannedLemmas: currentPlan.holdoutBannedLemmas,
                    voice: currentPlan.voice,
                    accent: currentPlan.accent,
                    priorContextWordsByItem: currentPlan.priorContextWordsByItem
                )
                dueWasReduced = true
                onProgress?("Reducing due list by 25% and regenerating…")
            }

            var feedbackFromPreviousAttempt: String? = nil

            for attempt in 0...3 {
                totalAttempts += 1
                onProgress?(attempt == 0 ? "Drafting passage with GPT-6.1 Terra…" : "Rewriting passage (attempt \(attempt)/3)…")

                let draft = try await draftPassageOnce(
                    plan: currentPlan,
                    rewriteFeedback: feedbackFromPreviousAttempt,
                    effort: attempt == 0 ? .medium : .high
                )

                let fullText = draft.sentences.map(\.spanish).joined(separator: " ")
                let passageSentences = draft.sentences.enumerated().map { idx, s in
                    PassageSentence(
                        id: idx,
                        spanish: s.spanish,
                        englishLine: s.englishLine,
                        wordTokens: verifier.lemmatizer.extractWordTokens(s.spanish)
                    )
                }

                // Stage A: Deterministic Local Verifier (<10ms)
                var report = verifier.verifyStageA(
                    passageText: fullText,
                    sentences: passageSentences,
                    plan: currentPlan,
                    priorContextByItem: priorContextByItem
                )

                if !report.passed {
                    lastReport = report
                    feedbackFromPreviousAttempt = formatViolationsForRewrite(report: report, previousText: fullText)
                    continue
                }

                // Stage B: Isolated Second-Model Verifier (Rules 5 & 6 + Naturalness)
                onProgress?("Verifying load-bearing probes & grammar (Stage B)…")
                let stageB = try await verifyStageB(
                    fullText: fullText,
                    probes: draft.probes,
                    plan: currentPlan
                )

                if !stageB.passed {
                    var combinedViolations = report.violations
                    combinedViolations.append(ContractViolation(
                        ruleNumber: stageB.violatedRuleNumber ?? 5,
                        ruleName: "Stage B Semantic & Grammar Check",
                        detail: stageB.feedback,
                        offendingTokens: []
                    ))
                    report = VerificationReport(
                        passed: false,
                        totalWords: report.totalWords,
                        closedVocabCompliance: report.closedVocabCompliance,
                        scheduledWordShare: report.scheduledWordShare,
                        missingDueItems: report.missingDueItems,
                        underSpacedNewItems: report.underSpacedNewItems,
                        freshContextCollisions: report.freshContextCollisions,
                        holdoutLeaks: report.holdoutLeaks,
                        offListLemmas: report.offListLemmas,
                        stageBPassed: false,
                        stageBFeedback: stageB.feedback,
                        violations: combinedViolations
                    )
                    lastReport = report
                    feedbackFromPreviousAttempt = formatViolationsForRewrite(report: report, previousText: fullText)
                    continue
                }

                // Passed BOTH Stage A and Stage B!
                let passage = buildGeneratedPassage(
                    draft: draft,
                    sentences: passageSentences,
                    fullText: fullText,
                    plan: currentPlan,
                    report: report,
                    rewriteAttempts: totalAttempts - 1,
                    dueWasReduced: dueWasReduced
                )
                return (passage, report)
            }
        }

        // Best-effort fallback: draft once more with relaxed due list if somehow exhausted, flagging report honestly
        let finalDraft = try await draftPassageOnce(plan: currentPlan, rewriteFeedback: nil, effort: .medium)
        let fullText = finalDraft.sentences.map(\.spanish).joined(separator: " ")
        let passageSentences = finalDraft.sentences.enumerated().map { idx, s in
            PassageSentence(
                id: idx,
                spanish: s.spanish,
                englishLine: s.englishLine,
                wordTokens: verifier.lemmatizer.extractWordTokens(s.spanish)
            )
        }
        let finalReport = verifier.verifyStageA(
            passageText: fullText,
            sentences: passageSentences,
            plan: currentPlan,
            priorContextByItem: priorContextByItem
        )
        let passage = buildGeneratedPassage(
            draft: finalDraft,
            sentences: passageSentences,
            fullText: fullText,
            plan: currentPlan,
            report: finalReport,
            rewriteAttempts: totalAttempts,
            dueWasReduced: dueWasReduced
        )
        return (passage, lastReport ?? finalReport)
    }

    private struct RawDraftSentence: Decodable {
        let spanish: String
        let englishLine: String?
    }
    private struct RawDraftGloss: Decodable {
        let itemId: String
        let spanish: String
        let englishGloss: String
        let contextClueNote: String
    }
    private struct RawDraftProbe: Decodable {
        let targetDueItemId: String
        let targetDueLemma: String
        let questionEn: String
        let options: [String]
        let correctOptionIndex: Int
        let hingeExplanation: String
    }
    private struct RawPassageDraft: Decodable {
        let titleEs: String
        let titleEn: String
        let episodeSummaryForNext: String
        let sentences: [RawDraftSentence]
        let newItemGlosses: [RawDraftGloss]
        let probes: [RawDraftProbe]
    }

    private func draftPassageOnce(
        plan: PassagePlan,
        rewriteFeedback: String?,
        effort: TerraReasoningEffort
    ) async throws -> RawPassageDraft {
        let newItemsDesc = plan.newItems.map { "\($0.id): \($0.spanish) = \($0.englishGloss) [\($0.pos)]" }.joined(separator: "\n")
        let dueItemsDesc = plan.dueItems.map { "\($0.id): \($0.spanish) (\($0.englishGloss))" }.joined(separator: ", ")
        let grammarDesc = plan.grammarPatterns.map { g in
            "\(g.id) [\(g.cefr.rawValue)] \(g.spanish): \(g.microNoteEn ?? "") | Example: \(g.contrastAEs ?? "")"
        }.joined(separator: "\n")

        let priorNeighborsText = plan.priorContextWordsByItem.isEmpty
            ? "None yet."
            : plan.priorContextWordsByItem.map { "\($0.key) — NEVER place near: \($0.value.joined(separator: ", "))" }.joined(separator: "\n")

        let allowedSample = plan.allowedKnownLemmas.prefix(1200).joined(separator: ", ")

        let lengthRule = plan.isBootstrap
            ? "BOOTSTRAP MODE (Sessions 1–4): Write 8–10 crisp sentences of AT MOST 8 words each. Include an accurate English translation in `englishLine` for each sentence. Keep vocabulary >=80% inside the allowed list."
            : "FULL CONTRACT: Write 185–240 natural, vivid Spanish words total. EVERY SINGLE word lemma MUST come strictly from ALLOWED_LEMMAS (or proper names/numbers). NO off-list words."

        let instructions = """
        You are the Spanish Passage Generator inside Cadencia, implementing the Signal-Starvation Hypothesis protocol.
        You write extraordinarily natural, vivid, emotionally engaging Spanish prose while obeying EVERY mechanical rule strictly:
        1. \(lengthRule)
        2. NEW ITEMS (at most 6): Every item in NEW_ITEMS MUST appear at least 3 times, spaced at least \(plan.isBootstrap ? 8 : 30) words apart. Its FIRST use must make the meaning clearly guessable from context.
        3. DUE ITEMS: EVERY item in DUE_ITEMS MUST appear at least once. Combined occurrences of NEW_ITEMS + DUE_ITEMS should be 18%–23% of total words.
        4. FRESH CONTEXT: Never place any due item within ±3 words of its banned prior neighbor words listed below.
        5. LOAD-BEARING PROBES: Write 2 or 3 English comprehension probes. EACH probe MUST hinge specifically on understanding ONE listed due item (cannot be guessed without knowing that due word!). Provide 4 plausible options and the 0-based `correctOptionIndex`.
        6. GRAMMAR PATTERNS: Each assigned grammar pattern MUST appear at least TWICE with different lexical words.
        7. CONTINUITY: Continue the serial episode or write a gripping short non-fiction piece matching the topic, with authentic Spanish register (\(plan.accent.displayName)).
        """

        var userPrompt = """
        FORMAT: \(plan.format.rawValue)
        TITLE / TOPIC: \(plan.topicOrEpisodeTitle)
        STORY SO FAR: \(plan.storySummarySoFar.isEmpty ? "Opening episode — establish vivid setting, protagonist, and immediate hook." : plan.storySummarySoFar)
        REGIONAL ACCENT FLAVOR: \(plan.accent.displayName)

        NEW_ITEMS (3x each, >=\(plan.isBootstrap ? 8 : 30) words apart):
        \(newItemsDesc.isEmpty ? "None this passage." : newItemsDesc)

        DUE_ITEMS (each >=1x):
        \(dueItemsDesc.isEmpty ? "None yet." : dueItemsDesc)

        ASSIGNED_GRAMMAR_PATTERNS (each >=2x with different words):
        \(grammarDesc.isEmpty ? "Natural A1-B2 grammar." : grammarDesc)

        RULE 4 BANNED PRIOR NEIGHBORS:
        \(priorNeighborsText)

        ALLOWED_LEMMAS (stay strictly inside this closed vocabulary + names/numbers):
        \(allowedSample)
        """

        if let fb = rewriteFeedback {
            userPrompt += "\n\nCRITICAL REWRITE FIXES REQUIRED FROM VERIFIER:\n\(fb)\nFix EVERY listed violation while keeping natural, vivid Spanish prose."
        }

        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "titleEs": ["type": "string"],
                "titleEn": ["type": "string"],
                "episodeSummaryForNext": ["type": "string"],
                "sentences": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "spanish": ["type": "string"],
                            "englishLine": ["type": ["string", "null"]],
                        ],
                        "required": ["spanish", "englishLine"],
                        "additionalProperties": false,
                    ],
                ],
                "newItemGlosses": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "itemId": ["type": "string"],
                            "spanish": ["type": "string"],
                            "englishGloss": ["type": "string"],
                            "contextClueNote": ["type": "string"],
                        ],
                        "required": ["itemId", "spanish", "englishGloss", "contextClueNote"],
                        "additionalProperties": false,
                    ],
                ],
                "probes": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "targetDueItemId": ["type": "string"],
                            "targetDueLemma": ["type": "string"],
                            "questionEn": ["type": "string"],
                            "options": ["type": "array", "items": ["type": "string"]],
                            "correctOptionIndex": ["type": "integer"],
                            "hingeExplanation": ["type": "string"],
                        ],
                        "required": ["targetDueItemId", "targetDueLemma", "questionEn", "options", "correctOptionIndex", "hingeExplanation"],
                        "additionalProperties": false,
                    ],
                ],
            ],
            "required": ["titleEs", "titleEn", "episodeSummaryForNext", "sentences", "newItemGlosses", "probes"],
            "additionalProperties": false,
        ]

        let (draft, _): (RawPassageDraft, TerraTokenUsage) = try await callStructuredJSON(
            instructions: instructions,
            userInput: userPrompt,
            schemaName: "spanish_passage_contract",
            jsonSchema: schema,
            effort: effort
        )
        return draft
    }

    // MARK: - 2. Stage B Isolated Second-Model Verifier (Rules 5 & 6 + Naturalness)

    private struct StageBVerdict: Decodable {
        let passed: Bool
        let probesAreLoadBearing: Bool
        let grammarPatternsVerified: Bool
        let spanishIsNatural: Bool
        let violatedRuleNumber: Int?
        let feedback: String
    }

    private func verifyStageB(
        fullText: String,
        probes: [RawDraftProbe],
        plan: PassagePlan
    ) async throws -> StageBVerdict {
        let probeText = probes.enumerated().map { idx, p in
            "Probe \(idx + 1) [target: \(p.targetDueLemma)]: \(p.questionEn) | Options: \(p.options.joined(separator: " / ")) | Correct: \(p.correctOptionIndex)"
        }.joined(separator: "\n")
        let gramText = plan.grammarPatterns.map { "\($0.id): \($0.spanish)" }.joined(separator: "\n")

        let instructions = """
        You are an independent, strict native-level Spanish Linguistic Verifier (Stage B).
        Evaluate ONLY three things:
        1. Rule 5 (Load-bearing probes): Are there 2–3 probes, and does each genuinely require understanding its bound Spanish due item (not guessable from general knowledge alone)?
        2. Rule 6 (Grammar patterns): Does each assigned grammar pattern appear at least twice with different lexical words?
        3. Naturalness: Is the Spanish idiomatic, grammatically accurate, and coherent?
        Return passed=true if all three checks are solid; otherwise return passed=false with concrete fix instructions.
        """
        let input = """
        PASSAGE:
        \(fullText)

        PROBES:
        \(probeText)

        ASSIGNED GRAMMAR PATTERNS:
        \(gramText.isEmpty ? "None specified" : gramText)
        """
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "passed": ["type": "boolean"],
                "probesAreLoadBearing": ["type": "boolean"],
                "grammarPatternsVerified": ["type": "boolean"],
                "spanishIsNatural": ["type": "boolean"],
                "violatedRuleNumber": ["type": ["integer", "null"]],
                "feedback": ["type": "string"],
            ],
            "required": ["passed", "probesAreLoadBearing", "grammarPatternsVerified", "spanishIsNatural", "violatedRuleNumber", "feedback"],
            "additionalProperties": false,
        ]
        let (verdict, _): (StageBVerdict, TerraTokenUsage) = try await callStructuredJSON(
            instructions: instructions,
            userInput: input,
            schemaName: "stage_b_verifier",
            jsonSchema: schema,
            effort: .low
        )
        return verdict
    }

    // MARK: - 3. Closed-Vocabulary Voice Speaking Partner (Output 1 Retell & Output 2 Open Talk / SIELE)

    public struct SpeakingPartnerResponse: Decodable, Sendable {
        public let recastCorrectionEs: String?
        public let partnerReplyEs: String
        public let followUpPromptEnHint: String?
        public let correctlyUsedLemmas: [String]
        public let failedGrammarPatternIds: [String]
    }

    public func evaluateSpeakingTurn(
        mode: SessionBlockType,
        week: Int,
        learnerTranscript: String,
        learnerIPA: String,
        phonemeAccuracy: Double,
        recentPassageSummary: String?,
        targetDueItems: [SyllabusItem],
        steeringProductiveItems: [SyllabusItem],
        scheduledGrammarPatterns: [SyllabusItem],
        allowedVocabularySample: [String],
        conversationHistory: [(role: String, text: String)]
    ) async throws -> SpeakingPartnerResponse {
        let dueNames = targetDueItems.map(\.spanish).joined(separator: ", ")
        let steerNames = steeringProductiveItems.map { "\($0.spanish) (\($0.englishGloss))" }.joined(separator: ", ")
        let gramList = scheduledGrammarPatterns.map { "\($0.id): \($0.spanish)" }.joined(separator: "\n")
        let examNote = (mode == .output2 && week >= 5)
            ? "WEEK 5+ SIELE EXAM FORMAT: Frame turns like SIELE Oral Expression & Interaction (monologue description, negotiating an everyday situation, or defending an opinion)."
            : ""

        let instructions = """
        You are Cadencia's warm, natural Spanish Voice Partner under the Signal-Starvation protocol:
        - STRICT CLOSED VOCABULARY: Reply using ONLY vocabulary from ALLOWED_VOCABULARY plus names/numbers. Keep replies concise (1–3 natural spoken Spanish sentences) so conversation flows briskly.
        - RECAST CORRECTIONS (PDF p. 9): If the learner made a Spanish grammar or lexical error, put the clean corrected version of their sentence in `recastCorrectionEs` and weave it naturally into your opening — with ZERO grammar lecture or metalinguistic explanation! Move the conversation forward immediately.
        - MODE: \(mode == .output1 ? "OUTPUT 1 RETELL — Guide the learner to retell the passage just heard, then ask 2–3 questions that CANNOT be answered without using the passage's DUE_ITEMS." : "OUTPUT 2 OPEN TALK / ROLE-PLAY — Natural everyday conversation gently steering topics so the learner produces STEERING_TARGETS.")
        \(examNote)
        - EVALUATION: List every Spanish target lemma from DUE_ITEMS or STEERING_TARGETS that the learner used accurately in `correctlyUsedLemmas`. List any `GRAM_XXX` ID from SCHEDULED_GRAMMAR where the learner made a clear grammatical error in `failedGrammarPatternIds`.
        """

        let historyText = conversationHistory.suffix(8).map { "\($0.role.uppercased()): \($0.text)" }.joined(separator: "\n")
        let userInput = """
        RECENT PASSAGE CONTEXT: \(recentPassageSummary ?? "Open conversation")
        DUE_ITEMS: \(dueNames)
        STEERING_TARGETS (top-2000 productive coverage): \(steerNames)
        SCHEDULED_GRAMMAR:
        \(gramList)
        ALLOWED_VOCABULARY: \(allowedVocabularySample.prefix(900).joined(separator: ", "))

        CONVERSATION SO FAR:
        \(historyText)

        LATEST LEARNER UTTERANCE (Parakeet Ultra): \(learnerTranscript)
        LEARNER IPA PHONEMES (ZIPA CR Large 300M): /\(learnerIPA)/ (accuracy: \(Int((phonemeAccuracy * 100).rounded()))%)
        """

        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "recastCorrectionEs": ["type": ["string", "null"]],
                "partnerReplyEs": ["type": "string"],
                "followUpPromptEnHint": ["type": ["string", "null"]],
                "correctlyUsedLemmas": ["type": "array", "items": ["type": "string"]],
                "failedGrammarPatternIds": ["type": "array", "items": ["type": "string"]],
            ],
            "required": ["recastCorrectionEs", "partnerReplyEs", "followUpPromptEnHint", "correctlyUsedLemmas", "failedGrammarPatternIds"],
            "additionalProperties": false,
        ]

        // Low reasoning effort for crisp real-time conversational latency
        let (resp, _): (SpeakingPartnerResponse, TerraTokenUsage) = try await callStructuredJSON(
            instructions: instructions,
            userInput: userInput,
            schemaName: "speaking_partner_turn",
            jsonSchema: schema,
            effort: .low
        )
        return resp
    }

    // MARK: - 4. 13-Minute Writing Coach (Week 3+, Twice Weekly)

    public func evaluateWritingSubmission(
        promptEs: String,
        learnerText: String,
        targetItems: [SyllabusItem],
        scheduledGrammar: [SyllabusItem],
        elapsedSeconds: Double
    ) async throws -> WritingSubmissionFeedback {
        struct RawWritingEval: Decodable {
            let recastSentences: [RecastPair]
            let correctlyUsedItemIds: [String]
            let failedGrammarPatternIds: [String]
        }

        let targetMap = targetItems.map { "\($0.id): \($0.spanish)" }.joined(separator: ", ")
        let gramMap = scheduledGrammar.map { "\($0.id): \($0.spanish)" }.joined(separator: "\n")

        let instructions = """
        You are Cadencia's Spanish Writing Evaluator (Protocol p. 9).
        For every sentence the learner wrote, output a clean corrected recast version with NO grammar explanation.
        Identify which `targetItems` IDs were used accurately in `correctlyUsedItemIds`, and any `scheduledGrammar` IDs where an error occurred in `failedGrammarPatternIds`.
        """

        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "recastSentences": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "original": ["type": "string"],
                            "corrected": ["type": "string"],
                            "hadError": ["type": "boolean"],
                            "failedPatternId": ["type": ["string", "null"]],
                        ],
                        "required": ["original", "corrected", "hadError", "failedPatternId"],
                        "additionalProperties": false,
                    ],
                ],
                "correctlyUsedItemIds": ["type": "array", "items": ["type": "string"]],
                "failedGrammarPatternIds": ["type": "array", "items": ["type": "string"]],
            ],
            "required": ["recastSentences", "correctlyUsedItemIds", "failedGrammarPatternIds"],
            "additionalProperties": false,
        ]

        let (eval, _): (RawWritingEval, TerraTokenUsage) = try await callStructuredJSON(
            instructions: instructions,
            userInput: "PROMPT: \(promptEs)\nTARGET ITEMS: \(targetMap)\nGRAMMAR:\n\(gramMap)\n\nLEARNER TEXT:\n\(learnerText)",
            schemaName: "writing_recast_eval",
            jsonSchema: schema,
            effort: .low
        )

        let words = learnerText.split { $0.isWhitespace || $0.isNewline }.count
        return WritingSubmissionFeedback(
            id: UUID().uuidString,
            promptEs: promptEs,
            learnerText: learnerText,
            recastSentences: eval.recastSentences,
            correctlyUsedItemIds: eval.correctlyUsedItemIds,
            failedGrammarPatternIds: eval.failedGrammarPatternIds,
            wordCount: words,
            elapsedSeconds: elapsedSeconds
        )
    }

    // MARK: - 5. Semantic Meaning Audit Grader (Weekly 60-Item Audit & 100-Item Recall)

    public func gradeTypedMeaningBatch(
        items: [(itemId: String, spanish: String, expectedGloss: String, typedEnglish: String)]
    ) async throws -> [String: (correct: Bool, note: String)] {
        struct GradeEntry: Decodable { let itemId: String; let correct: Bool; let note: String }
        struct GradeBatch: Decodable { let grades: [GradeEntry] }

        let lines = items.map {
            "\($0.itemId) | ES: \($0.spanish) | Expected: \($0.expectedGloss) | Learner typed: \"\($0.typedEnglish)\""
        }.joined(separator: "\n")

        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "grades": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "itemId": ["type": "string"],
                            "correct": ["type": "boolean"],
                            "note": ["type": "string"],
                        ],
                        "required": ["itemId", "correct", "note"],
                        "additionalProperties": false,
                    ],
                ],
            ],
            "required": ["grades"],
            "additionalProperties": false,
        ]

        let (batch, _): (GradeBatch, TerraTokenUsage) = try await callStructuredJSON(
            instructions: "Grade each Spanish->English meaning recall fairly. Accept true English synonyms and minor spelling typos; mark blank, wrong, or false-friend guesses false.",
            userInput: lines,
            schemaName: "meaning_audit_grades",
            jsonSchema: schema,
            effort: .low
        )
        return Dictionary(uniqueKeysWithValues: batch.grades.map { ($0.itemId, ($0.correct, $0.note)) })
    }

    // MARK: - Core OpenAI Responses API Transport

    private func callStructuredJSON<T: Decodable>(
        instructions: String,
        userInput: String,
        schemaName: String,
        jsonSchema: [String: Any],
        effort: TerraReasoningEffort
    ) async throws -> (T, TerraTokenUsage) {
        let apiKey = try resolveAPIKey()

        var payload: [String: Any] = [
            "model": Self.modelID,
            "instructions": instructions,
            "input": [
                [
                    "role": "user",
                    "content": userInput,
                ]
            ],
            "text": [
                "format": [
                    "type": "json_schema",
                    "name": schemaName,
                    "strict": true,
                    "schema": jsonSchema,
                ]
            ],
        ]
        if effort != .none {
            payload["reasoning"] = ["effort": effort.rawValue]
        } else {
            payload["reasoning"] = ["effort": "none"]
        }

        let bodyData = try JSONSerialization.data(withJSONObject: payload)
        var request = URLRequest(url: Self.responsesURL)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120
        request.httpBody = bodyData

        // Retry transient 429/5xx with exponential backoff
        var lastError: Error = TerraAPIError.emptyOutput
        for retry in 0..<3 {
            do {
                let (data, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse else {
                    throw TerraAPIError.httpError(-1, "No HTTP response")
                }
                if http.statusCode == 429 || http.statusCode >= 500 {
                    let bodyStr = String(data: data, encoding: .utf8) ?? ""
                    lastError = TerraAPIError.httpError(http.statusCode, bodyStr)
                    try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(retry))) * 1_000_000_000)
                    continue
                }
                guard (200..<300).contains(http.statusCode) else {
                    throw TerraAPIError.httpError(http.statusCode, String(data: data, encoding: .utf8) ?? "")
                }

                let (rawText, usage) = try extractResponsesOutputText(from: data)
                cumulativeInputTokens += usage.inputTokens
                cumulativeOutputTokens += usage.outputTokens

                let decoded = try JSONDecoder().decode(T.self, from: Data(rawText.utf8))
                return (decoded, usage)
            } catch let err as TerraAPIError {
                throw err
            } catch {
                lastError = error
                if retry < 2 {
                    try? await Task.sleep(nanoseconds: UInt64(retry + 1) * 800_000_000)
                }
            }
        }
        throw lastError
    }

    private func extractResponsesOutputText(from data: Data) throws -> (String, TerraTokenUsage) {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TerraAPIError.decodingFailed("Top-level response not a JSON object")
        }

        var usage = TerraTokenUsage(inputTokens: 0, outputTokens: 0, reasoningTokens: 0)
        if let u = root["usage"] as? [String: Any] {
            let inTok = u["input_tokens"] as? Int ?? 0
            let outTok = u["output_tokens"] as? Int ?? 0
            let details = u["output_tokens_details"] as? [String: Any]
            let rTok = details?["reasoning_tokens"] as? Int ?? 0
            usage = TerraTokenUsage(inputTokens: inTok, outputTokens: outTok, reasoningTokens: rTok)
        }

        // Standard OpenAI Responses API fields: `output_text` or `output[].content[].text`
        if let direct = root["output_text"] as? String, !direct.isEmpty {
            return (direct, usage)
        }
        if let outputs = root["output"] as? [[String: Any]] {
            for item in outputs {
                if let contents = item["content"] as? [[String: Any]] {
                    for part in contents {
                        if let text = part["text"] as? String, !text.isEmpty {
                            return (text, usage)
                        }
                    }
                }
            }
        }
        throw TerraAPIError.emptyOutput
    }

    // MARK: - Formatting Helpers

    private func formatViolationsForRewrite(report: VerificationReport, previousText: String) -> String {
        var lines: [String] = []
        for v in report.violations {
            let toks = v.offendingTokens.isEmpty ? "" : " -> Offending: \(v.offendingTokens.joined(separator: ", "))"
            lines.append("- Rule \(v.ruleNumber) (\(v.ruleName)): \(v.detail)\(toks)")
        }
        lines.append("\nPREVIOUS DRAFT TO FIX:\n\(previousText)")
        return lines.joined(separator: "\n")
    }

    private func buildGeneratedPassage(
        draft: RawPassageDraft,
        sentences: [PassageSentence],
        fullText: String,
        plan: PassagePlan,
        report: VerificationReport,
        rewriteAttempts: Int,
        dueWasReduced: Bool
    ) -> GeneratedPassage {
        let newItemMap = Dictionary(uniqueKeysWithValues: plan.newItems.map { ($0.id, $0) })
        let glosses: [NewItemGloss] = plan.newItems.map { item in
            if let g = draft.newItemGlosses.first(where: { $0.itemId == item.id || $0.spanish.lowercased() == item.spanish.lowercased() }) {
                return NewItemGloss(
                    itemId: item.id,
                    spanish: item.spanish,
                    englishGloss: g.englishGloss.isEmpty ? item.englishGloss : g.englishGloss,
                    contextClueNote: g.contextClueNote,
                    visualCueSymbol: item.visualSymbol
                )
            }
            return NewItemGloss(
                itemId: item.id,
                spanish: item.spanish,
                englishGloss: item.englishGloss,
                contextClueNote: "Inferred from passage context",
                visualCueSymbol: item.visualSymbol
            )
        }
        _ = newItemMap

        let probes: [PassageProbe] = draft.probes.enumerated().map { idx, p in
            PassageProbe(
                id: "\(plan.passageId)_probe_\(idx)",
                targetDueItemId: p.targetDueItemId,
                targetDueLemma: p.targetDueLemma,
                questionEn: p.questionEn,
                options: p.options,
                correctOptionIndex: max(0, min(p.correctOptionIndex, max(0, p.options.count - 1))),
                hingeExplanation: p.hingeExplanation
            )
        }

        let grammarNotes: [GrammarMicroNote] = plan.grammarPatterns.compactMap { g in
            guard let note = g.microNoteEn else { return nil }
            return GrammarMicroNote(
                grammarItemId: g.id,
                title: g.spanish,
                noteEn: note,
                contrastAEs: g.contrastAEs ?? "",
                contrastAEn: g.contrastAEn ?? "",
                contrastBEs: g.contrastBEs ?? "",
                contrastBEn: g.contrastBEn ?? ""
            )
        }

        // Flag random 2% of passages for weekly native-speaker audit (PDF p. 7)
        let flaggedAudit = Int.random(in: 1...100) <= 2

        return GeneratedPassage(
            id: plan.passageId,
            sessionNumber: plan.sessionNumber,
            blockType: plan.blockType,
            format: plan.format,
            titleEs: draft.titleEs,
            titleEn: draft.titleEn,
            episodeSummaryForNext: draft.episodeSummaryForNext,
            sentences: sentences,
            fullText: fullText,
            wordCount: report.totalWords,
            newItemGlosses: glosses,
            probes: probes,
            grammarNotes: grammarNotes,
            dueItemIds: plan.dueItems.map(\.id),
            newItemIds: plan.newItems.map(\.id),
            grammarItemIds: plan.grammarPatterns.map(\.id),
            voice: plan.voice,
            accent: plan.accent,
            targetWPM: plan.targetWPM,
            isAudioFirst: plan.isAudioFirst,
            isBootstrap: plan.isBootstrap,
            rewriteAttempts: rewriteAttempts,
            dueListWasReduced: dueWasReduced,
            scheduledWordShare: report.scheduledWordShare,
            flaggedForNativeAudit: flaggedAudit,
            createdAt: Date()
        )
    }
}
