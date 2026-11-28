import Foundation
#if canImport(Security)
import Security
#endif

// MARK: - Keychain API Key & Hugging Face Token Vault

public enum KeychainVault {
    private static let service = "com.aydenbottos.cadencia.openai"
    private static let account = "openai_api_key"
    private static let hfService = "com.aydenbottos.cadencia.huggingface"
    private static let hfAccount = "hf_read_token"

    public static func saveAPIKey(_ key: String) -> Bool {
        saveSecret(key, service: service, account: account)
    }

    public static func loadAPIKey() -> String? {
        if let envKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"], !envKey.isEmpty {
            return envKey
        }
        return loadSecret(service: service, account: account)
    }

    @discardableResult
    public static func deleteAPIKey() -> Bool {
        deleteSecret(service: service, account: account)
    }

    /// Saves the user's own Hugging Face Read Token (`hf_...`) for authenticating gated on-device model downloads
    /// (such as `gradium/phonon-40m-es`) directly from Hugging Face.
    public static func saveHuggingFaceToken(_ token: String) -> Bool {
        saveSecret(token, service: hfService, account: hfAccount)
    }

    public static func loadHuggingFaceToken() -> String? {
        if let env = ProcessInfo.processInfo.environment["HF_TOKEN"], !env.isEmpty {
            return env
        }
        if let env = ProcessInfo.processInfo.environment["HUGGINGFACE_HUB_TOKEN"], !env.isEmpty {
            return env
        }
        return loadSecret(service: hfService, account: hfAccount)
    }

    @discardableResult
    public static func deleteHuggingFaceToken() -> Bool {
        deleteSecret(service: hfService, account: hfAccount)
    }

    private static func saveSecret(_ value: String, service: String, account: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return deleteSecret(service: service, account: account) }
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
        UserDefaults.standard.set(trimmed, forKey: "\(service).\(account)")
        return true
        #endif
    }

    private static func loadSecret(service: String, account: String) -> String? {
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
        return UserDefaults.standard.string(forKey: "\(service).\(account)")
        #endif
    }

    private static func deleteSecret(service: String, account: String) -> Bool {
        #if canImport(Security)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
        #else
        UserDefaults.standard.removeObject(forKey: "\(service).\(account)")
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

    // MARK: - 1. Contract-Enforced Passage Generation (6-Step, Max-3-Generator-Call Pipeline, Protocol pp. 7–8)

    /// Executes the exact revised 6-step checking & generation pipeline (at most 3 generator calls per passage):
    ///  1. Zero-cost Feasibility Check: 3*new + due <= 23% of target length (splits excess to queue for next passage).
    ///  2. Call 1: Generate at standard effort.
    ///  3. Check by deterministic script (Stage A) FIRST, then second model (Stage B) only once Stage A passes.
    ///  4. Call 2 (Repair): Send exact violations & offending sentence indices; rewrite ONLY offending sentences.
    ///     If Call 2 fails with the exact same violation signature as Call 1 -> skip straight to Step 6!
    ///  5. Call 3 (High-Effort Regen): Regenerate once at `.high` effort. Cut due list by 25% ONLY if it is
    ///     non-empty AND the failure was overcrowding (cut items return to the queue).
    ///  6. Fall back: If Call 3 fails (or repeated violation), serve a pre-verified reserve passage and return
    ///     the failure report so the caller logs `passageFallbackServed`.
    public func generateVerifiedPassage(
        plan initialPlan: PassagePlan,
        verifier: ContractVerifier,
        priorContextByItem: [String: Set<String>],
        priorSessionSentenceCountsByItemId: [String: Int] = [:],
        onProgress: (@Sendable (String) -> Void)? = nil
    ) async throws -> (passage: GeneratedPassage, report: VerificationReport) {
        // Step 1: Zero-cost feasibility check (no model call)
        var currentPlan = initialPlan
        if !currentPlan.isBootstrap {
            let feasible = SessionScheduler.enforceFeasibility(
                newItems: currentPlan.newItems,
                dueItems: currentPlan.dueItems,
                targetWords: 215,
                scheduledShareCap: verifier.parameters.scheduledShareCap,
                minNewOccurrences: verifier.parameters.newItemMinOccurrences
            )
            if feasible.didSplit {
                currentPlan = Self.withUpdatedLists(currentPlan, newItems: feasible.newItems, dueItems: feasible.dueItems)
            }
        }

        var dueWasReduced = false
        var callsUsed = 0
        var lastReport: VerificationReport?
        var lastSignature: String?
        var lastDraft: RawPassageDraft?

        // Helper: runs Step 3 (Stage A script -> Stage B model only if Stage A passes)
        func evaluateDraft(_ draft: RawPassageDraft, plan: PassagePlan) async throws -> (sentences: [PassageSentence], fullText: String, report: VerificationReport) {
            let fullText = draft.sentences.map(\.spanish).joined(separator: " ")
            let passageSentences = draft.sentences.enumerated().map { idx, s in
                PassageSentence(
                    id: idx,
                    spanish: s.spanish,
                    englishLine: s.englishLine,
                    wordTokens: verifier.lemmatizer.extractWordTokens(s.spanish),
                    visualSymbol: s.visualSymbol
                )
            }
            let glossedCognates = Set(draft.transparentCognateGlosses.map { $0.spanish.lowercased() })

            // Stage A: Deterministic local script verifier (<10ms)
            var report = verifier.verifyStageA(
                passageText: fullText,
                sentences: passageSentences,
                plan: plan,
                priorContextByItem: priorContextByItem,
                glossedTransparentCognates: glossedCognates,
                priorSessionSentenceCountsByItemId: priorSessionSentenceCountsByItemId
            )
            if !report.passed {
                return (passageSentences, fullText, report)
            }

            // Stage B: Second model checks Rules 5 & 6 + naturalness ONLY once Stage A passes
            onProgress?("Verifying probes & grammar (Stage B)…")
            let stageB = try await verifyStageB(
                fullText: fullText,
                probes: draft.probes,
                plan: plan
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
            }
            return (passageSentences, fullText, report)
        }

        // Step 2 (Call 1): Generate at standard effort
        callsUsed = 1
        onProgress?("Drafting passage (call 1/3)…")
        let draft1 = try await draftPassageOnce(
            plan: currentPlan,
            repairContext: nil,
            effort: currentPlan.isBootstrap ? .low : .medium
        )
        let eval1 = try await evaluateDraft(draft1, plan: currentPlan)
        if eval1.report.passed {
            let passage = buildGeneratedPassage(
                draft: draft1,
                sentences: eval1.sentences,
                fullText: eval1.fullText,
                plan: currentPlan,
                report: eval1.report,
                verifier: verifier,
                generatorCallsUsed: callsUsed,
                dueWasReduced: dueWasReduced
            )
            return (passage, eval1.report)
        }
        lastReport = eval1.report
        lastSignature = eval1.report.violationSignature
        lastDraft = draft1

        // Step 4 (Call 2): Targeted Repair — rewrite ONLY the offending sentences
        callsUsed = 2
        let ruleHint2 = eval1.report.violations.prefix(2).map(\.ruleName).joined(separator: ", ")
        onProgress?("Repairing offending sentences (call 2/3): \(ruleHint2)…")
        let repairPrompt = formatViolationsForTargetedRepair(
            report: eval1.report,
            previousSentences: eval1.sentences
        )
        let draft2 = try await draftPassageOnce(
            plan: currentPlan,
            repairContext: repairPrompt,
            effort: .medium
        )
        let eval2 = try await evaluateDraft(draft2, plan: currentPlan)
        if eval2.report.passed {
            let passage = buildGeneratedPassage(
                draft: draft2,
                sentences: eval2.sentences,
                fullText: eval2.fullText,
                plan: currentPlan,
                report: eval2.report,
                verifier: verifier,
                generatorCallsUsed: callsUsed,
                dueWasReduced: dueWasReduced
            )
            return (passage, eval2.report)
        }

        let sig2 = eval2.report.violationSignature
        lastReport = eval2.report
        lastDraft = draft2

        // Step 6 Early Exit: "The same violation twice in a row skips straight to this step."
        if let prevSig = lastSignature, !prevSig.isEmpty, sig2 == prevSig {
            onProgress?("Repeated violation — serving pre-verified reserve passage…")
            let reserve = ReservePassageBank.selectReserve(
                for: currentPlan,
                lemmatizer: verifier.lemmatizer,
                generatorCallsUsed: callsUsed
            )
            return (reserve, eval2.report)
        }

        // Step 5 (Call 3): Regenerate once at HIGH effort.
        // Cut due list by 25% ONLY if it is non-empty AND the failure was overcrowding!
        if !currentPlan.dueItems.isEmpty && eval2.report.isOvercrowdingFailure {
            let reducedCount = max(1, Int((Double(currentPlan.dueItems.count) * 0.75).rounded(.down)))
            let reducedDue = Array(currentPlan.dueItems.prefix(reducedCount))
            currentPlan = Self.withUpdatedLists(currentPlan, newItems: currentPlan.newItems, dueItems: reducedDue)
            dueWasReduced = true
            onProgress?("Overcrowded — trimming due list 25% & regenerating at high effort (call 3/3)…")
        } else {
            onProgress?("Regenerating once at high effort (call 3/3)…")
        }

        callsUsed = 3
        let regenGuidance = formatViolationsForFullRegen(report: eval2.report)
        let draft3 = try await draftPassageOnce(
            plan: currentPlan,
            repairContext: regenGuidance,
            effort: .high
        )
        let eval3 = try await evaluateDraft(draft3, plan: currentPlan)
        if eval3.report.passed {
            let passage = buildGeneratedPassage(
                draft: draft3,
                sentences: eval3.sentences,
                fullText: eval3.fullText,
                plan: currentPlan,
                report: eval3.report,
                verifier: verifier,
                generatorCallsUsed: callsUsed,
                dueWasReduced: dueWasReduced
            )
            return (passage, eval3.report)
        }
        lastReport = eval3.report
        _ = lastDraft

        // Step 6: Third call failed -> serve pre-verified reserve passage & return failure report for logging!
        onProgress?("Serving pre-verified reserve passage…")
        let reserve = ReservePassageBank.selectReserve(
            for: currentPlan,
            lemmatizer: verifier.lemmatizer,
            generatorCallsUsed: callsUsed
        )
        return (reserve, lastReport ?? eval3.report)
    }

    private static func withUpdatedLists(
        _ plan: PassagePlan,
        newItems: [SyllabusItem],
        dueItems: [SyllabusItem]
    ) -> PassagePlan {
        var allowed = Set(plan.allowedKnownLemmas.map { $0.lowercased() })
        for item in dueItems { allowed.insert(item.spanish.lowercased()) }
        for item in newItems { allowed.insert(item.spanish.lowercased()) }
        return PassagePlan(
            passageId: plan.passageId,
            sessionNumber: plan.sessionNumber,
            passageIndexInSession: plan.passageIndexInSession,
            blockType: plan.blockType,
            isBootstrap: plan.isBootstrap,
            isAudioFirst: plan.isAudioFirst,
            targetWPM: plan.targetWPM,
            format: plan.format,
            topicOrEpisodeTitle: plan.topicOrEpisodeTitle,
            storySummarySoFar: plan.storySummarySoFar,
            dueItems: dueItems,
            newItems: newItems,
            grammarPatterns: plan.grammarPatterns,
            allowedKnownLemmas: Array(allowed).sorted(),
            holdoutBannedLemmas: plan.holdoutBannedLemmas,
            voice: plan.voice,
            accent: plan.accent,
            priorContextWordsByItem: plan.priorContextWordsByItem,
            learningProbeCandidates: plan.learningProbeCandidates,
            transparentCognateLemmas: plan.transparentCognateLemmas,
            bootstrapKitNumber: plan.bootstrapKitNumber,
            bootstrapPatternDescription: plan.bootstrapPatternDescription,
            bootstrapCastNames: plan.bootstrapCastNames,
            isBootstrapRecombination: plan.isBootstrapRecombination,
            sessionUnderExposedItems: plan.sessionUnderExposedItems
        )
    }

    private struct RawDraftSentence: Decodable {
        let spanish: String
        let englishLine: String?
        let visualSymbol: String?
    }
    private struct RawDraftGloss: Decodable {
        let itemId: String
        let spanish: String
        let englishGloss: String
        let contextClueNote: String
    }
    private struct RawTransparentCognateGloss: Decodable {
        let spanish: String
        let englishGloss: String
    }
    private struct RawDraftProbe: Decodable {
        let targetDueItemId: String
        let targetDueLemma: String
        let targetSentenceSpanish: String?
        let questionEn: String
        let options: [String]
        let correctOptionIndex: Int
        let hingeExplanation: String
        let isPictureMatch: Bool
        let optionSymbols: [String]?
    }
    private struct RawPassageDraft: Decodable {
        let titleEs: String
        let titleEn: String
        let episodeSummaryForNext: String
        let sentences: [RawDraftSentence]
        let newItemGlosses: [RawDraftGloss]
        let transparentCognateGlosses: [RawTransparentCognateGloss]
        let probes: [RawDraftProbe]
    }

    private func draftPassageOnce(
        plan: PassagePlan,
        repairContext: String?,
        effort: TerraReasoningEffort
    ) async throws -> RawPassageDraft {
        let newItemsDesc = plan.newItems.map { "\($0.id): \($0.spanish) = \($0.englishGloss) [\($0.pos), icon:\($0.visualSymbol)]" }.joined(separator: "\n")
        let dueItemsDesc = plan.dueItems.map { "\($0.id): \($0.spanish) (\($0.englishGloss))" }.joined(separator: ", ")
        let learningDesc = plan.learningProbeCandidates.map { "\($0.id): \($0.spanish) (\($0.englishGloss))" }.joined(separator: ", ")
        let grammarDesc = plan.grammarPatterns.map { g in
            "\(g.id) [\(g.cefr.rawValue)] \(g.spanish): \(g.microNoteEn ?? "") | Example: \(g.contrastAEs ?? "")"
        }.joined(separator: "\n")

        let priorNeighborsText = plan.priorContextWordsByItem.isEmpty
            ? "None yet."
            : plan.priorContextWordsByItem.map { "\($0.key) — NEVER place near: \($0.value.joined(separator: ", "))" }.joined(separator: "\n")

        let allowedSample = plan.allowedKnownLemmas.prefix(1200).joined(separator: ", ")
        let cognateSample = plan.transparentCognateLemmas.prefix(260).joined(separator: ", ")

        let eligibleTargets = plan.eligibleProbeTargets
        let targetProbeCountDesc: String = {
            if eligibleTargets.isEmpty { return "0 probes (return an empty `probes` array because no eligible items exist)" }
            if eligibleTargets.count == 1 { return "1 probe" }
            if eligibleTargets.count == 2 { return "2 probes" }
            return "2 or 3 probes"
        }()

        let instructions: String
        if plan.isBootstrap {
            let castList = (plan.bootstrapCastNames.isEmpty ? BootstrapKitLibrary.fixedCastDisplayNames : plan.bootstrapCastNames).joined(separator: ", ")
            let kitPattern = plan.bootstrapPatternDescription ?? plan.grammarPatterns.first?.spanish ?? "present tense only"
            let recombNote = plan.isBootstrapRecombination
                ? "RECOMBINATION SCENE (no new items): Recombine ONLY items met so far + fixed cast names (\(castList)), weaving in the UNDER_EXPOSED_SESSION_ITEMS listed below so each reaches >=3 sentences this session."
                : "KIT PASSAGE (Kit \(plan.bootstrapKitNumber ?? 1)/8): EVERY item in NEW_ITEMS (the passage's kit) MUST appear AT LEAST ONCE (aim for 2–3 sentences where natural; NO spacing rule and NO upper limit on uses!)."

            instructions = """
            You are the Spanish Passage Generator inside Cadencia implementing the Fixed-Kit Bootstrap Contract (Sessions 1–4, 17-Page Protocol pp. 8–9):
            1. PASSAGE FORM: Write 6 to 10 simple, grammatical Spanish sentences in PRESENT TENSE ONLY, EACH STRICTLY 8 WORDS OR FEWER! Include an accurate English translation in `englishLine` AND a fitting Apple SF Symbol name in `visualSymbol` (e.g. `stethoscope`, `building.2.fill`, `person.2.fill`, `fork.knife`, `cross.case.fill`, `music.note`, `cup.and.saucer.fill`, `book.fill`) for EVERY sentence.
            2. CLOSED VOCABULARY (>=80% Core + Glossed Cognates): At least 80% of words MUST come from ALLOWED_LEMMAS (items met so far + this kit's NEW_ITEMS + DUE_ITEMS) plus ONLY the fixed cast & place names (\(castList)) and numbers. Do NOT invent any other character or city names! All inflected present-tense/gender/plural forms of allowed lemmas are welcome (e.g. doctor/doctora/doctores, famoso/famosa, un/una, el/la, ser->es/son, estar->está/están, a+el=al, de+el=del). Prefer 100% core words when possible; any extra word (<=20%) MUST be a transparent cognate from TRANSPARENT_COGNATES AND listed in `transparentCognateGlosses`!
            3. \(recombNote)
            4. SCHEDULED SHARE & DUE ITEMS: There is NO scheduled share limit in Bootstrap! If DUE_ITEMS is non-empty, include each at least once and avoid its banned prior neighbors.
            5. PROBES (Whole-Sentence Picture Matching): Return 2 whole-sentence picture-matching probes (`isPictureMatch: true`). For EACH probe: set `targetSentenceSpanish` to one exact Spanish sentence from your passage containing kit/new items, `questionEn` to `"Which picture matches this sentence?"`, `options` to 4 distinct concise English scene descriptions (1 right, 3 clearly different distractors), `optionSymbols` to 4 matching Apple SF Symbol names, `targetDueItemId` to one kit item's id from that sentence, and `targetDueLemma` to that item's Spanish word.
            6. GRAMMAR PATTERN (Present Tense Only): Demonstrate this single pattern clearly at least twice: \(kitPattern).
            7. CONTINUITY: Simple scenes with the fixed cast (\(castList)). NO plot is required — crisp, clear standalone beginner sentences like "Ana es una doctora famosa. El hotel es famoso." are ideal!
            """
        } else {
            let dueRule = plan.dueItems.isEmpty
                ? "3. DUE ITEMS & SCHEDULED SHARE: DUE_ITEMS is empty. Keep NEW_ITEMS occurrences at most 23% of total words (no lower limit)."
                : "3. DUE ITEMS & SCHEDULED SHARE: EVERY item in DUE_ITEMS MUST appear at least once. Combined occurrences of NEW_ITEMS + DUE_ITEMS MUST be AT MOST 23% of total words (there is NO lower limit per passage)."
            let probeRule = eligibleTargets.isEmpty
                ? "5. LOAD-BEARING PROBES: No eligible target items; return an empty `probes` array."
                : "5. LOAD-BEARING PROBES: Write \(targetProbeCountDesc) (`isPictureMatch: false`, `optionSymbols: null`, `targetSentenceSpanish: null`). Each probe MUST hinge on understanding ONE specific target item, drawn in strict priority order: DUE_ITEMS first, then NEW_ITEMS, then LEARNING_CANDIDATES if fewer than 2 due/new items exist. Provide 4 plausible options and 0-based `correctOptionIndex`."
            let grammarRule = plan.grammarPatterns.isEmpty
                ? "6. GRAMMAR: Use natural, idiomatic Spanish."
                : "6. GRAMMAR PATTERNS: Each assigned grammar pattern MUST appear at least TWICE with different lexical words."

            instructions = """
            You are the Spanish Passage Generator inside Cadencia implementing the Full Input Contract (Sessions 5+, Protocol pp. 6–7):
            1. PASSAGE LENGTH & 100% CLOSED VOCABULARY: Write 180–250 natural, vivid Spanish words. EVERY SINGLE word lemma MUST come strictly from ALLOWED_LEMMAS (Known, Learning p>85%, DUE_ITEMS, NEW_ITEMS) or proper names/numbers — 100% compliance, ZERO off-list lemmas, ZERO banned hold-out words! Return `transparentCognateGlosses: []` and `visualSymbol: null`.
            2. NEW ITEMS (at most 6): If NEW_ITEMS is non-empty, EVERY item in NEW_ITEMS MUST appear at least 3 times, with 3 of those uses at least 30 words apart (further uses are free and never fail the rule!). Make the first use clearly guessable from context.
            \(dueRule)
            4. FRESH CONTEXT: Never place any due item within ±3 words of its banned prior neighbor words listed below.
            \(probeRule)
            \(grammarRule)
            7. CONTINUITY: Continue the serial episode or write a gripping short non-fiction piece matching the topic with authentic Spanish register (\(plan.accent.displayName)).
            """
        }

        let underExposedDesc = plan.sessionUnderExposedItems.map { "\($0.spanish) (\($0.englishGloss))" }.joined(separator: ", ")

        var userPrompt = """
        MODE: \(plan.isBootstrap ? (plan.isBootstrapRecombination ? "BOOTSTRAP RECOMBINATION (Sessions 1–4)" : "BOOTSTRAP FIXED KIT \(plan.bootstrapKitNumber ?? 1)/8 (Sessions 1–4)") : "FULL CONTRACT (Sessions 5+)")
        FORMAT: \(plan.format.rawValue)
        TITLE / TOPIC: \(plan.topicOrEpisodeTitle)
        \(plan.isBootstrap ? "FIXED CAST & PLACES (ONLY names allowed): \((plan.bootstrapCastNames.isEmpty ? BootstrapKitLibrary.fixedCastDisplayNames : plan.bootstrapCastNames).joined(separator: ", "))" : "STORY SO FAR: \(plan.storySummarySoFar.isEmpty ? "Opening episode — establish vivid setting, protagonist, and immediate hook." : plan.storySummarySoFar)")
        REGIONAL ACCENT FLAVOR: \(plan.accent.displayName)

        NEW_ITEMS (\(plan.isBootstrap ? "every kit item >=1x in this passage, no spacing limit" : "each >=3x with 3 uses >=30 words apart")):
        \(newItemsDesc.isEmpty ? "None this passage." : newItemsDesc)

        DUE_ITEMS (each >=1x):
        \(dueItemsDesc.isEmpty ? "None." : dueItemsDesc)
        """

        if plan.isBootstrap && !underExposedDesc.isEmpty {
            userPrompt += "\n\nUNDER_EXPOSED_SESSION_ITEMS (weave into sentences so they reach >=3 sentences this session):\n\(underExposedDesc)"
        }

        userPrompt += """

        LEARNING_CANDIDATES (tier-3 probe fallback only if due + new < 2):
        \(learningDesc.isEmpty ? "None." : learningDesc)

        ASSIGNED_GRAMMAR_PATTERNS (each >=2x with different words):
        \(plan.isBootstrap ? (plan.bootstrapPatternDescription ?? (grammarDesc.isEmpty ? "Present tense only." : grammarDesc)) : (grammarDesc.isEmpty ? "None assigned." : grammarDesc))

        RULE 4 BANNED PRIOR NEIGHBORS:
        \(priorNeighborsText)

        ALLOWED_LEMMAS (\(plan.isBootstrap ? ">=80% of words must come from this set + fixed cast names/numbers" : "100% of words must come from this set + names/numbers")):
        \(allowedSample)
        """

        if plan.isBootstrap {
            userPrompt += "\n\nTRANSPARENT_COGNATES (only allowed for the glossed <=20% remainder; gloss any you use in `transparentCognateGlosses`):\n\(cognateSample)"
        }

        if let repair = repairContext {
            userPrompt += "\n\n\(repair)"
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
                            "visualSymbol": ["type": ["string", "null"]],
                        ],
                        "required": ["spanish", "englishLine", "visualSymbol"],
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
                "transparentCognateGlosses": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "spanish": ["type": "string"],
                            "englishGloss": ["type": "string"],
                        ],
                        "required": ["spanish", "englishGloss"],
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
                            "targetSentenceSpanish": ["type": ["string", "null"]],
                            "questionEn": ["type": "string"],
                            "options": ["type": "array", "items": ["type": "string"]],
                            "correctOptionIndex": ["type": "integer"],
                            "hingeExplanation": ["type": "string"],
                            "isPictureMatch": ["type": "boolean"],
                            "optionSymbols": ["type": ["array", "null"], "items": ["type": "string"]],
                        ],
                        "required": [
                            "targetDueItemId", "targetDueLemma", "targetSentenceSpanish", "questionEn", "options",
                            "correctOptionIndex", "hingeExplanation", "isPictureMatch", "optionSymbols",
                        ],
                        "additionalProperties": false,
                    ],
                ],
            ],
            "required": [
                "titleEs", "titleEn", "episodeSummaryForNext", "sentences",
                "newItemGlosses", "transparentCognateGlosses", "probes",
            ],
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
        let probeText = probes.isEmpty
            ? "(No probes)"
            : probes.enumerated().map { idx, p in
                let sentNote = p.targetSentenceSpanish.map { " | Sentence: «\($0)»" } ?? ""
                return "Probe \(idx + 1) [target: \(p.targetDueLemma), pictureMatch: \(p.isPictureMatch)\(sentNote)]: \(p.questionEn) | Options: \(p.options.joined(separator: " / ")) | Correct: \(p.correctOptionIndex)"
            }.joined(separator: "\n")
        let gramText = plan.isBootstrap
            ? (plan.bootstrapPatternDescription ?? plan.grammarPatterns.map { "\($0.id): \($0.spanish)" }.joined(separator: "\n"))
            : plan.grammarPatterns.map { "\($0.id): \($0.spanish)" }.joined(separator: "\n")
        let eligible = plan.eligibleProbeTargets
        let targetList = eligible.map { "\($0.spanish) (\($0.englishGloss))" }.joined(separator: ", ")

        let probeCheckRule: String
        if plan.isBootstrap {
            probeCheckRule = "1. Rule 5 (Bootstrap Whole-Sentence Picture Matching): Are there 2–3 probes where each presents a whole Spanish sentence from the passage and 4 distinct English picture/scene options with 1 unambiguously right match?"
        } else if eligible.isEmpty {
            probeCheckRule = "1. Rule 5 (Probes): No eligible due/new/learning target items exist, so 0 probes is expected; set probesAreLoadBearing=true."
        } else {
            let minNeeded = min(2, eligible.count)
            probeCheckRule = "1. Rule 5 (Load-bearing probes): Are there \(minNeeded)–3 probes (fewer eligible items means fewer probes), and does each hinge on understanding one specific eligible target item (priority: due list first, then new list, then Learning items)?"
        }

        let grammarCheckRule = gramText.isEmpty
            ? "2. Rule 6 (Grammar patterns): None assigned for this passage; set grammarPatternsVerified=true."
            : (plan.isBootstrap
                ? "2. Rule 6 (Bootstrap Present-Tense Pattern): Does the passage demonstrate the single assigned present-tense pattern (\(gramText)) cleanly in present tense?"
                : "2. Rule 6 (Grammar patterns): Does each assigned grammar pattern appear at least twice with different lexical words?")

        let bootstrapContext = plan.isBootstrap
            ? "\nIMPORTANT BOOTSTRAP CONTEXT: This is a Sessions 1–4 fixed-kit beginner scene built from ONLY a 6–8 word kit and the fixed cast (Ana, Luis, Carla; Perú, México). Present tense only; NO plot is required! Simple, repetitive sentences like 'Ana es una doctora famosa. El hotel es famoso.' are 100% INTENTIONAL and PASS naturalness as long as the Spanish grammar and agreement are accurate."
            : ""

        let instructions = """
        You are an independent, fair native-level Spanish Linguistic Verifier (Stage B).
        Evaluate ONLY three things:
        \(probeCheckRule)
        \(grammarCheckRule)
        3. Naturalness: Is the Spanish grammatically accurate, clear, and coherent?\(bootstrapContext)
        Return passed=true if all three checks are solid; otherwise return passed=false with concrete fix instructions.
        """
        let input = """
        MODE: \(plan.isBootstrap ? "Bootstrap Micro-Story (Sessions 1–4)" : "Full Passage (Sessions 5+)")
        ELIGIBLE PROBE TARGETS (priority order: due -> new -> learning): \(targetList.isEmpty ? "None" : targetList)

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

    // MARK: - 3. Closed-Vocabulary Bilingual Voice Speaking Partner (Output 1 Retell & Output 2 Open Talk / SIELE)

    public struct SpeakingPartnerResponse: Decodable, Sendable {
        public let recastCorrectionEs: String?
        public let partnerReplyEs: String
        public let partnerReplyEn: String?
        public let spokenSegments: [TutorSpeechSegment]
        public let followUpPromptEnHint: String?
        public let correctlyUsedLemmas: [String]
        public let failedGrammarPatternIds: [String]

        public init(
            recastCorrectionEs: String?,
            partnerReplyEs: String,
            partnerReplyEn: String?,
            spokenSegments: [TutorSpeechSegment],
            followUpPromptEnHint: String?,
            correctlyUsedLemmas: [String],
            failedGrammarPatternIds: [String]
        ) {
            self.recastCorrectionEs = recastCorrectionEs
            self.partnerReplyEs = partnerReplyEs
            self.partnerReplyEn = partnerReplyEn
            self.spokenSegments = spokenSegments
            self.followUpPromptEnHint = followUpPromptEnHint
            self.correctlyUsedLemmas = correctlyUsedLemmas
            self.failedGrammarPatternIds = failedGrammarPatternIds
        }

        /// Returns clean, non-empty ordered segments for dual-engine synthesis (`es` -> Phonon-40m-es, `en` -> Kokoro-82M).
        public func resolvedSpokenSegments(allowEnglish: Bool) -> [TutorSpeechSegment] {
            let filtered = spokenSegments
                .map { TutorSpeechSegment(id: $0.id, language: $0.language, text: $0.text.trimmingCharacters(in: .whitespacesAndNewlines)) }
                .filter { !$0.text.isEmpty && (allowEnglish || $0.language == .spanish) }
            if !filtered.isEmpty { return filtered }

            var fallback: [TutorSpeechSegment] = []
            if allowEnglish, let en = partnerReplyEn?.trimmingCharacters(in: .whitespacesAndNewlines), !en.isEmpty {
                fallback.append(TutorSpeechSegment(language: .english, text: en))
            }
            let es = partnerReplyEs.trimmingCharacters(in: .whitespacesAndNewlines)
            if !es.isEmpty {
                fallback.append(TutorSpeechSegment(language: .spanish, text: es))
            }
            return fallback
        }
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
        conversationHistory: [(role: String, text: String)],
        allowEnglishTutor: Bool = true
    ) async throws -> SpeakingPartnerResponse {
        let dueNames = targetDueItems.map(\.spanish).joined(separator: ", ")
        let steerNames = steeringProductiveItems.map { "\($0.spanish) (\($0.englishGloss))" }.joined(separator: ", ")
        let gramList = scheduledGrammarPatterns.map { "\($0.id): \($0.spanish)" }.joined(separator: "\n")
        let examNote = (mode == .output2 && week >= 5)
            ? "WEEK 5+ SIELE EXAM FORMAT: Frame turns like SIELE Oral Expression & Interaction (monologue description, negotiating an everyday situation, or defending an opinion)."
            : ""

        let bilingualRule = allowEnglishTutor
            ? """
            - BILINGUAL VOICE TUTOR (SPANISH + ENGLISH):
              You may speak BOTH Spanish (`es`, voiced by Phonon-40m-es) AND English (`en`, voiced by Kokoro-82M)!
              * Always keep `partnerReplyEs` in natural Spanish (1–2 sentences using ALLOWED_VOCABULARY) to move the Spanish conversation forward.
              * Put concise, helpful spoken English in `partnerReplyEn` whenever it helps the learner: e.g. answering an English question they asked, giving a brief warm coaching tip, clarifying a word/phrase, or offering a quick pronunciation/phrasing nudge (keep it brief, 1–2 spoken sentences, or null if pure Spanish flows best).
              * Populate `spokenSegments` in the EXACT natural spoken order you want the tutor to speak aloud (each item has `language`: `"es"` or `"en"`, and `text`). Never mix Spanish and English inside the same segment so each engine speaks its native language cleanly!
              * If the learner speaks English or asks "how do you say…", answer concisely in an `"en"` segment and model the target phrase in an `"es"` segment!
            """
            : """
            - SPANISH-ONLY IMMERSION MODE:
              Keep all spoken output 100% Spanish (`partnerReplyEn` MUST be null, and `spokenSegments` MUST contain only `"es"` segments).
            """

        let instructions = """
        You are Cadencia's warm, natural Spanish Voice Tutor under the Signal-Starvation protocol:
        - STRICT CLOSED VOCABULARY FOR SPANISH: Every Spanish sentence MUST use vocabulary from ALLOWED_VOCABULARY plus names/numbers. Keep Spanish replies concise (1–3 natural spoken Spanish sentences) so conversation flows briskly.
        \(bilingualRule)
        - RECAST CORRECTIONS (PDF p. 9): If the learner made a Spanish grammar or lexical error, put the clean corrected version of their sentence in `recastCorrectionEs` and weave it naturally into your Spanish turn — keep feedback brief and conversational, never a long grammar lecture!
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
                "partnerReplyEn": ["type": ["string", "null"]],
                "spokenSegments": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "language": ["type": "string", "enum": ["es", "en"]],
                            "text": ["type": "string"],
                        ],
                        "required": ["language", "text"],
                        "additionalProperties": false,
                    ],
                ],
                "followUpPromptEnHint": ["type": ["string", "null"]],
                "correctlyUsedLemmas": ["type": "array", "items": ["type": "string"]],
                "failedGrammarPatternIds": ["type": "array", "items": ["type": "string"]],
            ],
            "required": [
                "recastCorrectionEs",
                "partnerReplyEs",
                "partnerReplyEn",
                "spokenSegments",
                "followUpPromptEnHint",
                "correctlyUsedLemmas",
                "failedGrammarPatternIds",
            ],
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

    // MARK: - Formatting Helpers (Step 4 Targeted Repair & Step 5 High-Effort Regen)

    private func formatViolationsForTargetedRepair(
        report: VerificationReport,
        previousSentences: [PassageSentence]
    ) -> String {
        let offendingSet = Set(report.offendingSentenceIndices)
        var lines: [String] = ["STEP 4 TARGETED REPAIR — REWRITE ONLY THE OFFENDING SENTENCES BELOW (keep clean sentences unchanged!):"]
        for v in report.violations {
            let toks = v.offendingTokens.isEmpty ? "" : " -> Offending: \(v.offendingTokens.joined(separator: ", "))"
            let sentNums = v.offendingSentenceIndices.isEmpty ? "" : " [Sentences: \(v.offendingSentenceIndices.map { "#\($0 + 1)" }.joined(separator: ", "))]"
            lines.append("- Rule \(v.ruleNumber) (\(v.ruleName))\(sentNums): \(v.detail)\(toks)")
        }
        lines.append("\nPREVIOUS NUMBERED SENTENCES (preserve all sentences marked [KEEP]; rewrite only [REPAIR]):")
        for (idx, s) in previousSentences.enumerated() {
            let tag = offendingSet.isEmpty || offendingSet.contains(idx) ? "[REPAIR]" : "[KEEP]"
            lines.append("S\(idx + 1) \(tag): \(s.spanish)")
        }
        return lines.joined(separator: "\n")
    }

    private func formatViolationsForFullRegen(report: VerificationReport) -> String {
        var lines: [String] = ["STEP 5 HIGH-EFFORT REGENERATION — Write a fresh passage cleanly avoiding ALL previous violations:"]
        for v in report.violations {
            let toks = v.offendingTokens.isEmpty ? "" : " -> Avoid: \(v.offendingTokens.joined(separator: ", "))"
            lines.append("- Rule \(v.ruleNumber) (\(v.ruleName)): \(v.detail)\(toks)")
        }
        return lines.joined(separator: "\n")
    }

    private func buildGeneratedPassage(
        draft: RawPassageDraft,
        sentences: [PassageSentence],
        fullText: String,
        plan: PassagePlan,
        report: VerificationReport,
        verifier: ContractVerifier,
        generatorCallsUsed: Int,
        dueWasReduced: Bool
    ) -> GeneratedPassage {
        var glosses: [NewItemGloss] = plan.newItems.map { item in
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
        // Include glossed transparent cognates for Bootstrap tap-to-gloss reference
        let existingLower = Set(glosses.map { $0.spanish.lowercased() })
        for cog in draft.transparentCognateGlosses where !existingLower.contains(cog.spanish.lowercased()) {
            glosses.append(NewItemGloss(
                itemId: "cog_\(cog.spanish.lowercased())",
                spanish: cog.spanish,
                englishGloss: cog.englishGloss,
                contextClueNote: "Transparent cognate (Bootstrap)",
                visualCueSymbol: "sparkles"
            ))
        }

        let probes: [PassageProbe] = draft.probes.enumerated().map { idx, p in
            let targetSentence = p.targetSentenceSpanish?.trimmingCharacters(in: .whitespacesAndNewlines)
            // Bootstrap Rule 5: log whole-sentence picture-matching result against EACH new item in that sentence!
            let matchedSentenceItemIds: [String]
            if plan.isBootstrap, let sent = targetSentence, !sent.isEmpty {
                let candidates = plan.newItems.isEmpty ? plan.dueItems : plan.newItems
                let ids = verifier.matchedItemIds(in: sent, from: candidates)
                matchedSentenceItemIds = ids.isEmpty ? [p.targetDueItemId] : ids
            } else {
                matchedSentenceItemIds = [p.targetDueItemId]
            }
            return PassageProbe(
                id: "\(plan.passageId)_probe_\(idx)",
                targetDueItemId: matchedSentenceItemIds.first ?? p.targetDueItemId,
                targetDueLemma: p.targetDueLemma,
                questionEn: p.questionEn,
                options: p.options,
                correctOptionIndex: max(0, min(p.correctOptionIndex, max(0, p.options.count - 1))),
                hingeExplanation: p.hingeExplanation,
                isPictureMatch: p.isPictureMatch,
                optionSymbols: p.optionSymbols,
                targetSentenceSpanish: targetSentence,
                targetItemIds: matchedSentenceItemIds
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

        // Flag random 2% of passages for weekly native-speaker audit (PDF p. 8)
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
            rewriteAttempts: max(0, generatorCallsUsed - 1),
            generatorCallsUsed: generatorCallsUsed,
            isReserveFallback: false,
            dueListWasReduced: dueWasReduced,
            scheduledWordShare: report.scheduledWordShare,
            flaggedForNativeAudit: flaggedAudit,
            createdAt: Date()
        )
    }
}
