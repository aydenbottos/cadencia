import Foundation

/// Implements the automatic Stage A (deterministic local) Input Contract Verifier
/// from the revised 16-page Signal-Starvation Hypothesis Test Protocol (pp. 6–8):
///
/// **Full Contract (Sessions 5+):**
/// - Passage form: 180–250 words.
/// - Rule 1 (Closed vocabulary): 100% of lemmas belong to the 5 allowed groups: Known,
///   Learning with p > 85%, this passage's due list, this passage's new list, or names/numbers.
///   ZERO tolerance for hold-out lemmas.
/// - Rule 2 (New items): at most 6 per passage; each appears >= 3 times with 3 of those uses
///   at least 30 words apart. Further uses are free and never fail the rule.
/// - Rule 3 (Due items & Scheduled share): every item on the due list appears >= 1 time;
///   new + due items together make up at most 23% of running words. A single passage has NO lower limit.
/// - Rule 4 (Fresh context): a due item never appears beside the same content words (±3 window)
///   as in an earlier meeting.
/// - Empty-list rule: a rule about a list that is empty passes automatically.
///
/// **Bootstrap Contract (Sessions 1–4):**
/// - Passage form: 6–10 sentences of <= 8 words each, each with a picture and an English line on tap.
/// - Rule 1 (Closed vocabulary): >= 80% of words are items met so far, items on the passage's
///   new/due list, names, or numbers. The rest (<= 20%) MUST be glossed transparent cognates.
///   ZERO tolerance for hold-out lemmas.
/// - Rule 2 (New items): at most 8 per passage; each appears in at least 3 different sentences
///   during the session. No spacing rule and no upper limit on uses.
/// - Rule 3 (Scheduled share): NO limit (neither cap nor floor). Due items (if any) appear >= 1x.
/// - Rule 4 (Fresh context): applies unchanged.
public struct ContractVerifier: Sendable {
    public let lemmatizer: SpanishLemmatizer
    public let parameters: ProtocolParameters

    public init(lemmatizer: SpanishLemmatizer, parameters: ProtocolParameters = .protocolDefaults) {
        self.lemmatizer = lemmatizer
        self.parameters = parameters
    }

    public func verifyStageA(
        passageText: String,
        sentences: [PassageSentence],
        plan: PassagePlan,
        priorContextByItem: [String: Set<String>],
        glossedTransparentCognates: Set<String> = [],
        priorSessionSentenceCountsByItemId: [String: Int] = [:]
    ) -> VerificationReport {
        // Analyze sentence-by-sentence so every violation carries exact 0-based offending sentence indices
        // for Step 4 targeted sentence repair!
        var tokens: [AnalyzedToken] = []
        var tokenSentenceIndex: [Int] = []
        var sentenceWordCounts: [Int] = []

        var globalWordIdx = 0
        for (sIdx, sentence) in sentences.enumerated() {
            let sentToks = lemmatizer.analyze(sentence.spanish)
            sentenceWordCounts.append(sentToks.count)
            for tok in sentToks {
                tokens.append(AnalyzedToken(
                    surface: tok.surface,
                    normalized: tok.normalized,
                    lemma: tok.lemma,
                    pos: tok.pos,
                    isProperNameOrNumber: tok.isProperNameOrNumber,
                    isContentWord: tok.isContentWord,
                    wordIndex: globalWordIdx
                ))
                tokenSentenceIndex.append(sIdx)
                globalWordIdx += 1
            }
        }
        if sentences.isEmpty {
            let fallbackToks = lemmatizer.analyze(passageText)
            tokens = fallbackToks
            tokenSentenceIndex = Array(repeating: 0, count: fallbackToks.count)
        }

        let totalWords = tokens.count
        var violations: [ContractViolation] = []

        // 0. Passage form check
        if plan.isBootstrap {
            // Bootstrap (p. 8): 6 to 10 sentences of 8 words or fewer
            if sentences.count < 6 || sentences.count > 10 {
                violations.append(ContractViolation(
                    ruleNumber: 0,
                    ruleName: "Bootstrap Sentence Count (6–10)",
                    detail: "Bootstrap passage has \(sentences.count) sentence(s); required 6–10 sentences.",
                    offendingTokens: [],
                    offendingSentenceIndices: Array(0..<sentences.count)
                ))
            }
            var overlongSentences: [String] = []
            var overlongIndices: [Int] = []
            for (idx, count) in sentenceWordCounts.enumerated() where count > 8 {
                overlongSentences.append("S\(idx + 1) (\(count) words): \(sentences[idx].spanish)")
                overlongIndices.append(idx)
            }
            if !overlongIndices.isEmpty {
                violations.append(ContractViolation(
                    ruleNumber: 0,
                    ruleName: "Bootstrap Sentence Length (<=8 words)",
                    detail: "\(overlongIndices.count) bootstrap sentence(s) exceeded 8 words.",
                    offendingTokens: Array(overlongSentences.prefix(4)),
                    offendingSentenceIndices: overlongIndices
                ))
            }
        } else {
            // Full contract (p. 6): 180 to 250 words
            if totalWords < parameters.passageMinWords || totalWords > parameters.passageMaxWords {
                violations.append(ContractViolation(
                    ruleNumber: 0,
                    ruleName: "Passage Length (\(parameters.passageMinWords)–\(parameters.passageMaxWords) words)",
                    detail: "Passage has \(totalWords) words; required \(parameters.passageMinWords)–\(parameters.passageMaxWords).",
                    offendingTokens: [],
                    offendingSentenceIndices: sentences.isEmpty ? [] : [max(0, sentences.count - 1)]
                ))
            }
        }

        // Build lookup sets
        let allowedSet = Set(plan.allowedKnownLemmas.map { $0.lowercased() })
        let holdoutSet = Set(plan.holdoutBannedLemmas.map { $0.lowercased() })
        let cognateSet = Set(plan.transparentCognateLemmas.map { $0.lowercased() })
            .union(glossedTransparentCognates.map { $0.lowercased() })

        let dueLexical = plan.dueItems.filter { $0.itemType == .lemma }
        let dueChunks = plan.dueItems.filter { $0.itemType == .chunk }
        let newLexical = plan.newItems.filter { $0.itemType == .lemma }
        let newChunks = plan.newItems.filter { $0.itemType == .chunk }

        let dueLemmaToItem = Dictionary(uniqueKeysWithValues: dueLexical.map { ($0.spanish.lowercased(), $0) })
        let newLemmaToItem = Dictionary(uniqueKeysWithValues: newLexical.map { ($0.spanish.lowercased(), $0) })

        // Multi-word chunk matches
        let chunkMatches = lemmatizer.findChunks(in: tokens)
        var chunkPositionsById: [String: [Int]] = [:]
        var chunkSentencesById: [String: Set<Int>] = [:]
        var chunkCoveredWordIndices: Set<Int> = []
        for m in chunkMatches {
            chunkPositionsById[m.itemId, default: []].append(m.startIndex)
            if m.startIndex < tokenSentenceIndex.count {
                chunkSentencesById[m.itemId, default: []].insert(tokenSentenceIndex[m.startIndex])
            }
            for k in m.startIndex..<(m.startIndex + m.length) {
                chunkCoveredWordIndices.insert(k)
            }
        }

        // Track occurrences & Rule 1 closed vocabulary
        var holdoutLeaks: [String] = []
        var holdoutSentenceIndices: Set<Int> = []
        var offListLemmas: [String] = []
        var offListSentenceIndices: Set<Int> = []
        var nonCognateBootstrapOffList: [String] = []
        var nonCognateSentenceIndices: Set<Int> = []

        var coreCompliantCount = 0
        var scheduledTokenIndices: Set<Int> = []
        var lemmaPositions: [String: [Int]] = [:]
        var surfacePositions: [String: [Int]] = [:]
        var lemmaSentences: [String: Set<Int>] = [:]
        var surfaceSentences: [String: Set<Int>] = [:]

        for tok in tokens {
            let lem = tok.lemma.lowercased()
            let norm = tok.normalized
            let sIdx = tok.wordIndex < tokenSentenceIndex.count ? tokenSentenceIndex[tok.wordIndex] : 0

            lemmaPositions[lem, default: []].append(tok.wordIndex)
            surfacePositions[norm, default: []].append(tok.wordIndex)
            lemmaSentences[lem, default: []].insert(sIdx)
            surfaceSentences[norm, default: []].insert(sIdx)

            if holdoutSet.contains(lem) && !allowedSet.contains(lem) {
                holdoutLeaks.append(tok.surface)
                holdoutSentenceIndices.insert(sIdx)
            }

            let isScheduledLemma = dueLemmaToItem[lem] != nil
                || newLemmaToItem[lem] != nil
                || dueLemmaToItem[norm] != nil
                || newLemmaToItem[norm] != nil
            if isScheduledLemma {
                scheduledTokenIndices.insert(tok.wordIndex)
            }

            let isCoreAllowed = tok.isProperNameOrNumber
                || allowedSet.contains(lem)
                || allowedSet.contains(norm)
                || Self.isArticleOrContractionAllowed(surface: norm, lemma: lem, allowedSet: allowedSet)
                || chunkCoveredWordIndices.contains(tok.wordIndex)

            if isCoreAllowed {
                coreCompliantCount += 1
            } else {
                offListLemmas.append("\(tok.surface) (\(lem))")
                offListSentenceIndices.insert(sIdx)
                if plan.isBootstrap {
                    let isAllowedCognate = !holdoutSet.contains(lem)
                        && (cognateSet.contains(lem) || cognateSet.contains(norm))
                    if !isAllowedCognate {
                        nonCognateBootstrapOffList.append("\(tok.surface) (\(lem))")
                        nonCognateSentenceIndices.insert(sIdx)
                    }
                }
            }
        }

        // Add chunk tokens belonging to due/new chunks to scheduled share
        let scheduledChunkIds = Set((dueChunks + newChunks).map(\.id))
        for m in chunkMatches where scheduledChunkIds.contains(m.itemId) {
            for k in m.startIndex..<(m.startIndex + m.length) {
                scheduledTokenIndices.insert(k)
            }
        }

        // Hold-out zero-tolerance check (all sessions)
        if !holdoutLeaks.isEmpty {
            violations.append(ContractViolation(
                ruleNumber: 1,
                ruleName: "Hold-Out Lemma Guard",
                detail: "Banned hold-out lemmas appeared in the passage. Remove them completely.",
                offendingTokens: Array(Set(holdoutLeaks)).sorted(),
                offendingSentenceIndices: Array(holdoutSentenceIndices).sorted()
            ))
        }

        // Rule 1: Closed Vocabulary
        //  - Sessions 5+: 100% of words belong to the 5 allowed groups (Known, Learning p>85%, due, new, names/numbers).
        //  - Bootstrap (Sessions 1–4): >= 80% are met-so-far / new / due / names / numbers; the rest MUST be glossed transparent cognates.
        let closedCompliance = totalWords > 0 ? Double(coreCompliantCount) / Double(totalWords) : 0.0
        if plan.isBootstrap {
            if closedCompliance < 0.80 {
                violations.append(ContractViolation(
                    ruleNumber: 1,
                    ruleName: "Bootstrap Closed Vocabulary (>=80%)",
                    detail: String(format: "Met/new/name word share %.1f%% < required 80%%. Use more items from the passage's new/met lists.", closedCompliance * 100),
                    offendingTokens: Array(Set(offListLemmas)).sorted().prefix(12).map { $0 },
                    offendingSentenceIndices: Array(offListSentenceIndices).sorted()
                ))
            }
            if !nonCognateBootstrapOffList.isEmpty {
                violations.append(ContractViolation(
                    ruleNumber: 1,
                    ruleName: "Bootstrap Cognate Remainder",
                    detail: "Words outside met/new/names in Bootstrap must be glossed transparent cognates. Replace non-cognate off-list words.",
                    offendingTokens: Array(Set(nonCognateBootstrapOffList)).sorted().prefix(12).map { $0 },
                    offendingSentenceIndices: Array(nonCognateSentenceIndices).sorted()
                ))
            }
        } else if closedCompliance < 1.0 {
            let uniqueOff = Array(Set(offListLemmas)).sorted()
            violations.append(ContractViolation(
                ruleNumber: 1,
                ruleName: "Closed Vocabulary (100%)",
                detail: String(format: "Closed vocabulary compliance %.1f%% < required 100%%. Replace EVERY off-list lemma with an allowed word.", closedCompliance * 100),
                offendingTokens: Array(uniqueOff.prefix(15)),
                offendingSentenceIndices: Array(offListSentenceIndices).sorted()
            ))
        }

        // Rule 2: New Items (passes automatically when newItems is empty!)
        var underSpacedNew: [String] = []
        if !plan.newItems.isEmpty {
            let maxNewAllowed = plan.isBootstrap
                ? parameters.maxNewItemsPerBootstrapPassage
                : parameters.maxNewItemsPerPassage
            if plan.newItems.count > maxNewAllowed {
                violations.append(ContractViolation(
                    ruleNumber: 2,
                    ruleName: "New Items Cap",
                    detail: "Too many new items (\(plan.newItems.count) > \(maxNewAllowed)).",
                    offendingTokens: plan.newItems.map(\.spanish)
                ))
            }

            if plan.isBootstrap {
                // Bootstrap Rule 2 (p. 8): each new item appears in >= 3 different sentences (no spacing rule, no upper limit)
                for item in plan.newItems {
                    let distinctSentCount: Int
                    if item.itemType == .chunk {
                        distinctSentCount = (chunkSentencesById[item.id] ?? []).count
                    } else {
                        let key = item.spanish.lowercased()
                        distinctSentCount = Self.matchedSentences(for: key, lemmaSentences: lemmaSentences, surfaceSentences: surfaceSentences).count
                    }
                    let cumulative = (priorSessionSentenceCountsByItemId[item.id] ?? 0) + distinctSentCount
                    if cumulative < parameters.newItemMinOccurrences {
                        underSpacedNew.append("\(item.spanish) (\(cumulative)/\(parameters.newItemMinOccurrences) distinct sentences)")
                    }
                }
                if !underSpacedNew.isEmpty {
                    violations.append(ContractViolation(
                        ruleNumber: 2,
                        ruleName: "Bootstrap New Items (>=3 Sentences)",
                        detail: "Every bootstrap new item must appear in at least \(parameters.newItemMinOccurrences) different sentences (no spacing limit).",
                        offendingTokens: underSpacedNew,
                        offendingSentenceIndices: sentences.isEmpty ? [] : [0, sentences.count / 2, max(0, sentences.count - 1)]
                    ))
                }
            } else {
                // Full Contract Rule 2 (p. 6): each appears >= 3 times with 3 of those uses >= 30 words apart; extra uses are free!
                let minSpacing = parameters.newItemMinWordSpacing
                for item in plan.newItems {
                    let positions: [Int]
                    if item.itemType == .chunk {
                        positions = (chunkPositionsById[item.id] ?? []).sorted()
                    } else {
                        let key = item.spanish.lowercased()
                        positions = Self.matchedPositions(for: key, lemmaPositions: lemmaPositions, surfacePositions: surfacePositions)
                    }
                    if positions.count < parameters.newItemMinOccurrences {
                        underSpacedNew.append("\(item.spanish) (count \(positions.count)/\(parameters.newItemMinOccurrences))")
                    } else {
                        let spacedCount = Self.maxSpacedOccurrences(positions: positions, minSpacing: minSpacing)
                        if spacedCount < parameters.newItemMinOccurrences {
                            underSpacedNew.append("\(item.spanish) (only \(spacedCount)/\(parameters.newItemMinOccurrences) uses spaced >=\(minSpacing) words apart)")
                        }
                    }
                }
                if !underSpacedNew.isEmpty {
                    violations.append(ContractViolation(
                        ruleNumber: 2,
                        ruleName: "New Items Repetition & Spacing",
                        detail: "Every new item must appear at least \(parameters.newItemMinOccurrences) times, with 3 uses spaced >=\(minSpacing) words apart (extra uses are free).",
                        offendingTokens: underSpacedNew,
                        offendingSentenceIndices: sentences.isEmpty ? [] : [0, sentences.count / 2, max(0, sentences.count - 1)]
                    ))
                }
            }
        }

        // Rule 3: Due Items & Scheduled Share
        //  - Every due item appears >= 1x (passes automatically if dueItems is empty).
        //  - From Session 5+: combined new + due <= 23% of running words; NO lower limit per passage.
        //  - Bootstrap (Sessions 1–4): NO limit on scheduled share.
        var missingDue: [String] = []
        if !plan.dueItems.isEmpty {
            for item in plan.dueItems {
                let count: Int
                if item.itemType == .chunk {
                    count = chunkPositionsById[item.id]?.count ?? 0
                } else {
                    let key = item.spanish.lowercased()
                    count = Self.matchedPositions(for: key, lemmaPositions: lemmaPositions, surfacePositions: surfacePositions).count
                }
                if count == 0 {
                    missingDue.append(item.spanish)
                }
            }
            if !missingDue.isEmpty {
                violations.append(ContractViolation(
                    ruleNumber: 3,
                    ruleName: "Due Items Coverage",
                    detail: "Every due item must appear at least once in the passage.",
                    offendingTokens: missingDue,
                    offendingSentenceIndices: sentences.isEmpty ? [] : [max(0, sentences.count - 1)]
                ))
            }
        }

        let scheduledShare = totalWords > 0 ? Double(scheduledTokenIndices.count) / Double(totalWords) : 0.0
        let hasAnyScheduledList = !plan.dueItems.isEmpty || !plan.newItems.isEmpty
        if !plan.isBootstrap && hasAnyScheduledList && scheduledShare > parameters.scheduledShareCap + 0.0005 {
            violations.append(ContractViolation(
                ruleNumber: 3,
                ruleName: "Scheduled Share Cap (<=23%)",
                detail: String(format: "Scheduled word share %.1f%% exceeds 23.0%% cap; add familiar Known prose without extra scheduled repetitions.", scheduledShare * 100),
                offendingTokens: [],
                offendingSentenceIndices: sentences.isEmpty ? [] : [max(0, sentences.count - 1)]
            ))
        }

        // Rule 4: Fresh Context (applies unchanged; passes automatically if dueItems is empty)
        var freshCollisions: [String] = []
        var freshCollisionSentenceIndices: Set<Int> = []
        for item in dueLexical {
            guard let priorNeighbors = priorContextByItem[item.id], !priorNeighbors.isEmpty else { continue }
            let positions = Self.matchedPositions(for: item.spanish.lowercased(), lemmaPositions: lemmaPositions, surfacePositions: surfacePositions)
            for pos in positions {
                let neighbors = contentNeighbors(around: pos, in: tokens, window: 3)
                let overlap = priorNeighbors.intersection(neighbors)
                if !overlap.isEmpty {
                    freshCollisions.append("\(item.spanish) repeated neighbor(s): \(overlap.sorted().joined(separator: ", "))")
                    if pos < tokenSentenceIndex.count {
                        freshCollisionSentenceIndices.insert(tokenSentenceIndex[pos])
                    }
                }
            }
        }
        if !freshCollisions.isEmpty {
            violations.append(ContractViolation(
                ruleNumber: 4,
                ruleName: "Fresh Context",
                detail: "Due items must appear in completely fresh lexical neighborhoods with no repeated adjacent content words from prior meetings.",
                offendingTokens: Array(freshCollisions.prefix(10)),
                offendingSentenceIndices: Array(freshCollisionSentenceIndices).sorted()
            ))
        }

        return VerificationReport(
            passed: violations.isEmpty,
            totalWords: totalWords,
            closedVocabCompliance: closedCompliance,
            scheduledWordShare: scheduledShare,
            missingDueItems: missingDue,
            underSpacedNewItems: underSpacedNew,
            freshContextCollisions: freshCollisions,
            holdoutLeaks: holdoutLeaks,
            offListLemmas: Array(Set(offListLemmas)).sorted(),
            stageBPassed: true,
            stageBFeedback: nil,
            violations: violations
        )
    }

    /// Returns the maximum number of sorted token positions whose pairwise spacing is at least `minSpacing`.
    /// Implements Revised Protocol Rule 2: "at least three times, with three of those uses at least 30 words apart.
    /// Further uses are free and never fail the rule."
    public static func maxSpacedOccurrences(positions: [Int], minSpacing: Int) -> Int {
        guard !positions.isEmpty else { return 0 }
        var count = 0
        var lastKept = Int.min / 2
        for pos in positions {
            if pos - lastKept >= minSpacing {
                count += 1
                lastKept = pos
            }
        }
        return count
    }

    private static func matchedPositions(
        for key: String,
        lemmaPositions: [String: [Int]],
        surfacePositions: [String: [Int]]
    ) -> [Int] {
        var set = Set(lemmaPositions[key] ?? []).union(surfacePositions[key] ?? [])
        // Account for Spanish article/determiner inflection quirks (e.g. "la" / "las" -> lemma "el"; "un" / "una" -> "uno")
        if key == "la" {
            set.formUnion(surfacePositions["las"] ?? [])
        } else if key == "el" {
            set.formUnion(surfacePositions["los"] ?? [])
            set.formUnion(surfacePositions["al"] ?? [])
            set.formUnion(surfacePositions["del"] ?? [])
        } else if key == "de" {
            set.formUnion(surfacePositions["del"] ?? [])
        } else if key == "a" {
            set.formUnion(surfacePositions["al"] ?? [])
        } else if key == "uno" || key == "un" {
            set.formUnion(surfacePositions["un"] ?? [])
            set.formUnion(surfacePositions["una"] ?? [])
            set.formUnion(surfacePositions["unos"] ?? [])
            set.formUnion(surfacePositions["unas"] ?? [])
        }
        return set.sorted()
    }

    private static func matchedSentences(
        for key: String,
        lemmaSentences: [String: Set<Int>],
        surfaceSentences: [String: Set<Int>]
    ) -> Set<Int> {
        var set = (lemmaSentences[key] ?? []).union(surfaceSentences[key] ?? [])
        if key == "la" {
            set.formUnion(surfaceSentences["las"] ?? [])
        } else if key == "el" {
            set.formUnion(surfaceSentences["los"] ?? [])
            set.formUnion(surfaceSentences["al"] ?? [])
            set.formUnion(surfaceSentences["del"] ?? [])
        } else if key == "de" {
            set.formUnion(surfaceSentences["del"] ?? [])
        } else if key == "a" {
            set.formUnion(surfaceSentences["al"] ?? [])
        } else if key == "uno" || key == "un" {
            set.formUnion(surfaceSentences["un"] ?? [])
            set.formUnion(surfaceSentences["una"] ?? [])
            set.formUnion(surfaceSentences["unos"] ?? [])
            set.formUnion(surfaceSentences["unas"] ?? [])
        }
        return set
    }

    /// Handles mandatory Spanish contractions & determiner inflections (`al`, `del`, `los`/`las`, `un`/`una`/`unos`/`unas`).
    private static func isArticleOrContractionAllowed(
        surface: String,
        lemma: String,
        allowedSet: Set<String>
    ) -> Bool {
        switch surface {
        case "al":
            return allowedSet.contains("a") && allowedSet.contains("el") || allowedSet.contains("al")
        case "del":
            return allowedSet.contains("de") && allowedSet.contains("el") || allowedSet.contains("del")
        case "los":
            return allowedSet.contains("el")
        case "las":
            return allowedSet.contains("la") || allowedSet.contains("el")
        case "un", "una", "unos", "unas":
            return allowedSet.contains("un") || allowedSet.contains("uno")
        default:
            if lemma == "uno" && allowedSet.contains("un") { return true }
            if lemma == "un" && allowedSet.contains("uno") { return true }
            return false
        }
    }

    /// Extracts ±window adjacent content-word lemmas around a target word position for Rule 4 tracking.
    public func contentNeighbors(around index: Int, in tokens: [AnalyzedToken], window: Int = 3) -> Set<String> {
        let low = max(0, index - window)
        let high = min(tokens.count - 1, index + window)
        var set: Set<String> = []
        for i in low...high where i != index {
            if tokens[i].isContentWord {
                set.insert(tokens[i].lemma.lowercased())
            }
        }
        return set
    }

    /// Computes `(itemId, sentenceText, neighborContentLemmas)` records to store in SQLite after a passage completes.
    public func extractSentenceContexts(
        passage: GeneratedPassage,
        trackedItems: [SyllabusItem]
    ) -> [(itemId: String, sentenceText: String, neighbors: [String])] {
        let lemmaToId = Dictionary(
            trackedItems.filter { $0.itemType == .lemma }.map { ($0.spanish.lowercased(), $0.id) },
            uniquingKeysWith: { first, _ in first }
        )
        var records: [(String, String, [String])] = []
        for sentence in passage.sentences {
            let tokens = lemmatizer.analyze(sentence.spanish)
            for tok in tokens {
                if let itemId = lemmaToId[tok.lemma.lowercased()] {
                    let neighbors = Array(contentNeighbors(around: tok.wordIndex, in: tokens, window: 3)).sorted()
                    records.append((itemId, sentence.spanish, neighbors))
                }
            }
        }
        return records
    }
}
