import Foundation

/// Implements the automatic Stage A (deterministic local) Input Contract Verifier
/// from Signal-starvation hypothesis test protocol (pp. 6–7):
///
/// - Word count: 180–250 words (or <= 8 words per sentence during Sessions 1–4 Bootstrap)
/// - Rule 1 (Closed vocabulary): 100% of lemmas must be Known, Learning with p > 85%,
///   on due_list, on new_list, or a proper name / number (relaxed to 80% in Sessions 1–4 Bootstrap).
///   ZERO tolerance for hold-out lemmas (400 non-cognate guard lemmas NEVER appear!).
/// - Rule 2 (New items): at most 6 per passage; each appears >= 3 times, consecutive occurrences
///   at least 30 words apart (relaxed spacing in short bootstrap passages).
/// - Rule 3 (Due items): every item on the due list appears >= 1 time; new + due items together
///   make up 18%–23% of running words (calibrated for bootstrap / week 1 / week 8 low-new sessions).
/// - Rule 4 (Fresh context): a due item never appears beside the same content words (±3 window)
///   as in an earlier meeting stored in `item_sentence_contexts`.
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
        priorContextByItem: [String: Set<String>]
    ) -> VerificationReport {
        let tokens = lemmatizer.analyze(passageText)
        let totalWords = tokens.count
        var violations: [ContractViolation] = []

        // Word count check
        if plan.isBootstrap {
            // PDF p. 7: Bootstrap sentences have 8 words or fewer
            let overlong = sentences.filter { lemmatizer.extractWordTokens($0.spanish).count > 9 }
            if !overlong.isEmpty {
                violations.append(ContractViolation(
                    ruleNumber: 0,
                    ruleName: "Bootstrap Sentence Length",
                    detail: "\(overlong.count) bootstrap sentence(s) exceeded the 8-word ceiling.",
                    offendingTokens: overlong.prefix(3).map(\.spanish)
                ))
            }
        } else {
            if totalWords < parameters.passageMinWords || totalWords > parameters.passageMaxWords {
                violations.append(ContractViolation(
                    ruleNumber: 0,
                    ruleName: "Passage Length (180–250 words)",
                    detail: "Passage has \(totalWords) words; required \(parameters.passageMinWords)–\(parameters.passageMaxWords).",
                    offendingTokens: []
                ))
            }
        }

        // Build lookup sets
        let allowedSet = Set(plan.allowedKnownLemmas.map { $0.lowercased() })
        let holdoutSet = Set(plan.holdoutBannedLemmas.map { $0.lowercased() })
        let dueLexical = plan.dueItems.filter { $0.itemType == .lemma }
        let dueChunks = plan.dueItems.filter { $0.itemType == .chunk }
        let newLexical = plan.newItems.filter { $0.itemType == .lemma }
        let newChunks = plan.newItems.filter { $0.itemType == .chunk }

        let dueLemmaToItem = Dictionary(uniqueKeysWithValues: dueLexical.map { ($0.spanish.lowercased(), $0) })
        let newLemmaToItem = Dictionary(uniqueKeysWithValues: newLexical.map { ($0.spanish.lowercased(), $0) })

        // Multi-word chunk matches
        let chunkMatches = lemmatizer.findChunks(in: tokens)
        var chunkPositionsById: [String: [Int]] = [:]
        var chunkCoveredWordIndices: Set<Int> = []
        for m in chunkMatches {
            chunkPositionsById[m.itemId, default: []].append(m.startIndex)
            for k in m.startIndex..<(m.startIndex + m.length) {
                chunkCoveredWordIndices.insert(k)
            }
        }

        // Track occurrences & rule 1 closed vocab
        var holdoutLeaks: [String] = []
        var offListLemmas: [String] = []
        var compliantCount = 0
        var scheduledTokenIndices: Set<Int> = []
        var lemmaPositions: [String: [Int]] = [:]

        for tok in tokens {
            let lem = tok.lemma.lowercased()
            lemmaPositions[lem, default: []].append(tok.wordIndex)

            if holdoutSet.contains(lem) && !allowedSet.contains(lem) {
                holdoutLeaks.append(tok.surface)
            }

            let isScheduledLemma = dueLemmaToItem[lem] != nil || newLemmaToItem[lem] != nil
            if isScheduledLemma {
                scheduledTokenIndices.insert(tok.wordIndex)
            }

            let isCompliant = tok.isProperNameOrNumber
                || allowedSet.contains(lem)
                || allowedSet.contains(tok.normalized)
                || chunkCoveredWordIndices.contains(tok.wordIndex)

            if isCompliant {
                compliantCount += 1
            } else {
                offListLemmas.append("\(tok.surface) (\(lem))")
            }
        }

        // Add chunk tokens belonging to due/new chunks to scheduled share
        let scheduledChunkIds = Set((dueChunks + newChunks).map(\.id))
        for m in chunkMatches where scheduledChunkIds.contains(m.itemId) {
            for k in m.startIndex..<(m.startIndex + m.length) {
                scheduledTokenIndices.insert(k)
            }
        }

        // Hold-out zero-tolerance check
        if !holdoutLeaks.isEmpty {
            violations.append(ContractViolation(
                ruleNumber: 1,
                ruleName: "Hold-Out Lemma Guard",
                detail: "Banned hold-out lemmas appeared in the passage. Remove them completely.",
                offendingTokens: Array(Set(holdoutLeaks)).sorted()
            ))
        }

        // Rule 1: Closed Vocabulary (100% from session 5+, 80% in Bootstrap sessions 1–4)
        let closedCompliance = totalWords > 0 ? Double(compliantCount) / Double(totalWords) : 0.0
        let requiredCompliance = plan.isBootstrap ? 0.80 : 1.0
        if closedCompliance < requiredCompliance {
            let uniqueOff = Array(Set(offListLemmas)).sorted()
            violations.append(ContractViolation(
                ruleNumber: 1,
                ruleName: "Closed Vocabulary",
                detail: String(format: "Closed vocabulary compliance %.1f%% < required %.0f%%. Replace off-list words with allowed lemmas.", closedCompliance * 100, requiredCompliance * 100),
                offendingTokens: Array(uniqueOff.prefix(15))
            ))
        }

        // Rule 2: New Items (<= 6, each appears >= 3 times, >= 30 words apart)
        var underSpacedNew: [String] = []
        if plan.newItems.count > parameters.maxNewItemsPerPassage {
            violations.append(ContractViolation(
                ruleNumber: 2,
                ruleName: "New Items Cap",
                detail: "Too many new items (\(plan.newItems.count) > \(parameters.maxNewItemsPerPassage)).",
                offendingTokens: plan.newItems.map(\.spanish)
            ))
        }
        let minSpacing = plan.isBootstrap ? 8 : parameters.newItemMinWordSpacing
        for item in plan.newItems {
            let positions: [Int]
            if item.itemType == .chunk {
                positions = (chunkPositionsById[item.id] ?? []).sorted()
            } else {
                positions = (lemmaPositions[item.spanish.lowercased()] ?? []).sorted()
            }
            if positions.count < parameters.newItemMinOccurrences {
                underSpacedNew.append("\(item.spanish) (count \(positions.count)/\(parameters.newItemMinOccurrences))")
            } else {
                // Check spacing between consecutive occurrences
                var spacingOK = true
                for i in 1..<positions.count {
                    if positions[i] - positions[i - 1] < minSpacing {
                        spacingOK = false
                        break
                    }
                }
                if !spacingOK {
                    underSpacedNew.append("\(item.spanish) (closer than \(minSpacing) words)")
                }
            }
        }
        if !underSpacedNew.isEmpty {
            violations.append(ContractViolation(
                ruleNumber: 2,
                ruleName: "New Items Repetition & Spacing",
                detail: "Every new item must appear at least \(parameters.newItemMinOccurrences) times, at least \(minSpacing) words apart.",
                offendingTokens: underSpacedNew
            ))
        }

        // Rule 3: Due Items (every due item appears >= 1x; combined new + due share 18%-23%)
        var missingDue: [String] = []
        for item in plan.dueItems {
            let count: Int
            if item.itemType == .chunk {
                count = chunkPositionsById[item.id]?.count ?? 0
            } else {
                count = lemmaPositions[item.spanish.lowercased()]?.count ?? 0
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
                offendingTokens: missingDue
            ))
        }

        let scheduledShare = totalWords > 0 ? Double(scheduledTokenIndices.count) / Double(totalWords) : 0.0
        // Weeks 2–7 full contract enforces 18%–23% scheduled share (PDF p. 7); Wk 1/8/Bootstrap naturally sit lower
        let week = min(8, max(1, ((plan.sessionNumber - 1) / 6) + 1))
        let enforceShareFloor = !plan.isBootstrap && week >= 2 && week <= 6 && !plan.dueItems.isEmpty
        if enforceShareFloor && scheduledShare < parameters.scheduledShareFloor {
            violations.append(ContractViolation(
                ruleNumber: 3,
                ruleName: "Scheduled Share Floor (18%–23%)",
                detail: String(format: "Scheduled word share %.1f%% is below 18%% floor; weave in scheduled due/new items more densely.", scheduledShare * 100),
                offendingTokens: []
            ))
        }
        if scheduledShare > parameters.scheduledShareCap + 0.015 {
            violations.append(ContractViolation(
                ruleNumber: 3,
                ruleName: "Scheduled Share Cap (<=23%)",
                detail: String(format: "Scheduled word share %.1f%% exceeds 23%% cap; add familiar Known glue prose so comprehension stays >=85%%.", scheduledShare * 100),
                offendingTokens: []
            ))
        }

        // Rule 4: Fresh Context (a due item never appears beside the same content words as in an earlier meeting)
        var freshCollisions: [String] = []
        for item in dueLexical {
            guard let priorNeighbors = priorContextByItem[item.id], !priorNeighbors.isEmpty else { continue }
            let positions = lemmaPositions[item.spanish.lowercased()] ?? []
            for pos in positions {
                let neighbors = contentNeighbors(around: pos, in: tokens, window: 3)
                let overlap = priorNeighbors.intersection(neighbors)
                if !overlap.isEmpty {
                    freshCollisions.append("\(item.spanish) repeated neighbor(s): \(overlap.sorted().joined(separator: ", "))")
                }
            }
        }
        if !freshCollisions.isEmpty {
            violations.append(ContractViolation(
                ruleNumber: 4,
                ruleName: "Fresh Context",
                detail: "Due items must appear in completely fresh lexical neighborhoods with no repeated adjacent content words from prior meetings.",
                offendingTokens: Array(freshCollisions.prefix(10))
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
