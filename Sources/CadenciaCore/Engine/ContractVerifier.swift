import Foundation

/// Implements the automatic Stage A (deterministic local) Input Contract Verifier
/// from the 17-page Signal-Starvation Hypothesis Test Protocol (pp. 6–9):
///
/// **Full Contract (Sessions 5+):**
/// - Passage form: 180–250 words.
/// - Rule 1 (Closed vocabulary): 100% of lemmas belong to the 5 allowed groups (Known, Learning p > 85%,
///   due list, new list, names/numbers). ZERO tolerance for hold-out lemmas.
/// - Rule 2 (New items): at most 6 per passage; each appears >= 3 times with 3 uses >= 30 words apart;
///   further uses are free and never fail the rule.
/// - Rule 3 (Due items & Scheduled share): every due item appears >= 1 time; combined new + due <= 23%;
///   no lower limit per passage.
/// - Rule 4 (Fresh context): no repeated ±3 content neighbors for due items.
/// - Empty-list rule: a rule about an empty list passes automatically.
///
/// **Bootstrap Contract (Sessions 1–4, Fixed Kits of 6–8, pp. 8–9):**
/// - Feasibility pre-check: 23% arithmetic is skipped; only pre-check is that a kit holds a verb and a noun.
/// - Passage form: 6–10 sentences of 8 words or fewer, each with a picture and an English line on tap.
/// - Rule 1 (Closed vocabulary): >= 80% of words are items met so far, items in the passage's kit,
///   fixed cast names (Ana, Luis, Carla, Perú, México — NOT a tagger!), or numbers. Remaining <= 20%
///   must be glossed transparent cognates. Words are matched against a table of each item's inflected forms,
///   and `al` and `del` count as TWO items (`a`+`el`, `de`+`el`). Patterns are not part of Rule 1.
/// - Rule 2 (New items): every kit item appears >= 1x in its passage (and in >= 3 different sentences
///   during the session across kit + recombination passages). No spacing rule and no upper limit on uses.
/// - Rule 3 (Scheduled share): NO limit. Due items (if any) appear >= 1x.
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
        _ = priorSessionSentenceCountsByItemId
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

        let totalSurfaceWords = tokens.count
        var violations: [ContractViolation] = []

        // 0. Passage form & Bootstrap kit feasibility check
        if plan.isBootstrap {
            if !plan.newItems.isEmpty && !BootstrapKitLibrary.checkKitFeasibility(items: plan.newItems) {
                violations.append(ContractViolation(
                    ruleNumber: 0,
                    ruleName: "Bootstrap Kit Feasibility",
                    detail: "Bootstrap kit must hold at least one verb and one noun.",
                    offendingTokens: plan.newItems.map(\.spanish)
                ))
            }
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
            if totalSurfaceWords < parameters.passageMinWords || totalSurfaceWords > parameters.passageMaxWords {
                violations.append(ContractViolation(
                    ruleNumber: 0,
                    ruleName: "Passage Length (\(parameters.passageMinWords)–\(parameters.passageMaxWords) words)",
                    detail: "Passage has \(totalSurfaceWords) words; required \(parameters.passageMinWords)–\(parameters.passageMaxWords).",
                    offendingTokens: [],
                    offendingSentenceIndices: sentences.isEmpty ? [] : [max(0, sentences.count - 1)]
                ))
            }
        }

        // Build lookup sets
        var allowedSet = Set(plan.allowedKnownLemmas.map { $0.lowercased() })
        for item in plan.dueItems { allowedSet.insert(item.spanish.lowercased()) }
        for item in plan.newItems { allowedSet.insert(item.spanish.lowercased()) }
        if allowedSet.contains("un") { allowedSet.insert("uno") }
        if allowedSet.contains("uno") { allowedSet.insert("un") }

        let holdoutSet = Set(plan.holdoutBannedLemmas.map { $0.lowercased() })
        let cognateSet = Set(plan.transparentCognateLemmas.map { $0.lowercased() })
            .union(glossedTransparentCognates.map { $0.lowercased() })

        // Bootstrap cast list comes from the fixed cast list (NOT a tagger!)
        var fixedCastSet = BootstrapKitLibrary.fixedCastLowercaseSet
        for name in plan.bootstrapCastNames {
            let low = name.lowercased()
            fixedCastSet.insert(low)
            fixedCastSet.insert(SpanishLemmatizer.stripSpanishAccents(low))
        }

        let dueLexical = plan.dueItems.filter { $0.itemType == .lemma }
        let dueChunks = plan.dueItems.filter { $0.itemType == .chunk }
        let newLexical = plan.newItems.filter { $0.itemType == .lemma }
        let newChunks = plan.newItems.filter { $0.itemType == .chunk }

        let dueKeys = Set(dueLexical.map { $0.spanish.lowercased() })
        let newKeys = Set(newLexical.map { $0.spanish.lowercased() })
        let scheduledKeys = dueKeys.union(newKeys)

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

        // Track occurrences using the table of inflected forms + two-item expansion for `al` and `del`!
        var holdoutLeaks: [String] = []
        var holdoutSentenceIndices: Set<Int> = []
        var offListLemmas: [String] = []
        var offListSentenceIndices: Set<Int> = []
        var nonCognateBootstrapOffList: [String] = []
        var nonCognateSentenceIndices: Set<Int> = []

        var totalCheckedUnits = 0
        var coreCompliantUnits = 0
        var scheduledUnitCount = 0
        var itemPositions: [String: [Int]] = [:]
        var itemSentences: [String: Set<Int>] = [:]

        for tok in tokens {
            let norm = tok.normalized
            let lem = tok.lemma.lowercased()
            let sIdx = tok.wordIndex < tokenSentenceIndex.count ? tokenSentenceIndex[tok.wordIndex] : 0
            let isNumber = tok.surface.allSatisfy(\.isNumber)

            // Fixed cast list in Bootstrap (NOT tagger!); standard proper name/number detection from Session 5+
            let isCastOrNumber: Bool = plan.isBootstrap
                ? (isNumber || fixedCastSet.contains(norm) || fixedCastSet.contains(SpanishLemmatizer.stripSpanishAccents(norm)))
                : tok.isProperNameOrNumber

            if isCastOrNumber {
                totalCheckedUnits += 1
                coreCompliantUnits += 1
                continue
            }

            // Expand `al` -> [a, el], `del` -> [de, el], and match every word against the inflected-forms table!
            let atomicSets = BootstrapKitLibrary.expandTokenToCandidateItemSets(
                normalizedSurface: norm,
                lemmatizerLemma: lem
            )

            for candSet in atomicSets {
                totalCheckedUnits += 1

                // Record positions & sentences for EVERY canonical item this inflected form belongs to
                for key in candSet {
                    itemPositions[key, default: []].append(tok.wordIndex)
                    itemSentences[key, default: []].insert(sIdx)
                }

                // Hold-out zero-tolerance guard
                let allowedMatch = !candSet.isDisjoint(with: allowedSet)
                if !allowedMatch, holdoutSet.contains(lem) || holdoutSet.contains(norm) {
                    holdoutLeaks.append(tok.surface)
                    holdoutSentenceIndices.insert(sIdx)
                }

                if !candSet.isDisjoint(with: scheduledKeys) {
                    scheduledUnitCount += 1
                }

                let isCoreAllowed = allowedMatch || chunkCoveredWordIndices.contains(tok.wordIndex)
                if isCoreAllowed {
                    coreCompliantUnits += 1
                } else {
                    offListLemmas.append("\(tok.surface) (\(lem))")
                    offListSentenceIndices.insert(sIdx)
                    if plan.isBootstrap {
                        let isGlossedCognate = !holdoutSet.contains(lem)
                            && !holdoutSet.contains(norm)
                            && !candSet.isDisjoint(with: cognateSet)
                        if !isGlossedCognate {
                            nonCognateBootstrapOffList.append("\(tok.surface) (\(lem))")
                            nonCognateSentenceIndices.insert(sIdx)
                        }
                    }
                }
            }
        }

        // Add chunk tokens belonging to due/new chunks to scheduled share (without double-counting)
        let scheduledChunkIds = Set((dueChunks + newChunks).map(\.id))
        if !scheduledChunkIds.isEmpty {
            var extraChunkTokens = 0
            for m in chunkMatches where scheduledChunkIds.contains(m.itemId) {
                extraChunkTokens += m.length
            }
            scheduledUnitCount = min(totalCheckedUnits, scheduledUnitCount + extraChunkTokens)
        }

        // Hold-out zero-tolerance check
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
        let closedCompliance = totalCheckedUnits > 0 ? Double(coreCompliantUnits) / Double(totalCheckedUnits) : 0.0
        if plan.isBootstrap {
            if closedCompliance < 0.80 {
                violations.append(ContractViolation(
                    ruleNumber: 1,
                    ruleName: "Bootstrap Closed Vocabulary (>=80%)",
                    detail: String(format: "Met/kit/cast word share %.1f%% < required 80%%. Build sentences from the kit items, met items, and cast names (%@).", closedCompliance * 100, BootstrapKitLibrary.fixedCastDisplayNames.joined(separator: ", ")),
                    offendingTokens: Array(Set(offListLemmas)).sorted().prefix(12).map { $0 },
                    offendingSentenceIndices: Array(offListSentenceIndices).sorted()
                ))
            }
            if !nonCognateBootstrapOffList.isEmpty {
                violations.append(ContractViolation(
                    ruleNumber: 1,
                    ruleName: "Bootstrap Cognate Remainder",
                    detail: "Words outside met/kit/cast in Bootstrap must be glossed transparent cognates. Replace non-cognate off-list words.",
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
                // Bootstrap Rule 2 (Revised Protocol p. 8):
                // "Every kit item appears in its passage, and in at least three different sentences during the session.
                //  No spacing rule and no upper limit on uses."
                for item in plan.newItems {
                    let countInPassage: Int
                    if item.itemType == .chunk {
                        countInPassage = (chunkSentencesById[item.id] ?? []).count
                    } else {
                        countInPassage = (itemSentences[item.spanish.lowercased()] ?? []).count
                    }
                    if countInPassage < 1 {
                        underSpacedNew.append("\(item.spanish) (missing from kit passage)")
                    }
                }
                if !underSpacedNew.isEmpty {
                    violations.append(ContractViolation(
                        ruleNumber: 2,
                        ruleName: "Bootstrap Kit Coverage",
                        detail: "Every kit item must appear at least once in its kit passage (no spacing limit).",
                        offendingTokens: underSpacedNew,
                        offendingSentenceIndices: sentences.isEmpty ? [] : [max(0, sentences.count - 1)]
                    ))
                }
            } else {
                // Full Contract Rule 2 (p. 6): each appears >= 3 times with 3 uses >= 30 words apart; extra uses are free!
                let minSpacing = parameters.newItemMinWordSpacing
                for item in plan.newItems {
                    let positions: [Int]
                    if item.itemType == .chunk {
                        positions = (chunkPositionsById[item.id] ?? []).sorted()
                    } else {
                        positions = Array(Set(itemPositions[item.spanish.lowercased()] ?? [])).sorted()
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
        var missingDue: [String] = []
        if !plan.dueItems.isEmpty {
            for item in plan.dueItems {
                let count: Int
                if item.itemType == .chunk {
                    count = chunkPositionsById[item.id]?.count ?? 0
                } else {
                    count = (itemPositions[item.spanish.lowercased()] ?? []).count
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

        let scheduledShare = totalCheckedUnits > 0 ? Double(scheduledUnitCount) / Double(totalCheckedUnits) : 0.0
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
            let positions = Array(Set(itemPositions[item.spanish.lowercased()] ?? [])).sorted()
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
            totalWords: totalSurfaceWords,
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

    /// Finds every new item ID from `candidateItems` whose inflected forms appear inside `sentenceSpanish`
    /// (Bootstrap Rule 5: whole-sentence picture-matching probe logs against each new item in that sentence!).
    public func matchedItemIds(in sentenceSpanish: String, from candidateItems: [SyllabusItem]) -> [String] {
        let toks = lemmatizer.analyze(sentenceSpanish)
        var presentKeys: Set<String> = []
        for tok in toks {
            for set in BootstrapKitLibrary.expandTokenToCandidateItemSets(normalizedSurface: tok.normalized, lemmatizerLemma: tok.lemma) {
                presentKeys.formUnion(set)
            }
        }
        return candidateItems
            .filter { presentKeys.contains($0.spanish.lowercased()) }
            .map(\.id)
    }

    /// Returns the maximum number of sorted token positions whose pairwise spacing is at least `minSpacing`.
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
                let sets = BootstrapKitLibrary.expandTokenToCandidateItemSets(normalizedSurface: tok.normalized, lemmatizerLemma: tok.lemma)
                for candSet in sets {
                    for key in candSet {
                        if let itemId = lemmaToId[key] {
                            let neighbors = Array(contentNeighbors(around: tok.wordIndex, in: tokens, window: 3)).sorted()
                            records.append((itemId, sentence.spanish, neighbors))
                            break
                        }
                    }
                }
            }
        }
        return records
    }
}
