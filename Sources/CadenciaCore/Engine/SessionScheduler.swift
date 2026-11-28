import Foundation

/// Implements the complete Signal-Starvation Hypothesis Scheduler across all 4 study modes:
/// - Group A (Full System): forgetting-timed due lists targeting p in [0.70, 0.90] (joins at <= 0.85),
///   1/2/4/8/16 session spacing awareness, 23% scheduled share cap, 8-week pace/intake/audio-first table,
///   Sessions 1–4 Bootstrap (60 commonest function words + transparent cognates per session).
/// - Group B (Scheduler Off): identical topics, known share & output, but natural frequency order without forgetting timing.
/// - Stage 1 (Capacity Test): 12 sessions balanced randomly to 30, 60, or 120 new items/session (4 each).
/// - Solo Pilot: Weeks 1–2 alternate 30/60/120 new items/session + random 10% never-scheduled hold-back items.
public struct SessionScheduler: Sendable {
    public let modelEngine: LearnerModelEngine

    public init(parameters: ProtocolParameters = .protocolDefaults) {
        self.modelEngine = LearnerModelEngine(parameters: parameters)
    }

    /// Computes today's new-item budget based on StudyMode and Session Number (1...48).
    public func targetNewItemsForSession(
        sessionNumber: Int,
        studyMode: StudyMode,
        learnerSeed: Int = 42
    ) -> Int {
        // Sessions 1–4 are Bootstrap in Group A / Group B / Solo Pilot (60 items/session, PDF p. 7)
        if sessionNumber <= 4 && studyMode != .stage1CapacityTest {
            return 60
        }

        switch studyMode {
        case .stage1CapacityTest:
            // 12 sessions balanced: four each of 30, 60, 120 in deterministic shuffled order per learner
            let schedule = deterministicBalancedCapacitySchedule(seed: learnerSeed)
            let idx = (sessionNumber - 1) % schedule.count
            return schedule[idx]

        case .soloPilot:
            // First 2 weeks (sessions 5–12 after bootstrap, or alternating 30/60/120 in weeks 1–2)
            if sessionNumber >= 5 && sessionNumber <= 12 {
                let pattern = [30, 60, 120]
                return pattern[(sessionNumber - 5) % 3]
            }
            let week = min(8, max(1, ((sessionNumber - 1) / 6) + 1))
            return min(modelEngine.parameters.maxNewItemsPerSession, WeeklyPlanSpec.forWeek(week).newItemsPerSession)

        case .groupAFullSystem, .groupBSchedulerOff:
            let week = min(8, max(1, ((sessionNumber - 1) / 6) + 1))
            return min(modelEngine.parameters.maxNewItemsPerSession, WeeklyPlanSpec.forWeek(week).newItemsPerSession)
        }
    }

    /// Returns whether today's Output 2 block (minutes 62–75) is a 13-minute Writing Task
    /// (twice a week from Week 3 onward: sessions 3 and 6 of each 6-session week, PDF p. 8–9).
    public func isWritingSession(sessionNumber: Int) -> Bool {
        let week = ((sessionNumber - 1) / 6) + 1
        guard week >= 3 else { return false }
        let dayInWeek = ((sessionNumber - 1) % 6) + 1
        return dayInWeek == 3 || dayInWeek == 6
    }

    /// Selects the 20 shakiest items from the previous session for the 0–3 min Warm-Up block (PDF p. 8).
    public func selectWarmupItems(
        allItems: [SyllabusItem],
        records: [String: LearnerItemRecord],
        currentSession: Int,
        now: Date = Date()
    ) -> [SyllabusItem] {
        let itemMap = Dictionary(uniqueKeysWithValues: allItems.map { ($0.id, $0) })
        let candidates: [(SyllabusItem, Double)] = records.values.compactMap { rec in
            guard rec.meetingsCount > 0, !rec.isHeldBackControl, let item = itemMap[rec.itemId] else { return nil }
            guard item.itemType == .lemma || item.itemType == .chunk else { return nil }
            let p = rec.predictedRecall(at: now)
            // Prioritize items met in the last session or with lowest recall / recent lookups
            let recencyBonus = (rec.lastMeetingSession == currentSession - 1) ? -0.25 : 0.0
            let failureBonus = rec.failuresCount > 0 ? -0.15 : 0.0
            return (item, p + recencyBonus + failureBonus)
        }
        return candidates
            .sorted { $0.1 < $1.1 }
            .prefix(20)
            .map(\.0)
    }

    /// Plans the exact next passage (180–250 words, or bootstrap micro-story) before generation.
    public func planNextPassage(
        profile: LearnerProfile,
        blockType: SessionBlockType,
        passageIndexInSession: Int,
        newItemsRemainingInSession: Int,
        allItems: [SyllabusItem],
        holdoutItems: [SyllabusItem],
        records: [String: LearnerItemRecord],
        priorContextByItem: [String: Set<String>],
        serialGenre: SerialGenre,
        serialEpisodeNumber: Int,
        storySummarySoFar: String,
        dueReductionFactor: Double = 1.0,
        now: Date = Date()
    ) -> PassagePlan {
        let week = profile.currentWeek
        let weekSpec = WeeklyPlanSpec.forWeek(week)
        let isBootstrap = profile.isBootstrapSession && profile.studyMode != .stage1CapacityTest

        // Audio-first determination: Input 2 follows weekly audio-first schedule; Input 1 always shows text + audio
        let isAudioFirst: Bool
        if isBootstrap || blockType == .input1 {
            isAudioFirst = false
        } else {
            // Match weekly audioFirstFraction (20% in Wk2 -> 80% in Wk7–8)
            let fraction = weekSpec.audioFirstFraction
            isAudioFirst = fraction > 0 && Double((passageIndexInSession * 37) % 100) / 100.0 < fraction
        }

        // Voice & Regional Accent Rotation (Rule 8: rotate through >=4 native Phonon voices and >=2 regional accents)
        let accents: [SpanishRegionalAccent] = [.castilian, .latinAmerican, .rioplatense]
        let accent = accents[(profile.currentSessionNumber + passageIndexInSession) % accents.count]
        let regionalPool = accent == .castilian ? TutorVoice.castilianVoices : TutorVoice.latinAmericanVoices
        let voice = regionalPool[(profile.currentSessionNumber * 7 + passageIndexInSession) % regionalPool.count]

        // Continuity alternation (Rule 7: episodes of chosen serial alternating with learner-named non-fiction)
        let format: PassageFormat
        let topicTitle: String
        if isBootstrap {
            format = .bootstrapMicroStory
            topicTitle = "Bootstrap Scene #\(passageIndexInSession + 1) · \(serialGenre.titleEs)"
        } else if passageIndexInSession % 2 == 0 {
            format = .serialEpisode
            topicTitle = "\(serialGenre.titleEs) — Episode \(serialEpisodeNumber + 1)"
        } else {
            format = .nonfictionBrief
            let topics = profile.nonfictionTopics.isEmpty
                ? ["Architecture & Cities", "Science & Nature", "Food & History", "Technology & Music"]
                : profile.nonfictionTopics
            topicTitle = topics[(profile.currentSessionNumber + passageIndexInSession / 2) % topics.count]
        }

        // Partition syllabus
        let activeLexical = allItems.filter { !$0.isHoldout && ($0.itemType == .lemma || $0.itemType == .chunk) }
        let activeGrammar = allItems.filter { !$0.isHoldout && $0.itemType == .grammar }

        // 1. Select New Items (at most 8 per passage in Bootstrap Sessions 1–4, at most 6 from Session 5+)
        let maxNewThisPassage = isBootstrap
            ? modelEngine.parameters.maxNewItemsPerBootstrapPassage
            : modelEngine.parameters.maxNewItemsPerPassage

        let targetNewCount: Int
        if newItemsRemainingInSession <= 0 {
            targetNewCount = 0
        } else if isBootstrap {
            targetNewCount = min(maxNewThisPassage, newItemsRemainingInSession)
        } else {
            let estimatedPassagesLeft = max(1, estimatedPassagesForWeek(week) - passageIndexInSession)
            let paced = Int(ceil(Double(newItemsRemainingInSession) / Double(estimatedPassagesLeft)))
            targetNewCount = min(maxNewThisPassage, max(1, min(paced, newItemsRemainingInSession)))
        }

        let selectedNewItemsRaw: [SyllabusItem]
        if isBootstrap {
            // Revised Bootstrap contract (p. 8): nothing is pre-loaded as Known; function words and
            // transparent cognates enter via the new list (up to 8 per passage).
            // Interleave unmet bootstrap function/core verbs (~5, ensuring a core verb like ser/estar
            // joins early) with unmet bootstrap transparent cognates (~3) so even Passage 1 has
            // nouns + verbs + function words on its new list and easily satisfies >=80% Rule 1!
            let bootstrapUnmet = activeLexical.filter { item in
                let rec = records[item.id]
                return item.isBootstrap && (rec?.meetingsCount ?? 0) == 0 && !(rec?.isHeldBackControl ?? false)
            }.sorted { $0.rankIndex < $1.rankIndex }

            let fallbackUnmet = activeLexical.filter {
                (records[$0.id]?.meetingsCount ?? 0) == 0 && !(records[$0.id]?.isHeldBackControl ?? false)
            }.sorted { $0.rankIndex < $1.rankIndex }

            let pool = bootstrapUnmet.isEmpty ? fallbackUnmet : bootstrapUnmet
            let corePool = pool.filter { !$0.isTransparentCognate }
            let cogPool = pool.filter { $0.isTransparentCognate }.sorted { lhs, rhs in
                let lhsContent = (lhs.pos == "NOUN" || lhs.pos == "ADJ")
                let rhsContent = (rhs.pos == "NOUN" || rhs.pos == "ADJ")
                if lhsContent != rhsContent { return lhsContent }
                return lhs.rankIndex < rhs.rankIndex
            }
            let verbPool = corePool.filter { $0.pos == "VERB" || $0.pos == "AUX" }

            var chosen: [SyllabusItem] = []
            var seenIds: Set<String> = []
            let wantCogs = min(cogPool.count, max(1, targetNewCount / 3))
            let wantCore = max(0, targetNewCount - wantCogs)

            // Guarantee at least 1 core verb (e.g. ser / estar) in the very first passage so natural sentences flow
            let hasAnyMetVerb = activeLexical.contains {
                ($0.pos == "VERB" || $0.pos == "AUX") && (records[$0.id]?.meetingsCount ?? 0) > 0
            }
            if !hasAnyMetVerb, let firstVerb = verbPool.first, chosen.count < targetNewCount {
                chosen.append(firstVerb)
                seenIds.insert(firstVerb.id)
            }
            for item in corePool where chosen.count < wantCore && !seenIds.contains(item.id) {
                chosen.append(item)
                seenIds.insert(item.id)
            }
            for item in cogPool where chosen.count < targetNewCount && !seenIds.contains(item.id) {
                chosen.append(item)
                seenIds.insert(item.id)
            }
            for item in pool where chosen.count < targetNewCount && !seenIds.contains(item.id) {
                chosen.append(item)
                seenIds.insert(item.id)
            }
            selectedNewItemsRaw = chosen.sorted { $0.rankIndex < $1.rankIndex }
        } else {
            // Mix lemmas (~94%) and chunks (~6%) in frequency order
            let newPool = activeLexical.filter { item in
                let rec = records[item.id]
                return (rec?.meetingsCount ?? 0) == 0 && !(rec?.isHeldBackControl ?? false)
            }.sorted { lhs, rhs in
                let lhsScore = lhs.itemType == .chunk ? lhs.rankIndex * 18 : lhs.rankIndex
                let rhsScore = rhs.itemType == .chunk ? rhs.rankIndex * 18 : rhs.rankIndex
                return lhsScore < rhsScore
            }
            selectedNewItemsRaw = Array(newPool.prefix(targetNewCount))
        }

        // 2. Select Due Items (Rule 3: from Session 5+, new + due <= 23% of target length, no per-passage floor)
        let targetPassageWords = isBootstrap ? 60 : 215
        let maxScheduledWordSlots = isBootstrap
            ? 24 // Bootstrap has no scheduled share limit (p. 8)
            : Int((Double(targetPassageWords) * modelEngine.parameters.scheduledShareCap).rounded(.down))
        let newWordSlots = selectedNewItemsRaw.count * modelEngine.parameters.newItemMinOccurrences
        let rawDueBudget = isBootstrap
            ? 12
            : max(0, min(32, maxScheduledWordSlots - newWordSlots))
        let adjustedDueBudget = max(0, Int((Double(rawDueBudget) * dueReductionFactor).rounded()))

        let selectedDueItemsRaw: [SyllabusItem]
        if adjustedDueBudget == 0 {
            selectedDueItemsRaw = []
        } else if profile.studyMode == .groupBSchedulerOff {
            // Group B (Control): NO forgetting timing — sample met items in natural frequency order
            let metLexical = activeLexical.filter { item in
                guard let rec = records[item.id] else { return false }
                return rec.meetingsCount > 0 && !rec.isHeldBackControl
            }.sorted { $0.rankIndex < $1.rankIndex }
            let strideStart = (passageIndexInSession * max(1, adjustedDueBudget)) % max(1, metLexical.count)
            let rotated = Array(metLexical.dropFirst(strideStart)) + Array(metLexical.prefix(strideStart))
            selectedDueItemsRaw = Array(rotated.prefix(min(adjustedDueBudget, metLexical.count)))
        } else {
            // Group A / Solo Pilot / Stage 1: Forgetting-timed due priority!
            let spacingIntervals: Set<Int> = [1, 2, 4, 8, 16]
            let scoredDue: [(SyllabusItem, Double)] = activeLexical.compactMap { item in
                guard let rec = records[item.id], rec.meetingsCount > 0, !rec.isHeldBackControl else { return nil }
                let p = rec.predictedRecall(at: now)
                let sessionGap = profile.currentSessionNumber - (rec.lastMeetingSession ?? profile.currentSessionNumber)
                let hitsSpacingMilestone = spacingIntervals.contains(sessionGap)
                guard rec.needsExtraMeetingThisSession || p <= modelEngine.parameters.dueRecallThreshold || hitsSpacingMilestone else {
                    return nil
                }
                var priority = 0.0
                if rec.needsExtraMeetingThisSession {
                    priority += 100.0
                }
                if p >= 0.70 && p <= 0.85 {
                    priority += 50.0 + (0.85 - p) * 100.0 // Sweet spot!
                } else if p < 0.70 {
                    priority += 35.0 + p * 20.0
                } else if hitsSpacingMilestone {
                    priority += 25.0
                }
                return (item, priority)
            }
            selectedDueItemsRaw = scoredDue
                .sorted { $0.1 > $1.1 }
                .prefix(adjustedDueBudget)
                .map(\.0)
        }

        // Step 1 Zero-Cost Feasibility Check (Protocol p. 7):
        // From Session 5+, 3 uses of each new item + 1 use of each due item must fit inside 23% of target length.
        // If they do not, split the lists across two passages (excess remains in queue for the next passage).
        let selectedNewItems: [SyllabusItem]
        let selectedDueItems: [SyllabusItem]
        if isBootstrap {
            selectedNewItems = selectedNewItemsRaw
            selectedDueItems = selectedDueItemsRaw
        } else {
            let feasible = Self.enforceFeasibility(
                newItems: selectedNewItemsRaw,
                dueItems: selectedDueItemsRaw,
                targetWords: targetPassageWords,
                scheduledShareCap: modelEngine.parameters.scheduledShareCap,
                minNewOccurrences: modelEngine.parameters.newItemMinOccurrences
            )
            selectedNewItems = feasible.newItems
            selectedDueItems = feasible.dueItems
        }

        // 3. Select Assigned Grammar Patterns (1-2 per passage matching current week, Rule 6)
        let weekGrammarCandidates = activeGrammar.filter { g in
            let gw = g.grammarWeek ?? 1
            let rec = records[g.id]
            return gw <= week && (rec?.state != .productive)
        }.sorted { lhs, rhs in
            let rl = records[lhs.id]?.predictedRecall(at: now) ?? 0.0
            let rr = records[rhs.id]?.predictedRecall(at: now) ?? 0.0
            if (records[lhs.id]?.meetingsCount ?? 0) == 0 && (records[rhs.id]?.meetingsCount ?? 0) > 0 { return true }
            return rl < rr
        }
        let selectedGrammar = Array(weekGrammarCandidates.prefix(isBootstrap ? 1 : 2))

        // 4. Compute Allowed Closed Vocabulary Set (Rule 1)
        // Revised protocol (pp. 6–8):
        //  - Bootstrap (Sessions 1–4): NOTHING is pre-loaded as Known! Allowed core (>=80%) is strictly
        //    items met so far + passage due/new lists (+ names/numbers). Remaining <=20% must be glossed
        //    transparent cognates (`transparentCognateLemmas`).
        //  - Full Contract (Sessions 5+): Known + Learning(p > 85%) + due list + new list (+ names/numbers) = 100%.
        var allowedSet: Set<String> = []
        var learningCandidates: [SyllabusItem] = []
        let scheduledIds = Set((selectedDueItems + selectedNewItems).map(\.id))

        for item in activeLexical {
            guard let rec = records[item.id] else { continue }
            if isBootstrap {
                if rec.meetingsCount > 0 {
                    allowedSet.insert(item.spanish.lowercased())
                }
            } else if modelEngine.isSafeUnscheduledVocabulary(record: rec, at: now) {
                allowedSet.insert(item.spanish.lowercased())
            }
            if rec.state == .learning && !scheduledIds.contains(item.id) && learningCandidates.count < 15 {
                learningCandidates.append(item)
            }
        }
        for item in selectedDueItems { allowedSet.insert(item.spanish.lowercased()) }
        for item in selectedNewItems { allowedSet.insert(item.spanish.lowercased()) }

        // Transparent cognates available as glossed <=20% remainder during Bootstrap Sessions 1–4
        let transparentCognates: [String] = isBootstrap
            ? activeLexical.filter { $0.isTransparentCognate }.map { $0.spanish.lowercased() }
            : []

        // Prior neighbor words for Rule 4 prompt guidance
        var priorMap: [String: [String]] = [:]
        for item in selectedDueItems {
            if let words = priorContextByItem[item.id], !words.isEmpty {
                priorMap[item.spanish] = Array(words.prefix(12)).sorted()
            }
        }

        return PassagePlan(
            passageId: UUID().uuidString,
            sessionNumber: profile.currentSessionNumber,
            passageIndexInSession: passageIndexInSession,
            blockType: blockType,
            isBootstrap: isBootstrap,
            isAudioFirst: isAudioFirst,
            targetWPM: weekSpec.paceWPM,
            format: format,
            topicOrEpisodeTitle: topicTitle,
            storySummarySoFar: storySummarySoFar,
            dueItems: selectedDueItems,
            newItems: selectedNewItems,
            grammarPatterns: selectedGrammar,
            allowedKnownLemmas: Array(allowedSet).sorted(),
            holdoutBannedLemmas: holdoutItems.map(\.spanish),
            voice: voice,
            accent: accent,
            priorContextWordsByItem: priorMap,
            learningProbeCandidates: learningCandidates,
            transparentCognateLemmas: transparentCognates
        )
    }

    /// Step 1 Feasibility Check (Protocol p. 7, zero model calls):
    /// Ensures `minNewOccurrences * newItems.count + dueItems.count <= floor(scheduledShareCap * targetWords)`.
    /// If not, trims due items first (and new items only if needed) so the overflow naturally splits to the next passage.
    public static func enforceFeasibility(
        newItems: [SyllabusItem],
        dueItems: [SyllabusItem],
        targetWords: Int = 215,
        scheduledShareCap: Double = 0.23,
        minNewOccurrences: Int = 3
    ) -> (newItems: [SyllabusItem], dueItems: [SyllabusItem], didSplit: Bool) {
        let maxSlots = max(0, Int((Double(targetWords) * scheduledShareCap).rounded(.down)))
        let needed = newItems.count * minNewOccurrences + dueItems.count
        if needed <= maxSlots {
            return (newItems, dueItems, false)
        }
        var keptNew = newItems
        if keptNew.count * minNewOccurrences > maxSlots {
            keptNew = Array(keptNew.prefix(maxSlots / max(1, minNewOccurrences)))
        }
        let remainingForDue = max(0, maxSlots - keptNew.count * minNewOccurrences)
        let keptDue = Array(dueItems.prefix(remainingForDue))
        return (keptNew, keptDue, true)
    }

    /// Selects under-practiced lemmas from the top-2,000 productive coverage target (< 2 correct uses)
    /// so the Speaking Partner steers Output 2 conversations toward them (PDF p. 9).
    public func selectProductiveSteeringTargets(
        allItems: [SyllabusItem],
        records: [String: LearnerItemRecord],
        count: Int = 8
    ) -> [SyllabusItem] {
        allItems.filter { item in
            guard item.isTop2000ProductiveTarget, !item.isHoldout else { return false }
            guard let rec = records[item.id] else { return false }
            return rec.meetingsCount > 0 && rec.productiveUsesCount < modelEngine.parameters.productiveMinUses
        }
        .sorted { lhs, rhs in
            let lu = records[lhs.id]?.productiveUsesCount ?? 0
            let ru = records[rhs.id]?.productiveUsesCount ?? 0
            if lu != ru { return lu < ru }
            return lhs.rankIndex < rhs.rankIndex
        }
        .prefix(count)
        .map { $0 }
    }

    /// Selects 60 random met items for the Weekly Calibration Audit (PDF p. 6).
    public func selectWeeklyAuditItems(
        allItems: [SyllabusItem],
        records: [String: LearnerItemRecord],
        count: Int = 60,
        now: Date = Date()
    ) -> [AuditPromptItem] {
        let itemMap = Dictionary(uniqueKeysWithValues: allItems.map { ($0.id, $0) })
        let met: [AuditPromptItem] = records.values.compactMap { rec in
            guard rec.meetingsCount > 0, let item = itemMap[rec.itemId], item.itemType == .lemma || item.itemType == .chunk else {
                return nil
            }
            return AuditPromptItem(
                item: item,
                predictedRecallAtAudit: rec.predictedRecall(at: now),
                halfLifeDaysAtAudit: rec.halfLifeDays
            )
        }
        return Array(met.shuffled().prefix(count))
    }

    private func estimatedPassagesForWeek(_ week: Int) -> Int {
        // PDF p. 8: ~15 passages in week 1 rising to ~29 by week 7
        switch week {
        case 1: return 15
        case 2: return 18
        case 3: return 21
        case 4: return 23
        case 5: return 25
        case 6: return 27
        default: return 29
        }
    }

    private func deterministicBalancedCapacitySchedule(seed: Int) -> [Int] {
        var arr = [30, 30, 30, 30, 60, 60, 60, 60, 120, 120, 120, 120]
        var state = UInt64(bitPattern: Int64(seed == 0 ? 42 : seed))
        for i in stride(from: arr.count - 1, through: 1, by: -1) {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let j = Int(state % UInt64(i + 1))
            arr.swapAt(i, j)
        }
        return arr
    }
}
