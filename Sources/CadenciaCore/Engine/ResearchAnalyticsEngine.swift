import Foundation

/// Computes every scientific metric, hypothesis claim check, assessment score, Stage 0 bench simulation,
/// and blind JSONL/CSV research export strictly from the immutable append-only `evidence_events` log
/// and local SQLite records (PDF pp. 1–3, 9–14).
public struct ResearchAnalyticsEngine: Sendable {
    public init() {}

    /// Computes live telemetry for Claims 1, 2, and 3 directly from the append-only evidence log.
    public func computeTelemetry(
        profile: LearnerProfile,
        allItems: [SyllabusItem],
        records: [String: LearnerItemRecord],
        sessions: [SessionSummary],
        evidenceEvents: [EvidenceEvent],
        audits: [WeeklyAuditResult],
        capacityStats: [CapacityBucketStat]
    ) -> HypothesisTelemetrySnapshot {
        let totalInputSec = sessions.reduce(0.0) { $0 + $1.activeInputSeconds }
        let totalOutputSec = sessions.reduce(0.0) { $0 + $1.activeOutputSeconds }
        let totalHours = (totalInputSec + totalOutputSec) / 3600.0

        // Count usable events strictly from append-only evidence_events
        let usableEvents = evidenceEvents.filter(\.isUsableEvent).count
        let inputMinutes = max(totalInputSec / 60.0, 0.001)
        let usablePerMin = totalInputSec > 0 ? Double(usableEvents) / inputMinutes : 0.0

        // Probe accuracy from evidence log
        let probeEvents = evidenceEvents.filter { $0.eventType == .probeAnswered }
        let probesCorrect = probeEvents.filter { ($0.numericValue ?? 0) >= 0.5 }.count
        let probeAcc = probeEvents.isEmpty ? 1.0 : Double(probesCorrect) / Double(probeEvents.count)

        // Claim 1: >= 15 usable events/min AND >= 85% comprehension probe accuracy
        let claim1Passing = usablePerMin >= 15.0 && probeAcc >= 0.85

        // Claim 2: 7-day retention at 120 new/session >= 80% of retention at 30/session
        let r30 = capacityStats.first(where: { $0.intakeTarget == 30 && $0.sevenDayTestedCount > 0 })?.sevenDayRetentionRate
        let r120 = capacityStats.first(where: { $0.intakeTarget == 120 && $0.sevenDayTestedCount > 0 })?.sevenDayRetentionRate
        let ratio120To30: Double? = (r30 != nil && r30! > 0.001 && r120 != nil) ? (r120! / r30!) : nil
        let claim2Passing: Bool? = ratio120To30.map { $0 >= 0.80 }

        // State distribution across 4,500 active syllabus items
        var counts: [MasteryState: Int] = [.new: 0, .learning: 0, .known: 0, .productive: 0]
        for item in allItems where !item.isHoldout {
            let st = records[item.id]?.state ?? .new
            counts[st, default: 0] += 1
        }

        // Claim 3 coverage target: each of the 2,000 most frequent lemmas used correctly >= 2 times
        let top2000 = allItems.filter { $0.isTop2000ProductiveTarget && !$0.isHoldout }
        let prod2000 = top2000.filter { (records[$0.id]?.productiveUsesCount ?? 0) >= profile.parameters.productiveMinUses }.count
        let prodRatio = top2000.isEmpty ? 0.0 : Double(prod2000) / Double(top2000.count)

        // Latest weekly audit calibration gap
        let latestGap = audits.first?.absoluteCalibrationError

        return HypothesisTelemetrySnapshot(
            totalLoggedHours: totalHours,
            totalInputHours: totalInputSec / 3600.0,
            totalOutputHours: totalOutputSec / 3600.0,
            sessionsCompleted: sessions.count,
            usableEventsTotal: usableEvents,
            usableEventsPerInputMinute: usablePerMin,
            comprehensionProbeAccuracy: probeAcc,
            claim1DensityPassing: claim1Passing,
            capacityStats: capacityStats,
            claim2CapacityRatio120To30: ratio120To30,
            claim2CapacityPassing: claim2Passing,
            stateCounts: counts,
            top2000ProductiveCount: prod2000,
            top2000ProductiveRatio: prodRatio,
            latestAuditCalibrationGap: latestGap,
            holdBackScheduledRetentionDelta: nil
        )
    }

    // MARK: - Vocabulary Size Test Scoring (Guessing Correction + Hold-Out Guard, PDF pp. 11, 13)

    /// Computes guessing-corrected vocabulary size estimate using Zimmerman/Meara Yes/No correction
    /// and verifies whether hold-out non-cognate claims exceed the 15% inflation guard threshold.
    public func scoreVocabSizeTest(
        checkpointLabel: String,
        realWordsTested: Int,
        realWordsClaimed: Int,
        pseudowordsTested: Int,
        pseudowordsFalseAlarms: Int,
        holdoutWordsTested: Int,
        holdoutWordsClaimed: Int,
        typedRecall100Score: Double?
    ) -> VocabSizeTestResult {
        let hReal = realWordsTested > 0 ? Double(realWordsClaimed) / Double(realWordsTested) : 0.0
        let fPseudo = pseudowordsTested > 0 ? Double(pseudowordsFalseAlarms) / Double(pseudowordsTested) : 0.0
        let hHoldout = holdoutWordsTested > 0 ? Double(holdoutWordsClaimed) / Double(holdoutWordsTested) : 0.0

        // Standard guessing correction: (H - F) / (1 - F) clamped to [0, 1]
        let denom = max(0.05, 1.0 - fPseudo)
        let correctedRealRate = min(1.0, max(0.0, (hReal - fPseudo) / denom))
        let correctedHoldoutRate = min(1.0, max(0.0, (hHoldout - fPseudo) / denom))

        let estimatedSize = Int((correctedRealRate * 5000.0).rounded())
        let inflated = correctedHoldoutRate > 0.15 // PDF p. 13: > 15% after guessing correction -> inflated

        return VocabSizeTestResult(
            id: UUID().uuidString,
            checkpointLabel: checkpointLabel,
            completedAt: Date(),
            realWordsTested: realWordsTested,
            realWordsClaimed: realWordsClaimed,
            pseudowordsTested: pseudowordsTested,
            pseudowordsFalseAlarms: pseudowordsFalseAlarms,
            holdoutWordsTested: holdoutWordsTested,
            holdoutWordsClaimed: holdoutWordsClaimed,
            correctedEstimatedVocabSize: estimatedSize,
            holdoutCorrectedClaimRate: correctedHoldoutRate,
            isInflatedWarning: inflated,
            typedRecall100Score: typedRecall100Score
        )
    }

    // MARK: - Spanish Elicited Imitation Task (EIT) Automated Scoring (Ortega 0–4 Scale)

    public func scoreEITResponse(
        prompt: EITPrompt,
        learnerTranscript: String,
        learnerIPA: String,
        phonemeAccuracy: Double
    ) -> EITItemScore {
        let targetWords = normalizeWords(prompt.spanish)
        let spokenWords = normalizeWords(learnerTranscript)
        let wordSim = normalizedWordOverlap(target: targetWords, spoken: spokenWords)

        // Blend lexical fidelity (Parakeet Ultra) + acoustic phoneme fidelity (ZIPA CR Large)
        let combined = 0.72 * wordSim + 0.28 * phonemeAccuracy
        let score: Int
        if combined >= 0.94 {
            score = 4 // Exact repetition
        } else if combined >= 0.78 {
            score = 3 // Meaning preserved, minor grammatical/form slip
        } else if combined >= 0.55 {
            score = 2 // Half or more preserved, noticeable omissions/changes
        } else if combined >= 0.25 {
            score = 1 // Fragmentary repetition
        } else {
            score = 0 // Silence or unintelligible
        }

        return EITItemScore(
            id: prompt.id,
            targetSpanish: prompt.spanish,
            learnerTranscript: learnerTranscript,
            learnerIPA: learnerIPA,
            score0To4: score,
            wordSimilarity: wordSim,
            phonemeAccuracy: phonemeAccuracy
        )
    }

    // MARK: - Stage 0 Bench Test Simulator (Hours 2, 10, 20, 35, 50 — PDF p. 10)

    /// Constructs a realistic simulated learner state at any target hour (2, 10, 20, 35, 50)
    /// for Stage 0 Bench Testing of the Generator + Verifier contract.
    public func buildSimulatedLearnerState(
        atHour hour: Int,
        allItems: [SyllabusItem],
        parameters: ProtocolParameters = .protocolDefaults
    ) -> (profile: LearnerProfile, records: [String: LearnerItemRecord]) {
        let sessionNum = max(1, min(48, Int((Double(hour) / 1.25).rounded())))
        let now = Date()
        let active = allItems.filter { !$0.isHoldout }.sorted { $0.rankIndex < $1.rankIndex }

        // Map hour -> approximate cumulative met items matching the 8-week simulation table (PDF p. 8)
        let targetMetCount: Int
        switch hour {
        case ...3: targetMetCount = 150
        case 4...12: targetMetCount = 720
        case 13...24: targetMetCount = 1650
        case 25...40: targetMetCount = 2900
        default: targetMetCount = 4150
        }

        var records: [String: LearnerItemRecord] = [:]
        for (idx, item) in active.enumerated() {
            if idx < targetMetCount {
                let isDueSlice = (idx % 9 == 0)
                let hDays = isDueSlice ? 2.5 : (item.isTransparentCognate ? 14.0 : 10.0)
                let deltaDays = isDueSlice ? 0.72 : 0.5 // p ~ 0.81 (due!) vs p ~ 0.96 (safe!)
                let rec = LearnerItemRecord(
                    learnerId: "bench_h\(hour)",
                    itemId: item.id,
                    halfLifeDays: hDays,
                    lastMeetingAt: now.addingTimeInterval(-deltaDays * 86_400),
                    firstMeetingSession: max(1, sessionNum - 5),
                    lastMeetingSession: max(1, sessionNum - 1),
                    meetingsCount: isDueSlice ? 3 : 6,
                    successesCount: isDueSlice ? 2 : 5,
                    earSuccessesCount: isDueSlice ? 0 : 2,
                    productiveUsesCount: idx < targetMetCount / 3 ? 2 : 0,
                    lookupsCount: 0,
                    failuresCount: 0,
                    state: isDueSlice ? .learning : (idx < targetMetCount / 3 ? .productive : .known)
                )
                records[item.id] = rec
            } else {
                records[item.id] = LearnerItemRecord(
                    learnerId: "bench_h\(hour)",
                    itemId: item.id,
                    halfLifeDays: item.isTransparentCognate ? parameters.newCognateHalfLifeDays : parameters.newRegularHalfLifeDays
                )
            }
        }

        let profile = LearnerProfile(
            learnerId: "bench_h\(hour)",
            codedGroupLabel: "BENCH-STAGE0",
            displayName: "Stage 0 Sim (h=\(hour))",
            studyMode: .groupAFullSystem,
            selectedSerialId: "madrid_noir",
            nonfictionTopics: ["Arquitectura", "Ciencia", "Historia", "Gastronomía"],
            currentSessionNumber: sessionNum,
            totalLoggedSeconds: Double(hour) * 3600.0,
            medianReadingWPM: Double(WeeklyPlanSpec.forWeek(((sessionNum - 1) / 6) + 1).paceWPM),
            parameters: parameters,
            createdAt: now
        )
        return (profile, records)
    }

    // MARK: - Blind Research JSONL & 2% Native-Rater Export

    public func exportBlindEvidenceJSONL(
        profile: LearnerProfile,
        events: [EvidenceEvent]
    ) -> Data {
        struct BlindEventRow: Encodable {
            let codedGroup: String
            let eventUUID: String
            let sessionNumber: Int?
            let passageId: String?
            let itemId: String?
            let eventType: String
            let timestampISO: String
            let predictedRecallBefore: Double?
            let halfLifeBefore: Double?
            let halfLifeAfter: Double?
            let isFirstMeetingGlossed: Bool
            let isUnseenSentence: Bool
            let passageUnderstood: Bool?
            let isUsableEvent: Bool
            let numericValue: Double?
        }
        let iso = ISO8601DateFormatter()
        let encoder = JSONEncoder()
        var lines: [String] = []
        lines.reserveCapacity(events.count)
        for ev in events {
            let row = BlindEventRow(
                codedGroup: profile.codedGroupLabel,
                eventUUID: ev.eventUUID,
                sessionNumber: ev.sessionNumber,
                passageId: ev.passageId,
                itemId: ev.itemId,
                eventType: ev.eventType.rawValue,
                timestampISO: iso.string(from: ev.timestamp),
                predictedRecallBefore: ev.predictedRecallBefore,
                halfLifeBefore: ev.halfLifeBefore,
                halfLifeAfter: ev.halfLifeAfter,
                isFirstMeetingGlossed: ev.isFirstMeetingGlossed,
                isUnseenSentence: ev.isUnseenSentence,
                passageUnderstood: ev.passageUnderstood,
                isUsableEvent: ev.isUsableEvent,
                numericValue: ev.numericValue
            )
            if let d = try? encoder.encode(row), let s = String(data: d, encoding: .utf8) {
                lines.append(s)
            }
        }
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    public func exportNativeRaterAuditMarkdown(passages: [GeneratedPassage]) -> String {
        var md = "# Cadencia Weekly 2% Native-Speaker Rater Pack\n\n"
        md += "Total flagged passages: \(passages.count)\n\n"
        for (idx, p) in passages.enumerated() {
            md += "## \(idx + 1). \(p.titleEs) (Session \(p.sessionNumber), \(p.wordCount) words, \(p.accent.displayName))\n\n"
            md += "\(p.fullText)\n\n"
            md += "- Scheduled share: \(String(format: "%.1f%%", p.scheduledWordShare * 100))\n"
            md += "- Rewrites: \(p.rewriteAttempts)\n"
            md += "- Naturalness rating (1–5): [ ]\n"
            md += "- Grammar issues noted: \n\n---\n\n"
        }
        return md
    }

    // MARK: - Helpers

    private func normalizeWords(_ text: String) -> [String] {
        text.lowercased()
            .folding(options: .diacriticInsensitive, locale: Locale(identifier: "es"))
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }

    private func normalizedWordOverlap(target: [String], spoken: [String]) -> Double {
        guard !target.isEmpty, !spoken.isEmpty else { return 0.0 }
        let m = target.count
        let n = spoken.count
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }
        for i in 1...m {
            for j in 1...n {
                let cost = target[i - 1] == spoken[j - 1] ? 0 : 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            }
        }
        let dist = dp[m][n]
        return max(0.0, 1.0 - Double(dist) / Double(max(m, n)))
    }
}
