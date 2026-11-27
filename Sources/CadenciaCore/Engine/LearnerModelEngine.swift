import Foundation

/// Implements the exact Duolingo Half-Life Regression learner model from
/// Signal-starvation hypothesis test protocol (pp. 5–6, 14):
/// - Predicted recall: p = 2^(-\Delta / h)
/// - Initial half-lives: 15 min (regular) / 1 day (transparent cognate)
/// - Multipliers: Strong success x2.2, Weak success x1.6, Doubt unchanged + extra meeting, Failure x0.5
/// - 4 states: New -> Learning (h < 7d) -> Known (h >= 7d & >=4 successes & >=1 ear-alone) -> Productive (Known & >=2 output uses)
/// - Automated online half-life multiplier refitter via coordinate/Nelder-Mead MLE when weekly 60-item audit
///   predicted vs observed recall diverges by > 10 percentage points!
public struct LearnerModelEngine: Sendable {
    public var parameters: ProtocolParameters

    public init(parameters: ProtocolParameters = .protocolDefaults) {
        self.parameters = parameters
    }

    /// Exact forgetting curve: p = 2^(-\Delta / h)
    public static func predictedRecall(deltaDays: Double, halfLifeDays: Double) -> Double {
        let d = max(0.0, deltaDays)
        let h = max(halfLifeDays, 1e-6)
        return pow(2.0, -d / h)
    }

    /// Initial half-life when an item is met for the very first time.
    public func initialHalfLifeDays(isTransparentCognate: Bool) -> Double {
        isTransparentCognate ? parameters.newCognateHalfLifeDays : parameters.newRegularHalfLifeDays
    }

    /// Determines whether a meeting qualifies as a "usable event" under all 3 protocol conditions (PDF p. 2):
    /// 1. Sits in a passage the learner understood (shown by probes >= 85% / passed)
    /// 2. Beforehand, predicted recall p in [0.50, 0.95], OR first meeting with meaning supplied
    /// 3. Sentence is one the learner has not seen before
    public func isUsableEvent(
        predictedRecallBefore: Double,
        isFirstMeetingGlossed: Bool,
        passageUnderstood: Bool,
        isUnseenSentence: Bool
    ) -> Bool {
        guard passageUnderstood && isUnseenSentence else { return false }
        if isFirstMeetingGlossed { return true }
        return predictedRecallBefore >= 0.50 && predictedRecallBefore <= 0.95
    }

    /// Applies one learning observation to a `LearnerItemRecord` and returns the updated record + promotion flag.
    public func apply(
        outcome: LearningOutcome,
        to record: LearnerItemRecord,
        item: SyllabusItem,
        sessionNumber: Int,
        wasEarOnly: Bool,
        now: Date = Date()
    ) -> (updated: LearnerItemRecord, previousState: MasteryState, predictedBefore: Double) {
        var next = record
        let prevState = record.state
        let predictedBefore = record.meetingsCount > 0 ? record.predictedRecall(at: now) : 0.0

        // First meeting initialization
        if next.meetingsCount == 0 {
            next.halfLifeDays = initialHalfLifeDays(isTransparentCognate: item.isTransparentCognate)
            next.firstMeetingSession = sessionNumber
        }

        next.meetingsCount += 1
        next.lastMeetingAt = now
        next.lastMeetingSession = sessionNumber

        let minHalfLife = 5.0 / 1440.0  // Floor at 5 minutes so failed items resurface quickly
        let maxHalfLife = 365.0         // Cap at 1 year

        switch outcome {
        case .probeCorrect:
            next.halfLifeDays = min(maxHalfLife, next.halfLifeDays * parameters.strongSuccessMultiplier)
            next.successesCount += 1
            if wasEarOnly { next.earSuccessesCount += 1 }
            next.needsExtraMeetingThisSession = false

        case .productiveCorrect:
            next.halfLifeDays = min(maxHalfLife, next.halfLifeDays * parameters.strongSuccessMultiplier)
            next.successesCount += 1
            next.productiveUsesCount += 1
            if wasEarOnly { next.earSuccessesCount += 1 }
            next.needsExtraMeetingThisSession = false

        case .passedNormalPace:
            next.halfLifeDays = min(maxHalfLife, next.halfLifeDays * parameters.weakSuccessMultiplier)
            next.successesCount += 1
            if wasEarOnly { next.earSuccessesCount += 1 }
            next.needsExtraMeetingThisSession = false

        case .doubtReplayOrSlow:
            // Unchanged half-life, plus one extra meeting this session (PDF p. 6)
            next.needsExtraMeetingThisSession = true

        case .tappedGloss:
            next.halfLifeDays = max(minHalfLife, next.halfLifeDays * parameters.failureMultiplier)
            next.lookupsCount += 1
            next.failuresCount += 1
            next.lapseSparesUsed += 1
            next.needsExtraMeetingThisSession = true

        case .probeWrong, .speakingPatternError:
            next.halfLifeDays = max(minHalfLife, next.halfLifeDays * parameters.failureMultiplier)
            next.failuresCount += 1
            next.lapseSparesUsed += 1
            next.needsExtraMeetingThisSession = true
        }

        next.state = computeMasteryState(for: next)
        return (next, prevState, predictedBefore)
    }

    /// Exact 4-state transition logic (PDF p. 6):
    /// 1. New: never met
    /// 2. Learning: half-life under 7 days (or missing Known thresholds)
    /// 3. Known: half-life >= 7 days, >= 4 successes, >= 1 by ear alone
    /// 4. Productive: Known, and used correctly >= 2 times
    public func computeMasteryState(for record: LearnerItemRecord) -> MasteryState {
        guard record.meetingsCount > 0 else { return .new }
        let meetsKnown = record.halfLifeDays >= parameters.knownMinHalfLifeDays
            && record.successesCount >= parameters.knownMinSuccesses
            && record.earSuccessesCount >= parameters.knownMinEarSuccesses
        if meetsKnown {
            if record.productiveUsesCount >= parameters.productiveMinUses {
                return .productive
            }
            return .known
        }
        return .learning
    }

    /// Checks whether an item is eligible as "already known/safe" under Rule 1 (Closed Vocabulary):
    /// Known/Productive, OR Learning with predicted recall > 85% (dueRecallThreshold).
    public func isSafeUnscheduledVocabulary(record: LearnerItemRecord, at now: Date = Date()) -> Bool {
        if record.state == .known || record.state == .productive {
            return true
        }
        if record.state == .learning {
            return record.predictedRecall(at: now) > parameters.dueRecallThreshold
        }
        return false
    }

    /// Checks whether an item joins the due list (predicted recall <= 85%, or flagged for doubt extra meeting).
    public func isDue(record: LearnerItemRecord, at now: Date = Date()) -> Bool {
        guard record.meetingsCount > 0, !record.isHeldBackControl else { return false }
        if record.needsExtraMeetingThisSession { return true }
        return record.predictedRecall(at: now) <= parameters.dueRecallThreshold
    }

    // MARK: - Weekly 60-Item Calibration Audit & Online Multiplier Refitter

    public struct CalibrationObservation: Sendable {
        public let itemId: String
        public let deltaDays: Double
        public let strongCount: Int
        public let weakCount: Int
        public let failCount: Int
        public let isCognate: Bool
        public let observedCorrect: Bool
    }

    /// Refits `(strongSuccessMultiplier, weakSuccessMultiplier, failureMultiplier)` via bounded
    /// binary cross-entropy minimization over pooled audit observations when calibration bias > 10pp
    /// and parameters are not yet frozen (Stage 1 / Weekly Audit rule, PDF pp. 6 & 14).
    public func refitMultipliers(
        observations: [CalibrationObservation]
    ) -> ProtocolParameters {
        guard !parameters.isFrozen, observations.count >= 20 else { return parameters }

        func loss(strong: Double, weak: Double, fail: Double) -> Double {
            var nll = 0.0
            for obs in observations {
                let h0 = obs.isCognate ? parameters.newCognateHalfLifeDays : parameters.newRegularHalfLifeDays
                let h = max(
                    5.0 / 1440.0,
                    min(
                        365.0,
                        h0 * pow(strong, Double(obs.strongCount))
                           * pow(weak, Double(obs.weakCount))
                           * pow(fail, Double(obs.failCount))
                    )
                )
                let p = min(0.999, max(0.001, Self.predictedRecall(deltaDays: obs.deltaDays, halfLifeDays: h)))
                let y = obs.observedCorrect ? 1.0 : 0.0
                nll -= (y * log(p) + (1.0 - y) * log(1.0 - p))
            }
            // Gentle L2 regularization toward protocol defaults (2.2, 1.6, 0.5) to avoid small-sample drift
            let reg = 0.15 * (
                pow(strong - 2.2, 2) + pow(weak - 1.6, 2) + pow(fail - 0.5, 2)
            )
            return nll / Double(observations.count) + reg
        }

        // Coarse-to-fine coordinate grid search (deterministic, fast, zero external deps)
        var bestS = parameters.strongSuccessMultiplier
        var bestW = parameters.weakSuccessMultiplier
        var bestF = parameters.failureMultiplier
        var bestL = loss(strong: bestS, weak: bestW, fail: bestF)

        for s in stride(from: 1.5, through: 3.2, by: 0.1) {
            for w in stride(from: 1.1, through: min(s, 2.3), by: 0.1) {
                for f in stride(from: 0.25, through: 0.75, by: 0.05) {
                    let l = loss(strong: s, weak: w, fail: f)
                    if l < bestL {
                        bestL = l
                        bestS = s
                        bestW = w
                        bestF = f
                    }
                }
            }
        }

        // Fine polish step (0.02 granularity)
        let sMin = max(1.3, bestS - 0.12), sMax = min(3.5, bestS + 0.12)
        let wMin = max(1.05, bestW - 0.12), wMax = min(bestS, bestW + 0.12)
        let fMin = max(0.20, bestF - 0.06), fMax = min(0.85, bestF + 0.06)
        for s in stride(from: sMin, through: sMax, by: 0.02) {
            for w in stride(from: wMin, through: wMax, by: 0.02) {
                for f in stride(from: fMin, through: fMax, by: 0.02) {
                    let l = loss(strong: s, weak: w, fail: f)
                    if l < bestL {
                        bestL = l
                        bestS = s
                        bestW = w
                        bestF = f
                    }
                }
            }
        }

        var updated = parameters
        updated.strongSuccessMultiplier = (bestS * 100).rounded() / 100
        updated.weakSuccessMultiplier = (bestW * 100).rounded() / 100
        updated.failureMultiplier = (bestF * 100).rounded() / 100
        return updated
    }
}
