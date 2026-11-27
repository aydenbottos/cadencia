import Foundation

// MARK: - Append-Only Evidence Log Event

public enum EvidenceEventType: String, Codable, Sendable {
    case sessionStarted
    case sessionCompleted
    case warmupCheck
    case passageShown
    case passageCompleted
    case wordGlossTapped
    case sentenceReplayed
    case readingPaceSample
    case probeAnswered
    case grammarNoteViewed
    case spokenUtterance
    case writingSubmitted
    case modelHalfLifeUpdated
    case itemStatePromoted
    case weeklyAuditItem
    case weeklyAuditCompleted
    case parametersRefit
    case vocabSizeCheck
    case eitItemScored
    case nativeListeningAnswered
    case dailyDiaryLogged
    case benchTestRun
}

public struct EvidenceEvent: Identifiable, Codable, Sendable {
    public let id: Int64?
    public let eventUUID: String
    public let learnerId: String
    public let sessionId: String?
    public let sessionNumber: Int?
    public let passageId: String?
    public let itemId: String?
    public let eventType: EvidenceEventType
    public let timestamp: Date
    public let predictedRecallBefore: Double?
    public let halfLifeBefore: Double?
    public let halfLifeAfter: Double?
    public let isFirstMeetingGlossed: Bool
    public let isUnseenSentence: Bool
    public let passageUnderstood: Bool?
    public let isUsableEvent: Bool
    public let numericValue: Double?
    public let payloadJSON: String

    public init(
        id: Int64? = nil,
        eventUUID: String = UUID().uuidString,
        learnerId: String,
        sessionId: String? = nil,
        sessionNumber: Int? = nil,
        passageId: String? = nil,
        itemId: String? = nil,
        eventType: EvidenceEventType,
        timestamp: Date = Date(),
        predictedRecallBefore: Double? = nil,
        halfLifeBefore: Double? = nil,
        halfLifeAfter: Double? = nil,
        isFirstMeetingGlossed: Bool = false,
        isUnseenSentence: Bool = true,
        passageUnderstood: Bool? = nil,
        isUsableEvent: Bool = false,
        numericValue: Double? = nil,
        payloadJSON: String = "{}"
    ) {
        self.id = id
        self.eventUUID = eventUUID
        self.learnerId = learnerId
        self.sessionId = sessionId
        self.sessionNumber = sessionNumber
        self.passageId = passageId
        self.itemId = itemId
        self.eventType = eventType
        self.timestamp = timestamp
        self.predictedRecallBefore = predictedRecallBefore
        self.halfLifeBefore = halfLifeBefore
        self.halfLifeAfter = halfLifeAfter
        self.isFirstMeetingGlossed = isFirstMeetingGlossed
        self.isUnseenSentence = isUnseenSentence
        self.passageUnderstood = passageUnderstood
        self.isUsableEvent = isUsableEvent
        self.numericValue = numericValue
        self.payloadJSON = payloadJSON
    }
}

// MARK: - Weekly 60-Item Calibration Audit

public struct AuditPromptItem: Identifiable, Codable, Sendable {
    public var id: String { item.id }
    public let item: SyllabusItem
    public let predictedRecallAtAudit: Double
    public let halfLifeDaysAtAudit: Double
}

public struct AuditItemResponse: Identifiable, Codable, Sendable {
    public var id: String { itemId }
    public let itemId: String
    public let spanish: String
    public let expectedGloss: String
    public let learnerTypedMeaning: String
    public let predictedRecall: Double
    public let observedCorrect: Bool
    public let gradingNote: String
}

public struct WeeklyAuditResult: Identifiable, Codable, Sendable {
    public let id: String
    public let week: Int
    public let completedAt: Date
    public let itemCount: Int
    public let meanPredictedRecall: Double
    public let meanObservedRecall: Double
    public let absoluteCalibrationError: Double
    public let exceededTenPointThreshold: Bool
    public let didRefitModel: Bool
    public let durationSeconds: Double
    public let responses: [AuditItemResponse]
}

// MARK: - Research Measures (Outside 60h Timer)

public struct VocabSizeTestResult: Identifiable, Codable, Sendable {
    public let id: String
    public let checkpointLabel: String // Baseline, Hour 20, Hour 40, Hour 60, 8-Week Retest
    public let completedAt: Date
    public let realWordsTested: Int
    public let realWordsClaimed: Int
    public let pseudowordsTested: Int
    public let pseudowordsFalseAlarms: Int
    public let holdoutWordsTested: Int
    public let holdoutWordsClaimed: Int
    public let correctedEstimatedVocabSize: Int
    public let holdoutCorrectedClaimRate: Double
    public let isInflatedWarning: Bool // > 15% holdout claim rate after guessing correction
    public let typedRecall100Score: Double?
}

public struct EITPrompt: Identifiable, Codable, Sendable {
    public let id: Int
    public let syllables: Int
    public let cefr: CEFRLevel
    public let spanish: String
    public let english: String
    public let ipaEs: String
    public let ipaLatam: String
}

public struct EITItemScore: Identifiable, Codable, Sendable {
    public let id: Int
    public let targetSpanish: String
    public let learnerTranscript: String
    public let learnerIPA: String
    public let score0To4: Int // Standard Ortega 0-4 EIT rubric
    public let wordSimilarity: Double
    public let phonemeAccuracy: Double
}

public struct EITSessionResult: Identifiable, Codable, Sendable {
    public let id: String
    public let checkpointLabel: String
    public let completedAt: Date
    public let totalScore: Int
    public let maxScore: Int
    public let normalizedPercent: Double
    public let items: [EITItemScore]
}

public struct NativeListeningQuestion: Codable, Sendable, Hashable {
    public let q: String
    public let options: [String]
    public let correct: Int
}

public struct NativeListeningPassage: Identifiable, Codable, Sendable {
    public let id: String
    public let checkpointHour: Int
    public let accent: SpanishRegionalAccent
    public let speaker: MagpieVoice
    public let wpm: Int
    public let title: String
    public let transcript: String
    public let questions: [NativeListeningQuestion]
}

// MARK: - Claim 1 / Claim 2 / Claim 3 Live Hypothesis Dashboard

public struct CapacityBucketStat: Identifiable, Codable, Sendable {
    public var id: Int { intakeTarget }
    public let intakeTarget: Int // 30, 60, or 120
    public let sessionsLogged: Int
    public let sevenDayTestedCount: Int
    public let sevenDayRetentionRate: Double
}

public struct HypothesisTelemetrySnapshot: Codable, Sendable {
    public let totalLoggedHours: Double
    public let totalInputHours: Double
    public let totalOutputHours: Double
    public let sessionsCompleted: Int
    public let usableEventsTotal: Int
    public let usableEventsPerInputMinute: Double   // Claim 1 target >= 15.0
    public let comprehensionProbeAccuracy: Double   // Claim 1 target >= 0.85
    public let claim1DensityPassing: Bool
    public let capacityStats: [CapacityBucketStat]  // Claim 2: retention(120) >= 0.80 * retention(30)
    public let claim2CapacityRatio120To30: Double?
    public let claim2CapacityPassing: Bool?
    public let stateCounts: [MasteryState: Int]
    public let top2000ProductiveCount: Int          // Claim 3 coverage target: 2,000 used >=2x
    public let top2000ProductiveRatio: Double
    public let latestAuditCalibrationGap: Double?
    public let holdBackScheduledRetentionDelta: Double? // Solo pilot 10% hold-back comparison
}

public struct BenchTestStateReport: Identifiable, Codable, Sendable {
    public var id: Int { simulatedHour }
    public let simulatedHour: Int
    public let knownCount: Int
    public let passagesAttempted: Int
    public let passedWithinThreeRewrites: Int
    public let averageRewrites: Double
    public let averageScheduledShare: Double
    public let averageClosedVocabCompliance: Double
}

public struct BenchTestSummary: Codable, Sendable {
    public let runAt: Date
    public let totalPassages: Int
    public let overallPassRate: Double      // Stage 0 gate: >= 0.95
    public let overallScheduledShare: Double // Stage 0 gate: >= 0.20
    public let gatePassed: Bool
    public let perHourStates: [BenchTestStateReport]
}
