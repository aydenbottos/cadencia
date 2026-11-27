import Foundation

// MARK: - Weekly Schedule & 75-Minute Session Structure

public struct WeeklyPlanSpec: Sendable, Codable, Hashable {
    public let week: Int
    public let paceWPM: Int
    public let newItemsPerSession: Int
    public let cumulativeItemsTarget: Int
    public let scheduledWordShareTarget: Double
    public let audioFirstFraction: Double
    public let hasWritingTasks: Bool
    public let hasExamSpeakingFormat: Bool
    public let hasUnscriptedNativeAudio: Bool

    /// Exact weekly schedule table from protocol PDF p. 8
    public static let eightWeekPlan: [WeeklyPlanSpec] = [
        WeeklyPlanSpec(week: 1, paceWPM: 75,  newItemsPerSession: 73,  cumulativeItemsTarget: 440,  scheduledWordShareTarget: 0.11, audioFirstFraction: 0.00, hasWritingTasks: false, hasExamSpeakingFormat: false, hasUnscriptedNativeAudio: false),
        WeeklyPlanSpec(week: 2, paceWPM: 87,  newItemsPerSession: 105, cumulativeItemsTarget: 1070, scheduledWordShareTarget: 0.21, audioFirstFraction: 0.20, hasWritingTasks: false, hasExamSpeakingFormat: false, hasUnscriptedNativeAudio: false),
        WeeklyPlanSpec(week: 3, paceWPM: 99,  newItemsPerSession: 110, cumulativeItemsTarget: 1730, scheduledWordShareTarget: 0.22, audioFirstFraction: 0.50, hasWritingTasks: true,  hasExamSpeakingFormat: false, hasUnscriptedNativeAudio: false),
        WeeklyPlanSpec(week: 4, paceWPM: 111, newItemsPerSession: 115, cumulativeItemsTarget: 2420, scheduledWordShareTarget: 0.22, audioFirstFraction: 0.50, hasWritingTasks: true,  hasExamSpeakingFormat: false, hasUnscriptedNativeAudio: false),
        WeeklyPlanSpec(week: 5, paceWPM: 123, newItemsPerSession: 125, cumulativeItemsTarget: 3170, scheduledWordShareTarget: 0.22, audioFirstFraction: 0.60, hasWritingTasks: true,  hasExamSpeakingFormat: true,  hasUnscriptedNativeAudio: false),
        WeeklyPlanSpec(week: 6, paceWPM: 135, newItemsPerSession: 135, cumulativeItemsTarget: 3980, scheduledWordShareTarget: 0.21, audioFirstFraction: 0.70, hasWritingTasks: true,  hasExamSpeakingFormat: true,  hasUnscriptedNativeAudio: false),
        WeeklyPlanSpec(week: 7, paceWPM: 140, newItemsPerSession: 87,  cumulativeItemsTarget: 4500, scheduledWordShareTarget: 0.17, audioFirstFraction: 0.80, hasWritingTasks: true,  hasExamSpeakingFormat: true,  hasUnscriptedNativeAudio: false),
        WeeklyPlanSpec(week: 8, paceWPM: 140, newItemsPerSession: 0,   cumulativeItemsTarget: 4500, scheduledWordShareTarget: 0.07, audioFirstFraction: 0.80, hasWritingTasks: true,  hasExamSpeakingFormat: true,  hasUnscriptedNativeAudio: true),
    ]

    public static func forWeek(_ week: Int) -> WeeklyPlanSpec {
        let clamped = min(max(week, 1), 8)
        return eightWeekPlan[clamped - 1]
    }
}

public enum SessionBlockType: String, Codable, Sendable, CaseIterable {
    case warmup      // Minutes 0–3:  20 quick meaning checks on shakiest items from last session
    case input1      // Minutes 3–28: Passages with audio + text together, 2-3 probes after each
    case output1     // Minutes 28–40: Retell passage just heard + 2-3 follow-up questions aloud
    case input2      // Minutes 40–62: New passages, audio first, text shown only after probes
    case output2     // Minutes 62–75: Role-play / open talk / SIELE speaking / 2x-weekly writing
    case closing     // Post-session 1-7 fatigue & enjoyment rating + daily diary

    public var title: String {
        switch self {
        case .warmup: return "Warm-Up"
        case .input1: return "Input I · Read & Listen"
        case .output1: return "Output I · Retell"
        case .input2: return "Input II · Audio First"
        case .output2: return "Output II · Conversation"
        case .closing: return "Session Reflection"
        }
    }

    public var durationSeconds: TimeInterval {
        switch self {
        case .warmup: return 3 * 60      // 3 min
        case .input1: return 25 * 60     // 25 min
        case .output1: return 12 * 60    // 12 min
        case .input2: return 22 * 60     // 22 min
        case .output2: return 13 * 60    // 13 min (total = 75 min)
        case .closing: return 0
        }
    }

    public var isInputBlock: Bool {
        self == .input1 || self == .input2
    }

    public var isOutputBlock: Bool {
        self == .output1 || self == .output2
    }
}

// MARK: - Voice & Regional Accent Rotation (Rule 8)

public enum MagpieVoice: String, Codable, Sendable, CaseIterable {
    case sofia = "Sofia"
    case aria = "Aria"
    case jason = "Jason"
    case leo = "Leo"
    case john = "John"

    public var speakerIndex: Int {
        switch self {
        case .john: return 0
        case .sofia: return 1
        case .aria: return 2
        case .jason: return 3
        case .leo: return 4
        }
    }

    /// NVIDIA NeMo v2607 speaker index mapping (alphabetical in v2607 checkpoint vs CoreML speaker_info.json)
    public var v2607SpeakerIndex: Int {
        switch self {
        case .aria: return 0
        case .jason: return 1
        case .john: return 2
        case .leo: return 3
        case .sofia: return 4
        }
    }
}

public enum SpanishRegionalAccent: String, Codable, Sendable, CaseIterable {
    case castilian = "es-ES"      // Madrid / Spain — distinción (/θ/), apical /s/
    case latinAmerican = "es-419" // Mexico & Colombia — seseo (/s/), yeísmo (/ʝ/)
    case rioplatense = "es-AR"    // Buenos Aires — seseo (/s/), sheísmo (/ʃ/)

    public var displayName: String {
        switch self {
        case .castilian: return "Castellano · Madrid"
        case .latinAmerican: return "Latinoamérica · CDMX"
        case .rioplatense: return "Rioplatense · Buenos Aires"
        }
    }
}

// MARK: - Passage, Probes, Verification & Output Turns

public enum PassageFormat: String, Codable, Sendable {
    case serialEpisode
    case nonfictionBrief
    case bootstrapMicroStory
}

public struct PassagePlan: Codable, Sendable {
    public let passageId: String
    public let sessionNumber: Int
    public let passageIndexInSession: Int
    public let blockType: SessionBlockType
    public let isBootstrap: Bool
    public let isAudioFirst: Bool
    public let targetWPM: Int
    public let format: PassageFormat
    public let topicOrEpisodeTitle: String
    public let storySummarySoFar: String
    public let dueItems: [SyllabusItem]
    public let newItems: [SyllabusItem]
    public let grammarPatterns: [SyllabusItem]
    public let allowedKnownLemmas: [String]
    public let holdoutBannedLemmas: [String]
    public let voice: MagpieVoice
    public let accent: SpanishRegionalAccent
    public let priorContextWordsByItem: [String: [String]]
}

public struct NewItemGloss: Codable, Sendable, Hashable {
    public let itemId: String
    public let spanish: String
    public let englishGloss: String
    public let contextClueNote: String
    public let visualCueSymbol: String
}

public struct PassageProbe: Identifiable, Codable, Sendable, Hashable {
    public let id: String
    public let targetDueItemId: String
    public let targetDueLemma: String
    public let questionEn: String
    public let options: [String]
    public let correctOptionIndex: Int
    public let hingeExplanation: String
}

public struct GrammarMicroNote: Identifiable, Codable, Sendable, Hashable {
    public var id: String { grammarItemId }
    public let grammarItemId: String
    public let title: String
    public let noteEn: String
    public let contrastAEs: String
    public let contrastAEn: String
    public let contrastBEs: String
    public let contrastBEn: String
}

public struct PassageSentence: Identifiable, Codable, Sendable, Hashable {
    public let id: Int
    public let spanish: String
    public let englishLine: String? // Shown on tap in Bootstrap Sessions 1-4
    public let wordTokens: [String]
}

public struct GeneratedPassage: Identifiable, Codable, Sendable {
    public let id: String
    public let sessionNumber: Int
    public let blockType: SessionBlockType
    public let format: PassageFormat
    public let titleEs: String
    public let titleEn: String
    public let episodeSummaryForNext: String
    public let sentences: [PassageSentence]
    public let fullText: String
    public let wordCount: Int
    public let newItemGlosses: [NewItemGloss]
    public let probes: [PassageProbe]
    public let grammarNotes: [GrammarMicroNote]
    public let dueItemIds: [String]
    public let newItemIds: [String]
    public let grammarItemIds: [String]
    public let voice: MagpieVoice
    public let accent: SpanishRegionalAccent
    public let targetWPM: Int
    public let isAudioFirst: Bool
    public let isBootstrap: Bool
    public let rewriteAttempts: Int
    public let dueListWasReduced: Bool
    public let scheduledWordShare: Double
    public let flaggedForNativeAudit: Bool
    public let createdAt: Date
}

public struct ContractViolation: Codable, Sendable, Hashable {
    public let ruleNumber: Int
    public let ruleName: String
    public let detail: String
    public let offendingTokens: [String]
}

public struct VerificationReport: Codable, Sendable {
    public let passed: Bool
    public let totalWords: Int
    public let closedVocabCompliance: Double
    public let scheduledWordShare: Double
    public let missingDueItems: [String]
    public let underSpacedNewItems: [String]
    public let freshContextCollisions: [String]
    public let holdoutLeaks: [String]
    public let offListLemmas: [String]
    public let stageBPassed: Bool
    public let stageBFeedback: String?
    public let violations: [ContractViolation]
}

// MARK: - Speaking & Writing Output Models

public struct PhonemeAlignmentCell: Identifiable, Codable, Sendable, Hashable {
    public let id: Int
    public let expectedPhone: String
    public let observedPhone: String
    public let distance: Double // 0.0 = exact, <=0.35 = close allophone, >0.35 = mismatch
    public let coachingHint: String?
}

public struct SpokenTurnEvaluation: Identifiable, Codable, Sendable {
    public let id: String
    public let timestamp: Date
    public let learnerTranscript: String
    public let learnerIPA: String
    public let canonicalSpanishIPA: String
    public let phonemeAccuracy: Double
    public let alignments: [PhonemeAlignmentCell]
    public let partnerRecastEs: String?
    public let partnerReplyEs: String
    public let correctlyUsedItemIds: [String]
    public let failedGrammarPatternIds: [String]
    public let parakeetLatencyMs: Double
    public let zipaLatencyMs: Double
}

public struct WritingSubmissionFeedback: Identifiable, Codable, Sendable {
    public let id: String
    public let promptEs: String
    public let learnerText: String
    public let recastSentences: [RecastPair]
    public let correctlyUsedItemIds: [String]
    public let failedGrammarPatternIds: [String]
    public let wordCount: Int
    public let elapsedSeconds: Double
}

public struct RecastPair: Codable, Sendable, Hashable {
    public let original: String
    public let corrected: String
    public let hadError: Bool
    public let failedPatternId: String?
}

public struct SessionSummary: Identifiable, Codable, Sendable {
    public let id: String
    public let sessionNumber: Int
    public let week: Int
    public let studyMode: StudyMode
    public let startedAt: Date
    public let endedAt: Date
    public let activeInputSeconds: Double
    public let activeOutputSeconds: Double
    public let passagesCompleted: Int
    public let runningWordsProcessed: Int
    public let usableEventsCount: Int
    public let usableEventsPerInputMinute: Double
    public let probesCorrect: Int
    public let probesTotal: Int
    public let newItemsIntroduced: Int
    public let itemsPromotedToKnown: Int
    public let itemsPromotedToProductive: Int
    public let fatigueRating: Int?     // 1-7
    public let enjoymentRating: Int?   // 1-7
    public let noOtherSpanishConfirmed: Bool
}
