import Foundation

// MARK: - Syllabus & Item Bank Models

public enum ItemType: String, Codable, Sendable, CaseIterable {
    case lemma
    case chunk
    case grammar
    case holdout
}

public enum CEFRLevel: String, Codable, Sendable, CaseIterable, Comparable {
    case a1 = "A1"
    case a2 = "A2"
    case b1 = "B1"
    case b2 = "B2"
    case c1 = "C1"

    public static func < (lhs: CEFRLevel, rhs: CEFRLevel) -> Bool {
        let order: [CEFRLevel] = [.a1, .a2, .b1, .b2, .c1]
        return (order.firstIndex(of: lhs) ?? 0) < (order.firstIndex(of: rhs) ?? 0)
    }
}

public struct SyllabusItem: Identifiable, Codable, Sendable, Hashable {
    public let id: String
    public let itemType: ItemType
    public let rankIndex: Int
    public let spanish: String
    public let englishGloss: String
    public let pos: String
    public let cefr: CEFRLevel
    public let ipaEs: String
    public let ipaLatam: String
    public let isTransparentCognate: Bool
    public let cognateScore: Double
    public let isBootstrap: Bool
    public let isHoldout: Bool
    public let isTop2000ProductiveTarget: Bool
    public let zipfFrequency: Double
    public let visualSymbol: String
    public let grammarCategory: String?
    public let grammarWeek: Int?
    public let microNoteEn: String?
    public let contrastAEs: String?
    public let contrastAEn: String?
    public let contrastBEs: String?
    public let contrastBEn: String?
    public let literalGloss: String?
    public let exampleEs: String?

    public init(
        id: String,
        itemType: ItemType,
        rankIndex: Int,
        spanish: String,
        englishGloss: String,
        pos: String,
        cefr: CEFRLevel,
        ipaEs: String,
        ipaLatam: String,
        isTransparentCognate: Bool,
        cognateScore: Double,
        isBootstrap: Bool,
        isHoldout: Bool,
        isTop2000ProductiveTarget: Bool,
        zipfFrequency: Double,
        visualSymbol: String,
        grammarCategory: String? = nil,
        grammarWeek: Int? = nil,
        microNoteEn: String? = nil,
        contrastAEs: String? = nil,
        contrastAEn: String? = nil,
        contrastBEs: String? = nil,
        contrastBEn: String? = nil,
        literalGloss: String? = nil,
        exampleEs: String? = nil
    ) {
        self.id = id
        self.itemType = itemType
        self.rankIndex = rankIndex
        self.spanish = spanish
        self.englishGloss = englishGloss
        self.pos = pos
        self.cefr = cefr
        self.ipaEs = ipaEs
        self.ipaLatam = ipaLatam
        self.isTransparentCognate = isTransparentCognate
        self.cognateScore = cognateScore
        self.isBootstrap = isBootstrap
        self.isHoldout = isHoldout
        self.isTop2000ProductiveTarget = isTop2000ProductiveTarget
        self.zipfFrequency = zipfFrequency
        self.visualSymbol = visualSymbol
        self.grammarCategory = grammarCategory
        self.grammarWeek = grammarWeek
        self.microNoteEn = microNoteEn
        self.contrastAEs = contrastAEs
        self.contrastAEn = contrastAEn
        self.contrastBEs = contrastBEs
        self.contrastBEn = contrastBEn
        self.literalGloss = literalGloss
        self.exampleEs = exampleEs
    }
}

// MARK: - Learner Model State (Duolingo Half-Life Curve)

public enum MasteryState: String, Codable, Sendable, CaseIterable {
    case new
    case learning
    case known
    case productive

    public var displayName: String {
        switch self {
        case .new: return "New"
        case .learning: return "Learning"
        case .known: return "Known"
        case .productive: return "Productive"
        }
    }
}

public enum LearningOutcome: String, Codable, Sendable {
    case probeCorrect          // Strong success: x2.2
    case productiveCorrect     // Strong success: x2.2 (+ productive use)
    case passedNormalPace      // Weak success: x1.6
    case doubtReplayOrSlow     // Doubt: unchanged + 1 extra meeting this session
    case tappedGloss           // Failure: x0.5
    case probeWrong            // Failure: x0.5
    case speakingPatternError  // Failure: x0.5 (error on scheduled pattern)
}

public struct LearnerItemRecord: Identifiable, Codable, Sendable, Hashable {
    public var id: String { itemId }
    public let learnerId: String
    public let itemId: String
    public var halfLifeDays: Double
    public var lastMeetingAt: Date?
    public var firstMeetingSession: Int?
    public var lastMeetingSession: Int?
    public var meetingsCount: Int
    public var successesCount: Int
    public var earSuccessesCount: Int
    public var productiveUsesCount: Int
    public var lookupsCount: Int
    public var failuresCount: Int
    public var state: MasteryState
    public var needsExtraMeetingThisSession: Bool
    public var isHeldBackControl: Bool
    public var lapseSparesUsed: Int

    public init(
        learnerId: String,
        itemId: String,
        halfLifeDays: Double,
        lastMeetingAt: Date? = nil,
        firstMeetingSession: Int? = nil,
        lastMeetingSession: Int? = nil,
        meetingsCount: Int = 0,
        successesCount: Int = 0,
        earSuccessesCount: Int = 0,
        productiveUsesCount: Int = 0,
        lookupsCount: Int = 0,
        failuresCount: Int = 0,
        state: MasteryState = .new,
        needsExtraMeetingThisSession: Bool = false,
        isHeldBackControl: Bool = false,
        lapseSparesUsed: Int = 0
    ) {
        self.learnerId = learnerId
        self.itemId = itemId
        self.halfLifeDays = halfLifeDays
        self.lastMeetingAt = lastMeetingAt
        self.firstMeetingSession = firstMeetingSession
        self.lastMeetingSession = lastMeetingSession
        self.meetingsCount = meetingsCount
        self.successesCount = successesCount
        self.earSuccessesCount = earSuccessesCount
        self.productiveUsesCount = productiveUsesCount
        self.lookupsCount = lookupsCount
        self.failuresCount = failuresCount
        self.state = state
        self.needsExtraMeetingThisSession = needsExtraMeetingThisSession
        self.isHeldBackControl = isHeldBackControl
        self.lapseSparesUsed = lapseSparesUsed
    }

    /// Duolingo half-life recall curve: p = 2^(-\Delta / h)
    public func predictedRecall(at now: Date = Date()) -> Double {
        guard let last = lastMeetingAt, meetingsCount > 0 else { return 0.0 }
        let deltaDays = max(0.0, now.timeIntervalSince(last) / 86_400.0)
        let h = max(halfLifeDays, 1e-6)
        return pow(2.0, -deltaDays / h)
    }
}

// MARK: - Study Mode & Protocol Parameters

public enum StudyMode: String, Codable, Sendable, CaseIterable, Identifiable {
    public var id: String { rawValue }
    case groupAFullSystem = "group_a_full"
    case groupBSchedulerOff = "group_b_scheduler_off"
    case stage1CapacityTest = "stage_1_capacity"
    case soloPilot = "solo_pilot"

    public var title: String {
        switch self {
        case .groupAFullSystem: return "Group A · Full System"
        case .groupBSchedulerOff: return "Group B · Scheduler Off"
        case .stage1CapacityTest: return "Stage 1 · Capacity Test"
        case .soloPilot: return "Solo Pilot · 8 Weeks"
        }
    }

    public var subtitle: String {
        switch self {
        case .groupAFullSystem:
            return "Full forgetting-timed scheduling across 4,500 items, 48 sessions (60h)."
        case .groupBSchedulerOff:
            return "Control mode: same app, topics, known share & output, but natural frequency order without forgetting timing."
        case .stage1CapacityTest:
            return "12 sessions balanced across 30, 60, and 120 new items/session with 7-day delayed recall audits."
        case .soloPilot:
            return "First 2 weeks alternate 30/60/120 intake + 10% random never-scheduled hold-back control."
        }
    }
}

public struct ProtocolParameters: Codable, Sendable, Equatable {
    public var strongSuccessMultiplier: Double
    public var weakSuccessMultiplier: Double
    public var failureMultiplier: Double
    public var dueRecallThreshold: Double
    public var scheduledShareCap: Double
    public var scheduledShareFloor: Double
    public var maxNewItemsPerSession: Int
    public var maxNewItemsPerPassage: Int
    public var passageMinWords: Int
    public var passageMaxWords: Int
    public var newItemMinOccurrences: Int
    public var newItemMinWordSpacing: Int
    public var knownMinHalfLifeDays: Double
    public var knownMinSuccesses: Int
    public var knownMinEarSuccesses: Int
    public var productiveMinUses: Int
    public var doubtPaceMultiplier: Double
    public var newRegularHalfLifeDays: Double
    public var newCognateHalfLifeDays: Double
    public var isFrozen: Bool

    public init(
        strongSuccessMultiplier: Double,
        weakSuccessMultiplier: Double,
        failureMultiplier: Double,
        dueRecallThreshold: Double,
        scheduledShareCap: Double,
        scheduledShareFloor: Double,
        maxNewItemsPerSession: Int,
        maxNewItemsPerPassage: Int,
        passageMinWords: Int,
        passageMaxWords: Int,
        newItemMinOccurrences: Int,
        newItemMinWordSpacing: Int,
        knownMinHalfLifeDays: Double,
        knownMinSuccesses: Int,
        knownMinEarSuccesses: Int,
        productiveMinUses: Int,
        doubtPaceMultiplier: Double,
        newRegularHalfLifeDays: Double,
        newCognateHalfLifeDays: Double,
        isFrozen: Bool
    ) {
        self.strongSuccessMultiplier = strongSuccessMultiplier
        self.weakSuccessMultiplier = weakSuccessMultiplier
        self.failureMultiplier = failureMultiplier
        self.dueRecallThreshold = dueRecallThreshold
        self.scheduledShareCap = scheduledShareCap
        self.scheduledShareFloor = scheduledShareFloor
        self.maxNewItemsPerSession = maxNewItemsPerSession
        self.maxNewItemsPerPassage = maxNewItemsPerPassage
        self.passageMinWords = passageMinWords
        self.passageMaxWords = passageMaxWords
        self.newItemMinOccurrences = newItemMinOccurrences
        self.newItemMinWordSpacing = newItemMinWordSpacing
        self.knownMinHalfLifeDays = knownMinHalfLifeDays
        self.knownMinSuccesses = knownMinSuccesses
        self.knownMinEarSuccesses = knownMinEarSuccesses
        self.productiveMinUses = productiveMinUses
        self.doubtPaceMultiplier = doubtPaceMultiplier
        self.newRegularHalfLifeDays = newRegularHalfLifeDays
        self.newCognateHalfLifeDays = newCognateHalfLifeDays
        self.isFrozen = isFrozen
    }

    /// Exact defaults from Signal-starvation hypothesis test protocol (pp. 5-7, 14-15)
    public static let protocolDefaults = ProtocolParameters(
        strongSuccessMultiplier: 2.2,
        weakSuccessMultiplier: 1.6,
        failureMultiplier: 0.5,
        dueRecallThreshold: 0.85,
        scheduledShareCap: 0.23,
        scheduledShareFloor: 0.18,
        maxNewItemsPerSession: 135,
        maxNewItemsPerPassage: 6,
        passageMinWords: 180,
        passageMaxWords: 250,
        newItemMinOccurrences: 3,
        newItemMinWordSpacing: 30,
        knownMinHalfLifeDays: 7.0,
        knownMinSuccesses: 4,
        knownMinEarSuccesses: 1,
        productiveMinUses: 2,
        doubtPaceMultiplier: 2.5,
        newRegularHalfLifeDays: 15.0 / 1440.0, // 15 minutes
        newCognateHalfLifeDays: 1.0,           // 1 day
        isFrozen: false
    )
}

public struct SerialGenre: Identifiable, Codable, Sendable, Hashable {
    public let id: String
    public let titleEs: String
    public let titleEn: String
    public let tagline: String
    public let setting: String
    public let premise: String
    public let accentPalette: String
    public let protagonist: String

    public init(
        id: String,
        titleEs: String,
        titleEn: String,
        tagline: String,
        setting: String,
        premise: String,
        accentPalette: String,
        protagonist: String
    ) {
        self.id = id
        self.titleEs = titleEs
        self.titleEn = titleEn
        self.tagline = tagline
        self.setting = setting
        self.premise = premise
        self.accentPalette = accentPalette
        self.protagonist = protagonist
    }
}

public struct LearnerProfile: Codable, Sendable {
    public let learnerId: String
    public var codedGroupLabel: String
    public var displayName: String
    public var studyMode: StudyMode
    public var selectedSerialId: String
    public var nonfictionTopics: [String]
    public var currentSessionNumber: Int // 1...48
    public var totalLoggedSeconds: Double
    public var medianReadingWPM: Double
    public var parameters: ProtocolParameters
    public var createdAt: Date

    public init(
        learnerId: String,
        codedGroupLabel: String,
        displayName: String,
        studyMode: StudyMode,
        selectedSerialId: String,
        nonfictionTopics: [String],
        currentSessionNumber: Int,
        totalLoggedSeconds: Double,
        medianReadingWPM: Double,
        parameters: ProtocolParameters,
        createdAt: Date
    ) {
        self.learnerId = learnerId
        self.codedGroupLabel = codedGroupLabel
        self.displayName = displayName
        self.studyMode = studyMode
        self.selectedSerialId = selectedSerialId
        self.nonfictionTopics = nonfictionTopics
        self.currentSessionNumber = currentSessionNumber
        self.totalLoggedSeconds = totalLoggedSeconds
        self.medianReadingWPM = medianReadingWPM
        self.parameters = parameters
        self.createdAt = createdAt
    }

    public var currentWeek: Int {
        min(8, max(1, ((currentSessionNumber - 1) / 6) + 1))
    }

    public var isBootstrapSession: Bool {
        currentSessionNumber <= 4
    }
}
