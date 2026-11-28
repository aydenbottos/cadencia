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

    public init(
        week: Int,
        paceWPM: Int,
        newItemsPerSession: Int,
        cumulativeItemsTarget: Int,
        scheduledWordShareTarget: Double,
        audioFirstFraction: Double,
        hasWritingTasks: Bool,
        hasExamSpeakingFormat: Bool,
        hasUnscriptedNativeAudio: Bool
    ) {
        self.week = week
        self.paceWPM = paceWPM
        self.newItemsPerSession = newItemsPerSession
        self.cumulativeItemsTarget = cumulativeItemsTarget
        self.scheduledWordShareTarget = scheduledWordShareTarget
        self.audioFirstFraction = audioFirstFraction
        self.hasWritingTasks = hasWritingTasks
        self.hasExamSpeakingFormat = hasExamSpeakingFormat
        self.hasUnscriptedNativeAudio = hasUnscriptedNativeAudio
    }

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

// MARK: - Native Spanish Tutor Voices (Gradium Phonon-40m-es) & Regional Accent Rotation (Rule 8)

/// Curated native Spanish voices from `gradium/phonon-voices` (CC0/public voice embeddings, ~250 KB each).
/// In Phonon-40m-es the voice embedding itself carries authentic regional accent & prosody:
///  - `es-es` (Castilian distinción): Celia, Iker, Nerea, Marcos
///  - `es-mx` (Latin American seseo): Regina, Leonel, Rebeca, Nico
public enum TutorVoice: String, Codable, Sendable, CaseIterable, Identifiable {
    case celia = "Celia"
    case iker = "Iker"
    case nerea = "Nerea"
    case marcos = "Marcos"
    case regina = "Regina"
    case leonel = "Leonel"
    case rebeca = "Rebeca"
    case nico = "Nico"

    public var id: String { rawValue }

    /// Voice ID in `gradium/phonon-voices` (`voices/<phononVoiceId>.safetensors`).
    public var phononVoiceId: String {
        switch self {
        case .celia: return "rmLBOdHrfkFym9MM"
        case .iker: return "t-_TS1e-0GzDAX02"
        case .nerea: return "EQcQFGkYfMn5YJgx"
        case .marcos: return "jvPx8j8zLGQ3utZz"
        case .regina: return "s58clrg2fe0MO7Y-"
        case .leonel: return "BkuASiOJRrJEB5qA"
        case .rebeca: return "Sx8m6WdXBG06zwZT"
        case .nico: return "PwgPEaweQILZdGc4"
        }
    }

    /// Stem filename written into the exported Phonon Core ML `voices/` folder.
    public var phononBundleName: String { rawValue }

    /// Native regional accent encoded in the voice embedding.
    public var nativeAccent: SpanishRegionalAccent {
        switch self {
        case .celia, .iker, .nerea, .marcos: return .castilian
        case .regina, .leonel, .rebeca, .nico: return .latinAmerican
        }
    }

    public var isFemale: Bool {
        switch self {
        case .celia, .nerea, .regina, .rebeca: return true
        case .iker, .marcos, .leonel, .nico: return false
        }
    }

    /// Matching English Kokoro-82M voice for bilingual tutor conversations (preserves speaker gender).
    public var pairedKokoroEnglishVoice: String {
        isFemale ? "af_heart" : "am_michael"
    }

    public var localeLabel: String {
        nativeAccent == .castilian ? "es-ES" : "es-MX"
    }

    public static let castilianVoices: [TutorVoice] = [.celia, .iker, .nerea, .marcos]
    public static let latinAmericanVoices: [TutorVoice] = [.regina, .leonel, .rebeca, .nico]

    /// Selects a native Phonon voice whose regional accent matches `accent` while respecting `preferred` gender/identity.
    public static func voice(for accent: SpanishRegionalAccent, preferred: TutorVoice = .celia) -> TutorVoice {
        switch accent {
        case .castilian:
            if preferred.nativeAccent == .castilian { return preferred }
            return preferred.isFemale ? .celia : .iker
        case .latinAmerican, .rioplatense:
            if preferred.nativeAccent == .latinAmerican { return preferred }
            return preferred.isFemale ? .regina : .leonel
        }
    }

    /// Maps both current Phonon names and legacy stored SQLite voice names seamlessly.
    public static func fromStoredRawValue(_ raw: String) -> TutorVoice {
        if let exact = TutorVoice(rawValue: raw) { return exact }
        switch raw.lowercased() {
        case "sofia", "celia": return .celia
        case "aria", "nerea": return .nerea
        case "jason", "iker": return .iker
        case "leo", "leonel": return .leonel
        case "john", "marcos": return .marcos
        case "regina": return .regina
        case "rebeca": return .rebeca
        case "nico": return .nico
        default: return .celia
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
        self = Self.fromStoredRawValue(raw)
    }
}

public enum TutorSpeechLanguage: String, Codable, Sendable, Hashable {
    case spanish = "es"
    case english = "en"
}

/// Ordered speech segment within a bilingual tutor turn.
/// Spanish segments are voiced natively by `Phonon-40m-es`; English segments are voiced by `Kokoro-82M`.
public struct TutorSpeechSegment: Identifiable, Codable, Sendable, Hashable {
    public let id: String
    public let language: TutorSpeechLanguage
    public let text: String

    public init(id: String = UUID().uuidString, language: TutorSpeechLanguage, text: String) {
        self.id = id
        self.language = language
        self.text = text
    }

    private enum CodingKeys: String, CodingKey {
        case id, language, text
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = (try? c.decode(String.self, forKey: .id)) ?? UUID().uuidString
        self.language = try c.decode(TutorSpeechLanguage.self, forKey: .language)
        self.text = try c.decode(String.self, forKey: .text)
    }
}

public enum ConversationTutorMode: String, Codable, Sendable, CaseIterable, Identifiable {
    case bilingualCoach = "bilingual"
    case spanishImmersion = "spanish_only"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .bilingualCoach: return "ES + EN Tutor"
        case .spanishImmersion: return "Spanish Only"
        }
    }

    public var allowsEnglish: Bool {
        self == .bilingualCoach
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
    public let voice: TutorVoice
    public let accent: SpanishRegionalAccent
    public let priorContextWordsByItem: [String: [String]]
    /// Rule 5 tier-3 probe candidates (Learning items with p > 85% or already met, used if due + new < 2).
    public let learningProbeCandidates: [SyllabusItem]
    /// Transparent cognates allowed as the glossed <=20% remainder in Bootstrap Sessions 1–4.
    public let transparentCognateLemmas: [String]

    public init(
        passageId: String,
        sessionNumber: Int,
        passageIndexInSession: Int,
        blockType: SessionBlockType,
        isBootstrap: Bool,
        isAudioFirst: Bool,
        targetWPM: Int,
        format: PassageFormat,
        topicOrEpisodeTitle: String,
        storySummarySoFar: String,
        dueItems: [SyllabusItem],
        newItems: [SyllabusItem],
        grammarPatterns: [SyllabusItem],
        allowedKnownLemmas: [String],
        holdoutBannedLemmas: [String],
        voice: TutorVoice,
        accent: SpanishRegionalAccent,
        priorContextWordsByItem: [String: [String]],
        learningProbeCandidates: [SyllabusItem] = [],
        transparentCognateLemmas: [String] = []
    ) {
        self.passageId = passageId
        self.sessionNumber = sessionNumber
        self.passageIndexInSession = passageIndexInSession
        self.blockType = blockType
        self.isBootstrap = isBootstrap
        self.isAudioFirst = isAudioFirst
        self.targetWPM = targetWPM
        self.format = format
        self.topicOrEpisodeTitle = topicOrEpisodeTitle
        self.storySummarySoFar = storySummarySoFar
        self.dueItems = dueItems
        self.newItems = newItems
        self.grammarPatterns = grammarPatterns
        self.allowedKnownLemmas = allowedKnownLemmas
        self.holdoutBannedLemmas = holdoutBannedLemmas
        self.voice = voice
        self.accent = accent
        self.priorContextWordsByItem = priorContextWordsByItem
        self.learningProbeCandidates = learningProbeCandidates
        self.transparentCognateLemmas = transparentCognateLemmas
    }

    /// Rule 5 target probe items in strict protocol priority order:
    /// due list first -> new list -> Learning items in context.
    public var eligibleProbeTargets: [SyllabusItem] {
        var seen: Set<String> = []
        var out: [SyllabusItem] = []
        for item in dueItems + newItems + learningProbeCandidates {
            if seen.insert(item.id).inserted {
                out.append(item)
            }
        }
        return out
    }
}

public struct NewItemGloss: Codable, Sendable, Hashable {
    public let itemId: String
    public let spanish: String
    public let englishGloss: String
    public let contextClueNote: String
    public let visualCueSymbol: String

    public init(itemId: String, spanish: String, englishGloss: String, contextClueNote: String, visualCueSymbol: String) {
        self.itemId = itemId
        self.spanish = spanish
        self.englishGloss = englishGloss
        self.contextClueNote = contextClueNote
        self.visualCueSymbol = visualCueSymbol
    }
}

public struct PassageProbe: Identifiable, Codable, Sendable, Hashable {
    public let id: String
    public let targetDueItemId: String
    public let targetDueLemma: String
    public let questionEn: String
    public let options: [String]
    public let correctOptionIndex: Int
    public let hingeExplanation: String
    /// Bootstrap Rule 5: picture-matching probe on new items when the due list is empty.
    public let isPictureMatch: Bool
    public let optionSymbols: [String]?

    public init(
        id: String,
        targetDueItemId: String,
        targetDueLemma: String,
        questionEn: String,
        options: [String],
        correctOptionIndex: Int,
        hingeExplanation: String,
        isPictureMatch: Bool = false,
        optionSymbols: [String]? = nil
    ) {
        self.id = id
        self.targetDueItemId = targetDueItemId
        self.targetDueLemma = targetDueLemma
        self.questionEn = questionEn
        self.options = options
        self.correctOptionIndex = correctOptionIndex
        self.hingeExplanation = hingeExplanation
        self.isPictureMatch = isPictureMatch
        self.optionSymbols = optionSymbols
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.targetDueItemId = try c.decode(String.self, forKey: .targetDueItemId)
        self.targetDueLemma = try c.decode(String.self, forKey: .targetDueLemma)
        self.questionEn = try c.decode(String.self, forKey: .questionEn)
        self.options = try c.decode([String].self, forKey: .options)
        self.correctOptionIndex = try c.decode(Int.self, forKey: .correctOptionIndex)
        self.hingeExplanation = try c.decode(String.self, forKey: .hingeExplanation)
        self.isPictureMatch = (try? c.decode(Bool.self, forKey: .isPictureMatch)) ?? false
        self.optionSymbols = try? c.decode([String].self, forKey: .optionSymbols)
    }
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

    public init(
        grammarItemId: String,
        title: String,
        noteEn: String,
        contrastAEs: String,
        contrastAEn: String,
        contrastBEs: String,
        contrastBEn: String
    ) {
        self.grammarItemId = grammarItemId
        self.title = title
        self.noteEn = noteEn
        self.contrastAEs = contrastAEs
        self.contrastAEn = contrastAEn
        self.contrastBEs = contrastBEs
        self.contrastBEn = contrastBEn
    }
}

public struct PassageSentence: Identifiable, Codable, Sendable, Hashable {
    public let id: Int
    public let spanish: String
    public let englishLine: String? // Shown on tap in Bootstrap Sessions 1-4
    public let wordTokens: [String]
    /// SF Symbol visual illustration for Bootstrap Sessions 1-4 ("each with a picture and an English line on tap").
    public let visualSymbol: String?

    public init(id: Int, spanish: String, englishLine: String?, wordTokens: [String], visualSymbol: String? = nil) {
        self.id = id
        self.spanish = spanish
        self.englishLine = englishLine
        self.wordTokens = wordTokens
        self.visualSymbol = visualSymbol
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(Int.self, forKey: .id)
        self.spanish = try c.decode(String.self, forKey: .spanish)
        self.englishLine = try? c.decode(String.self, forKey: .englishLine)
        self.wordTokens = try c.decode([String].self, forKey: .wordTokens)
        self.visualSymbol = try? c.decode(String.self, forKey: .visualSymbol)
    }
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
    public let voice: TutorVoice
    public let accent: SpanishRegionalAccent
    public let targetWPM: Int
    public let isAudioFirst: Bool
    public let isBootstrap: Bool
    public let rewriteAttempts: Int
    public let generatorCallsUsed: Int
    public let isReserveFallback: Bool
    public let dueListWasReduced: Bool
    public let scheduledWordShare: Double
    public let flaggedForNativeAudit: Bool
    public let createdAt: Date

    public init(
        id: String,
        sessionNumber: Int,
        blockType: SessionBlockType,
        format: PassageFormat,
        titleEs: String,
        titleEn: String,
        episodeSummaryForNext: String,
        sentences: [PassageSentence],
        fullText: String,
        wordCount: Int,
        newItemGlosses: [NewItemGloss],
        probes: [PassageProbe],
        grammarNotes: [GrammarMicroNote],
        dueItemIds: [String],
        newItemIds: [String],
        grammarItemIds: [String],
        voice: TutorVoice,
        accent: SpanishRegionalAccent,
        targetWPM: Int,
        isAudioFirst: Bool,
        isBootstrap: Bool,
        rewriteAttempts: Int,
        generatorCallsUsed: Int? = nil,
        isReserveFallback: Bool = false,
        dueListWasReduced: Bool,
        scheduledWordShare: Double,
        flaggedForNativeAudit: Bool,
        createdAt: Date
    ) {
        self.id = id
        self.sessionNumber = sessionNumber
        self.blockType = blockType
        self.format = format
        self.titleEs = titleEs
        self.titleEn = titleEn
        self.episodeSummaryForNext = episodeSummaryForNext
        self.sentences = sentences
        self.fullText = fullText
        self.wordCount = wordCount
        self.newItemGlosses = newItemGlosses
        self.probes = probes
        self.grammarNotes = grammarNotes
        self.dueItemIds = dueItemIds
        self.newItemIds = newItemIds
        self.grammarItemIds = grammarItemIds
        self.voice = voice
        self.accent = accent
        self.targetWPM = targetWPM
        self.isAudioFirst = isAudioFirst
        self.isBootstrap = isBootstrap
        self.rewriteAttempts = rewriteAttempts
        self.generatorCallsUsed = generatorCallsUsed ?? (rewriteAttempts + 1)
        self.isReserveFallback = isReserveFallback
        self.dueListWasReduced = dueListWasReduced
        self.scheduledWordShare = scheduledWordShare
        self.flaggedForNativeAudit = flaggedForNativeAudit
        self.createdAt = createdAt
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.sessionNumber = try c.decode(Int.self, forKey: .sessionNumber)
        self.blockType = try c.decode(SessionBlockType.self, forKey: .blockType)
        self.format = try c.decode(PassageFormat.self, forKey: .format)
        self.titleEs = try c.decode(String.self, forKey: .titleEs)
        self.titleEn = try c.decode(String.self, forKey: .titleEn)
        self.episodeSummaryForNext = try c.decode(String.self, forKey: .episodeSummaryForNext)
        self.sentences = try c.decode([PassageSentence].self, forKey: .sentences)
        self.fullText = try c.decode(String.self, forKey: .fullText)
        self.wordCount = try c.decode(Int.self, forKey: .wordCount)
        self.newItemGlosses = try c.decode([NewItemGloss].self, forKey: .newItemGlosses)
        self.probes = try c.decode([PassageProbe].self, forKey: .probes)
        self.grammarNotes = try c.decode([GrammarMicroNote].self, forKey: .grammarNotes)
        self.dueItemIds = try c.decode([String].self, forKey: .dueItemIds)
        self.newItemIds = try c.decode([String].self, forKey: .newItemIds)
        self.grammarItemIds = try c.decode([String].self, forKey: .grammarItemIds)
        self.voice = try c.decode(TutorVoice.self, forKey: .voice)
        self.accent = try c.decode(SpanishRegionalAccent.self, forKey: .accent)
        self.targetWPM = try c.decode(Int.self, forKey: .targetWPM)
        self.isAudioFirst = try c.decode(Bool.self, forKey: .isAudioFirst)
        self.isBootstrap = try c.decode(Bool.self, forKey: .isBootstrap)
        self.rewriteAttempts = try c.decode(Int.self, forKey: .rewriteAttempts)
        self.generatorCallsUsed = (try? c.decode(Int.self, forKey: .generatorCallsUsed)) ?? (self.rewriteAttempts + 1)
        self.isReserveFallback = (try? c.decode(Bool.self, forKey: .isReserveFallback)) ?? false
        self.dueListWasReduced = try c.decode(Bool.self, forKey: .dueListWasReduced)
        self.scheduledWordShare = try c.decode(Double.self, forKey: .scheduledWordShare)
        self.flaggedForNativeAudit = try c.decode(Bool.self, forKey: .flaggedForNativeAudit)
        self.createdAt = try c.decode(Date.self, forKey: .createdAt)
    }
}

public struct ContractViolation: Codable, Sendable, Hashable {
    public let ruleNumber: Int
    public let ruleName: String
    public let detail: String
    public let offendingTokens: [String]
    public let offendingSentenceIndices: [Int]

    public init(
        ruleNumber: Int,
        ruleName: String,
        detail: String,
        offendingTokens: [String],
        offendingSentenceIndices: [Int] = []
    ) {
        self.ruleNumber = ruleNumber
        self.ruleName = ruleName
        self.detail = detail
        self.offendingTokens = offendingTokens
        self.offendingSentenceIndices = offendingSentenceIndices
    }
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

    /// True when the failure stemmed from overcrowding (protocol Step 5: only cut due list by 25% on overcrowding!).
    public var isOvercrowdingFailure: Bool {
        violations.contains { v in
            v.ruleName.localizedCaseInsensitiveContains("Scheduled Share")
                || v.ruleName.localizedCaseInsensitiveContains("Length")
                || v.ruleName.localizedCaseInsensitiveContains("Due Items Coverage")
                || v.ruleName.localizedCaseInsensitiveContains("Fresh Context")
        }
    }

    /// Canonical violation fingerprint for Step 6 ("same violation twice in a row skips straight to fallback").
    public var violationSignature: String {
        violations
            .map { v in
                let toks = v.offendingTokens.prefix(5).sorted().joined(separator: ",")
                return "\(v.ruleNumber):\(v.ruleName):\(toks)"
            }
            .sorted()
            .joined(separator: "|")
    }

    /// Deduplicated 0-based sentence indices needing targeted repair in Step 4.
    public var offendingSentenceIndices: [Int] {
        Array(Set(violations.flatMap(\.offendingSentenceIndices))).sorted()
    }

    public init(
        passed: Bool,
        totalWords: Int,
        closedVocabCompliance: Double,
        scheduledWordShare: Double,
        missingDueItems: [String],
        underSpacedNewItems: [String],
        freshContextCollisions: [String],
        holdoutLeaks: [String],
        offListLemmas: [String],
        stageBPassed: Bool,
        stageBFeedback: String?,
        violations: [ContractViolation]
    ) {
        self.passed = passed
        self.totalWords = totalWords
        self.closedVocabCompliance = closedVocabCompliance
        self.scheduledWordShare = scheduledWordShare
        self.missingDueItems = missingDueItems
        self.underSpacedNewItems = underSpacedNewItems
        self.freshContextCollisions = freshContextCollisions
        self.holdoutLeaks = holdoutLeaks
        self.offListLemmas = offListLemmas
        self.stageBPassed = stageBPassed
        self.stageBFeedback = stageBFeedback
        self.violations = violations
    }
}

// MARK: - Speaking & Writing Output Models

public struct PhonemeAlignmentCell: Identifiable, Codable, Sendable, Hashable {
    public let id: Int
    public let expectedPhone: String
    public let observedPhone: String
    public let distance: Double // 0.0 = exact, <=0.35 = close allophone, >0.35 = mismatch
    public let coachingHint: String?

    public var isExactMatch: Bool { distance < 0.01 }

    public init(
        id: Int,
        expectedPhone: String,
        observedPhone: String,
        distance: Double,
        coachingHint: String?
    ) {
        self.id = id
        self.expectedPhone = expectedPhone
        self.observedPhone = observedPhone
        self.distance = distance
        self.coachingHint = coachingHint
    }
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
    public let partnerReplyEn: String?
    public let spokenSegments: [TutorSpeechSegment]
    public let correctlyUsedItemIds: [String]
    public let failedGrammarPatternIds: [String]
    public let parakeetLatencyMs: Double
    public let zipaLatencyMs: Double

    public init(
        id: String,
        timestamp: Date,
        learnerTranscript: String,
        learnerIPA: String,
        canonicalSpanishIPA: String,
        phonemeAccuracy: Double,
        alignments: [PhonemeAlignmentCell],
        partnerRecastEs: String?,
        partnerReplyEs: String,
        partnerReplyEn: String? = nil,
        spokenSegments: [TutorSpeechSegment] = [],
        correctlyUsedItemIds: [String],
        failedGrammarPatternIds: [String],
        parakeetLatencyMs: Double,
        zipaLatencyMs: Double
    ) {
        self.id = id
        self.timestamp = timestamp
        self.learnerTranscript = learnerTranscript
        self.learnerIPA = learnerIPA
        self.canonicalSpanishIPA = canonicalSpanishIPA
        self.phonemeAccuracy = phonemeAccuracy
        self.alignments = alignments
        self.partnerRecastEs = partnerRecastEs
        self.partnerReplyEs = partnerReplyEs
        self.partnerReplyEn = partnerReplyEn
        self.spokenSegments = spokenSegments
        self.correctlyUsedItemIds = correctlyUsedItemIds
        self.failedGrammarPatternIds = failedGrammarPatternIds
        self.parakeetLatencyMs = parakeetLatencyMs
        self.zipaLatencyMs = zipaLatencyMs
    }
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

    public init(
        id: String,
        promptEs: String,
        learnerText: String,
        recastSentences: [RecastPair],
        correctlyUsedItemIds: [String],
        failedGrammarPatternIds: [String],
        wordCount: Int,
        elapsedSeconds: Double
    ) {
        self.id = id
        self.promptEs = promptEs
        self.learnerText = learnerText
        self.recastSentences = recastSentences
        self.correctlyUsedItemIds = correctlyUsedItemIds
        self.failedGrammarPatternIds = failedGrammarPatternIds
        self.wordCount = wordCount
        self.elapsedSeconds = elapsedSeconds
    }
}

public struct RecastPair: Codable, Sendable, Hashable {
    public let original: String
    public let corrected: String
    public let hadError: Bool
    public let failedPatternId: String?

    public init(original: String, corrected: String, hadError: Bool, failedPatternId: String?) {
        self.original = original
        self.corrected = corrected
        self.hadError = hadError
        self.failedPatternId = failedPatternId
    }
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

    public init(
        id: String,
        sessionNumber: Int,
        week: Int,
        studyMode: StudyMode,
        startedAt: Date,
        endedAt: Date,
        activeInputSeconds: Double,
        activeOutputSeconds: Double,
        passagesCompleted: Int,
        runningWordsProcessed: Int,
        usableEventsCount: Int,
        usableEventsPerInputMinute: Double,
        probesCorrect: Int,
        probesTotal: Int,
        newItemsIntroduced: Int,
        itemsPromotedToKnown: Int,
        itemsPromotedToProductive: Int,
        fatigueRating: Int?,
        enjoymentRating: Int?,
        noOtherSpanishConfirmed: Bool
    ) {
        self.id = id
        self.sessionNumber = sessionNumber
        self.week = week
        self.studyMode = studyMode
        self.startedAt = startedAt
        self.endedAt = endedAt
        self.activeInputSeconds = activeInputSeconds
        self.activeOutputSeconds = activeOutputSeconds
        self.passagesCompleted = passagesCompleted
        self.runningWordsProcessed = runningWordsProcessed
        self.usableEventsCount = usableEventsCount
        self.usableEventsPerInputMinute = usableEventsPerInputMinute
        self.probesCorrect = probesCorrect
        self.probesTotal = probesTotal
        self.newItemsIntroduced = newItemsIntroduced
        self.itemsPromotedToKnown = itemsPromotedToKnown
        self.itemsPromotedToProductive = itemsPromotedToProductive
        self.fatigueRating = fatigueRating
        self.enjoymentRating = enjoymentRating
        self.noOtherSpanishConfirmed = noOtherSpanishConfirmed
    }
}
