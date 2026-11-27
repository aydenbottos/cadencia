import Foundation
#if canImport(NaturalLanguage)
import NaturalLanguage
#endif

public struct AnalyzedToken: Sendable, Hashable {
    public let surface: String
    public let normalized: String
    public let lemma: String
    public let pos: String
    public let isProperNameOrNumber: Bool
    public let isContentWord: Bool
    public let wordIndex: Int
}

/// Fast, deterministic, offline Spanish Lemmatizer, Multi-Word Chunk Matcher, and G2P Phonemizer.
/// Combines:
///  1. 64,397-entry spaCy-generated Spanish surface -> (lemma, POS) table from `cadencia_seed.sqlite`
///  2. Spanish enclitic pronoun stripping (`dámelo` -> `dar`, `levantarse` -> `levantar`, `viéndolo` -> `ver`)
///  3. Regular & morphological suffix fallback + Apple `NaturalLanguage` `NLTagger` disambiguation
///  4. Multi-word chunk phrase detector for the 200 non-compositional Spanish chunks
///  5. Castilian (`es-ES`), Latin American (`es-419`), and Rioplatense (`es-AR`) IPA grapheme-to-phoneme engine
public final class SpanishLemmatizer: Sendable {
    private let inflections: [String: (lemma: String, pos: String)]
    private let chunksByLowerSurface: [String: String] // lower surface -> item_id
    private let maxChunkWordLength: Int

    private static let functionPOS: Set<String> = [
        "DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART", "PUNCT", "NUM"
    ]
    private static let encliticSuffixes: [String] = [
        "melo", "mela", "melos", "melas", "telo", "tela", "telos", "telas",
        "selo", "sela", "selos", "selas", "noslo", "nosla", "noslos", "noslas",
        "me", "te", "se", "nos", "os", "lo", "la", "los", "las", "le", "les"
    ]
    private static let commonSpanishFirstNames: Set<String> = [
        "clara", "mateo", "lucía", "lucia", "tomás", "tomas", "ana", "pedro", "maría", "maria",
        "juan", "carlos", "elena", "diego", "sofía", "sofia", "marta", "pablo", "laura",
        "javier", "carmen", "andrés", "andres", "isabel", "miguel", "david", "sara", "paula",
        "sergio", "jorge", "daniel", "raquel", "roberto", "fernando", "alberto", "luis",
        "navarro", "solís", "solis", "morales", "garcía", "garcia", "lópez", "lopez", "martínez",
        "madrid", "barcelona", "sevilla", "valencia", "oaxaca", "patagonia", "buenos", "aires",
        "españa", "méxico", "mexico", "argentina", "colombia", "chile", "perú", "peru", "bilbao",
        "zaragoza", "málaga", "malaga", "cádiz", "cadiz", "toledo", "granada", "retiro", "atocha"
    ]

    public init(
        inflections: [String: (lemma: String, pos: String)],
        chunkItems: [SyllabusItem] = []
    ) {
        self.inflections = inflections
        var map: [String: String] = [:]
        var maxLen = 1
        for c in chunkItems where c.itemType == .chunk {
            let key = c.spanish.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            map[key] = c.id
            maxLen = max(maxLen, key.split(separator: " ").count)
        }
        self.chunksByLowerSurface = map
        self.maxChunkWordLength = maxLen
    }

    // MARK: - Tokenization & Sentence Splitting

    public func splitSentences(_ text: String) -> [String] {
        var sentences: [String] = []
        var current = ""
        let chars = Array(text)
        for i in 0..<chars.count {
            let ch = chars[i]
            current.append(ch)
            if ch == "." || ch == "!" || ch == "?" {
                let nextIsQuote = (i + 1 < chars.count) && (chars[i + 1] == "»" || chars[i + 1] == "\"" || chars[i + 1] == ")")
                if !nextIsQuote {
                    let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        sentences.append(trimmed)
                    }
                    current = ""
                }
            }
        }
        let rem = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !rem.isEmpty { sentences.append(rem) }
        return sentences
    }

    public func extractWordTokens(_ text: String) -> [String] {
        var tokens: [String] = []
        var current = ""
        for scalar in text.unicodeScalars {
            if CharacterSet.letters.contains(scalar) || CharacterSet.decimalDigits.contains(scalar) {
                current.unicodeScalars.append(scalar)
            } else if !current.isEmpty {
                tokens.append(current)
                current = ""
            }
        }
        if !current.isEmpty { tokens.append(current) }
        return tokens
    }

    // MARK: - Full Morphological Lemmatization

    public func analyze(_ text: String) -> [AnalyzedToken] {
        let sentences = splitSentences(text)
        var results: [AnalyzedToken] = []
        var globalIdx = 0

        #if canImport(NaturalLanguage)
        let nlLemmas = extractNLTaggerHints(for: text)
        #else
        let nlLemmas: [String: (lemma: String, isName: Bool)] = [:]
        #endif

        for sentence in sentences {
            let words = extractWordTokens(sentence)
            for (localIdx, surface) in words.enumerated() {
                let lower = surface.lowercased()
                let isNumber = surface.allSatisfy(\.isNumber)
                let isMidSentenceCapitalized = localIdx > 0 && (surface.first?.isUppercase ?? false)

                let (lemma, pos, knownInDict) = lookupLemma(lower: lower, nlHint: nlLemmas[lower]?.lemma)

                let isProperName = isNumber
                    || Self.commonSpanishFirstNames.contains(lower)
                    || (isMidSentenceCapitalized && !knownInDict)
                    || ((nlLemmas[lower]?.isName ?? false) && !knownInDict)

                let isContent = !isProperName && !Self.functionPOS.contains(pos) && lemma.count >= 3

                results.append(AnalyzedToken(
                    surface: surface,
                    normalized: lower,
                    lemma: lemma,
                    pos: isNumber ? "NUM" : (isProperName ? "PROPN" : pos),
                    isProperNameOrNumber: isProperName,
                    isContentWord: isContent,
                    wordIndex: globalIdx
                ))
                globalIdx += 1
            }
        }
        return results
    }

    public func lookupLemma(lower: String, nlHint: String? = nil) -> (lemma: String, pos: String, found: Bool) {
        if let entry = inflections[lower] {
            return (entry.lemma, entry.pos, true)
        }
        // Contractions al / del
        if lower == "al" { return ("al", "ADP", true) }
        if lower == "del" { return ("del", "ADP", true) }

        // Enclitic pronoun stripping (e.g., dármelo -> dar, sentarse -> sentar, diciéndolo -> decir)
        if lower.count >= 5 {
            for suf in Self.encliticSuffixes where lower.hasSuffix(suf) {
                let base = String(lower.dropLast(suf.count))
                guard base.count >= 3 else { continue }
                let deaccented = Self.stripSpanishAccents(base)
                if let entry = inflections[base] {
                    return (entry.lemma, entry.pos, true)
                }
                if let entry = inflections[deaccented] {
                    return (entry.lemma, entry.pos, true)
                }
                // Handle infinitives/gerunds with accent added due to enclitics (decír + telo -> decir)
                if deaccented.hasSuffix("ar") || deaccented.hasSuffix("er") || deaccented.hasSuffix("ir") {
                    if let entry = inflections[deaccented] {
                        return (entry.lemma, entry.pos, true)
                    }
                    return (deaccented, "VERB", true)
                }
            }
        }

        // Diminutive / superlative normalization (-ito/-ita/-ísimo)
        for suf in ["ísimo", "ísima", "ísimos", "ísimas", "ito", "ita", "itos", "itas", "cito", "cita"] {
            if lower.count > suf.count + 3 && lower.hasSuffix(suf) {
                let stem = String(lower.dropLast(suf.count))
                for ending in ["o", "a", "e", ""] {
                    let cand = stem + ending
                    if let entry = inflections[cand] {
                        return (entry.lemma, entry.pos, true)
                    }
                }
            }
        }

        // Apple NaturalLanguage framework fallback
        if let nl = nlHint, !nl.isEmpty {
            let nlLower = nl.lowercased()
            if let entry = inflections[nlLower] {
                return (entry.lemma, entry.pos, true)
            }
            return (nlLower, "NOUN", false)
        }

        // Simple plural strip fallback
        if lower.hasSuffix("ces") && lower.count > 4 {
            let cand = String(lower.dropLast(3)) + "z"
            if let entry = inflections[cand] { return (entry.lemma, entry.pos, true) }
        }
        if lower.hasSuffix("es") && lower.count > 4 {
            let cand = String(lower.dropLast(2))
            if let entry = inflections[cand] { return (entry.lemma, entry.pos, true) }
        }
        if lower.hasSuffix("s") && lower.count > 3 {
            let cand = String(lower.dropLast(1))
            if let entry = inflections[cand] { return (entry.lemma, entry.pos, true) }
        }

        return (lower, "NOUN", false)
    }

    /// Detects multi-word chunks (from the 200-chunk syllabus) inside tokenized text, returning
    /// `(chunkItemId, startWordIndex, lengthInWords)` matches.
    public func findChunks(in tokens: [AnalyzedToken]) -> [(itemId: String, startIndex: Int, length: Int)] {
        guard !chunksByLowerSurface.isEmpty else { return [] }
        var matches: [(String, Int, Int)] = []
        let n = tokens.count
        for i in 0..<n {
            let maxLen = min(maxChunkWordLength, n - i)
            guard maxLen >= 2 else { continue }
            for len in stride(from: maxLen, through: 2, by: -1) {
                let phrase = tokens[i..<(i + len)].map(\.normalized).joined(separator: " ")
                if let cid = chunksByLowerSurface[phrase] {
                    matches.append((cid, i, len))
                    break
                }
                // Also test lemmatized head verb for verbal chunks (e.g. "di cuenta" -> "dar cuenta" / "darse cuenta")
                let lemmaFirst = tokens[i].lemma + " " + tokens[(i + 1)..<(i + len)].map(\.normalized).joined(separator: " ")
                if let cid = chunksByLowerSurface[lemmaFirst] {
                    matches.append((cid, i, len))
                    break
                }
                let reflexiveFirst = tokens[i].lemma + "se " + tokens[(i + 1)..<(i + len)].map(\.normalized).joined(separator: " ")
                if let cid = chunksByLowerSurface[reflexiveFirst] {
                    matches.append((cid, i, len))
                    break
                }
            }
        }
        return matches
    }

    #if canImport(NaturalLanguage)
    private func extractNLTaggerHints(for text: String) -> [String: (lemma: String, isName: Bool)] {
        let tagger = NLTagger(tagSchemes: [.lemma, .nameType])
        tagger.string = text
        tagger.setLanguage(.spanish, range: text.startIndex..<text.endIndex)
        var hints: [String: (String, Bool)] = [:]
        let options: NLTagger.Options = [.omitPunctuation, .omitWhitespace]
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .lemma, options: options) { tag, range in
            let surf = String(text[range]).lowercased()
            let lem = tag?.rawValue.lowercased() ?? surf
            let nameTag = tagger.tag(at: range.lowerBound, unit: .word, scheme: .nameType).0
            let isName = nameTag == .personalName || nameTag == .placeName || nameTag == .organizationName
            hints[surf] = (lem, isName)
            return true
        }
        return hints
    }
    #endif

    // MARK: - Regional Spanish G2P (Castilian es-ES, Latin American es-419, Rioplatense es-AR)

    public static func stripSpanishAccents(_ input: String) -> String {
        input
            .replacingOccurrences(of: "á", with: "a")
            .replacingOccurrences(of: "é", with: "e")
            .replacingOccurrences(of: "í", with: "i")
            .replacingOccurrences(of: "ó", with: "o")
            .replacingOccurrences(of: "ú", with: "u")
            .replacingOccurrences(of: "ü", with: "u")
    }

    /// Converts Spanish orthography into narrow IPA phones for:
    /// - Regional accent comparison (Rule 8 Castilian / LatAm / Rioplatense IPA), and
    /// - Canonical expected IPA sequence for ZIPA CR Large 300M phoneme alignment!
    public static func spanishToIPA(_ text: String, accent: SpanishRegionalAccent = .castilian) -> String {
        let words = text
            .lowercased()
            .components(separatedBy: CharacterSet.letters.inverted)
            .filter { !$0.isEmpty }
        return words.map { wordToIPA($0, accent: accent) }.joined(separator: " ")
    }

    public static func wordToIPA(_ word: String, accent: SpanishRegionalAccent) -> String {
        let chars = Array(word.lowercased())
        let n = chars.count
        let vowels: Set<Character> = ["a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú", "ü"]
        var out: [String] = []
        var i = 0
        while i < n {
            let ch = chars[i]
            let nxt: Character? = (i + 1 < n) ? chars[i + 1] : nil
            let nxt2: Character? = (i + 2 < n) ? chars[i + 2] : nil
            let prv: Character? = (i > 0) ? chars[i - 1] : nil

            if ch == "c" && nxt == "h" {
                out.append("tʃ")
                i += 2; continue
            }
            if ch == "l" && nxt == "l" {
                switch accent {
                case .castilian: out.append("ʎ")
                case .latinAmerican: out.append("ʝ")
                case .rioplatense: out.append("ʃ")
                }
                i += 2; continue
            }
            if ch == "r" && nxt == "r" {
                out.append("r")
                i += 2; continue
            }
            if ch == "q" && nxt == "u" {
                out.append("k")
                i += 2; continue
            }
            if ch == "g" && nxt == "u", let n2 = nxt2, ["e", "i", "é", "í"].contains(n2) {
                out.append((i == 0 || prv == "n" || prv == "m") ? "ɡ" : "ɣ")
                i += 2; continue
            }
            if ch == "g" && nxt == "ü" {
                out.append((i == 0 || prv == "n" || prv == "m") ? "ɡw" : "ɣw")
                i += 2; continue
            }

            switch ch {
            case "a", "á": out.append("a")
            case "e", "é": out.append("e")
            case "i", "í": out.append("i")
            case "o", "ó": out.append("o")
            case "u", "ú", "ü": out.append("u")
            case "h": break
            case "b", "v":
                out.append((i == 0 || prv == "m" || prv == "n") ? "b" : "β")
            case "c":
                if let nx = nxt, ["e", "i", "é", "í"].contains(nx) {
                    out.append(accent == .castilian ? "θ" : "s")
                } else {
                    out.append("k")
                }
            case "z":
                out.append(accent == .castilian ? "θ" : "s")
            case "d":
                out.append((i == 0 || prv == "l" || prv == "n") ? "d" : "ð")
            case "f": out.append("f")
            case "g":
                if let nx = nxt, ["e", "i", "é", "í"].contains(nx) {
                    out.append("x")
                } else {
                    out.append((i == 0 || prv == "m" || prv == "n") ? "ɡ" : "ɣ")
                }
            case "j": out.append("x")
            case "k": out.append("k")
            case "l": out.append("l")
            case "m": out.append("m")
            case "n":
                if let nx = nxt, ["c", "k", "q", "g"].contains(nx) {
                    out.append("ŋ")
                } else {
                    out.append("n")
                }
            case "ñ": out.append("ɲ")
            case "p": out.append("p")
            case "r":
                if i == 0 || prv == "l" || prv == "n" || prv == "s" {
                    out.append("r")
                } else {
                    out.append("ɾ")
                }
            case "s": out.append("s")
            case "t": out.append("t")
            case "w": out.append("w")
            case "x": out.append(i > 0 ? "ks" : "s")
            case "y":
                if i == n - 1 || (nxt.map { !vowels.contains($0) } ?? true) {
                    out.append("i")
                } else {
                    out.append(accent == .rioplatense ? "ʃ" : "ʝ")
                }
            default: break
            }
            i += 1
        }
        return out.joined()
    }
}
