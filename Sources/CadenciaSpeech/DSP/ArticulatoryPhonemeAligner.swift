import CadenciaCore
import Foundation

/// Weighted Articulatory-Feature IPA Needleman–Wunsch Aligner comparing ZIPA CR Large 300M's
/// raw acoustic IPA transcription against canonical Spanish IPA.
///
/// Gives zero-hallucination, phone-by-phone pronunciation coaching tuned specifically for
/// native English speakers learning Spanish (detecting aspirated stops [pʰ/tʰ/kʰ], English
/// retroflex rhotic [ɹ] vs Spanish tap [ɾ]/trill [r], schwa vowel reduction [ə], palatal [ɲ],
/// velar fricative [x], and spirantized approximants [β, ð, ɣ]).
public struct ArticulatoryPhonemeAligner: Sendable {
    private struct PhoneFeatures: Sendable {
        let isVowel: Bool
        let voiced: Bool
        let place: Int   // 0=labial, 1=dental/alveolar, 2=postalveolar/palatal, 3=velar/uvular, 4=glottal, 5=front, 6=central, 7=back
        let manner: Int  // 0=stop, 1=fricative, 2=affricate, 3=nasal, 4=tap/trill, 5=lateral/approx, 6=highV, 7=midV, 8=lowV
    }

    private static let featureTable: [String: PhoneFeatures] = [
        // Spanish & English Vowels
        "a": .init(isVowel: true, voiced: true, place: 6, manner: 8),
        "ɑ": .init(isVowel: true, voiced: true, place: 7, manner: 8),
        "æ": .init(isVowel: true, voiced: true, place: 5, manner: 8),
        "ɐ": .init(isVowel: true, voiced: true, place: 6, manner: 8),
        "e": .init(isVowel: true, voiced: true, place: 5, manner: 7),
        "ɛ": .init(isVowel: true, voiced: true, place: 5, manner: 7),
        "ə": .init(isVowel: true, voiced: true, place: 6, manner: 7),
        "ʌ": .init(isVowel: true, voiced: true, place: 6, manner: 7),
        "i": .init(isVowel: true, voiced: true, place: 5, manner: 6),
        "ɪ": .init(isVowel: true, voiced: true, place: 5, manner: 6),
        "o": .init(isVowel: true, voiced: true, place: 7, manner: 7),
        "ɔ": .init(isVowel: true, voiced: true, place: 7, manner: 7),
        "u": .init(isVowel: true, voiced: true, place: 7, manner: 6),
        "ʊ": .init(isVowel: true, voiced: true, place: 7, manner: 6),
        // Stops
        "p": .init(isVowel: false, voiced: false, place: 0, manner: 0),
        "b": .init(isVowel: false, voiced: true,  place: 0, manner: 0),
        "t": .init(isVowel: false, voiced: false, place: 1, manner: 0),
        "d": .init(isVowel: false, voiced: true,  place: 1, manner: 0),
        "k": .init(isVowel: false, voiced: false, place: 3, manner: 0),
        "ɡ": .init(isVowel: false, voiced: true,  place: 3, manner: 0),
        "g": .init(isVowel: false, voiced: true,  place: 3, manner: 0),
        // Fricatives & Spanish Approximants
        "f": .init(isVowel: false, voiced: false, place: 0, manner: 1),
        "v": .init(isVowel: false, voiced: true,  place: 0, manner: 1),
        "β": .init(isVowel: false, voiced: true,  place: 0, manner: 1),
        "θ": .init(isVowel: false, voiced: false, place: 1, manner: 1),
        "ð": .init(isVowel: false, voiced: true,  place: 1, manner: 1),
        "s": .init(isVowel: false, voiced: false, place: 1, manner: 1),
        "z": .init(isVowel: false, voiced: true,  place: 1, manner: 1),
        "ʃ": .init(isVowel: false, voiced: false, place: 2, manner: 1),
        "ʒ": .init(isVowel: false, voiced: true,  place: 2, manner: 1),
        "ʝ": .init(isVowel: false, voiced: true,  place: 2, manner: 1),
        "x": .init(isVowel: false, voiced: false, place: 3, manner: 1),
        "χ": .init(isVowel: false, voiced: false, place: 3, manner: 1),
        "ɣ": .init(isVowel: false, voiced: true,  place: 3, manner: 1),
        "h": .init(isVowel: false, voiced: false, place: 4, manner: 1),
        // Affricates
        "tʃ": .init(isVowel: false, voiced: false, place: 2, manner: 2),
        // Nasals
        "m": .init(isVowel: false, voiced: true,  place: 0, manner: 3),
        "n": .init(isVowel: false, voiced: true,  place: 1, manner: 3),
        "ɲ": .init(isVowel: false, voiced: true,  place: 2, manner: 3),
        "ŋ": .init(isVowel: false, voiced: true,  place: 3, manner: 3),
        // Rhotics & Laterals
        "r": .init(isVowel: false, voiced: true,  place: 1, manner: 4),
        "ɾ": .init(isVowel: false, voiced: true,  place: 1, manner: 4),
        "ɹ": .init(isVowel: false, voiced: true,  place: 2, manner: 5),
        "l": .init(isVowel: false, voiced: true,  place: 1, manner: 5),
        "ʎ": .init(isVowel: false, voiced: true,  place: 2, manner: 5),
        "j": .init(isVowel: false, voiced: true,  place: 2, manner: 5),
        "w": .init(isVowel: false, voiced: true,  place: 0, manner: 5),
    ]

    public init() {}

    /// Tokenizes an IPA string into clean phonological segments (preserving affricates `tʃ` and diacritics like `ʰ`).
    public func segmentIPA(_ ipa: String) -> [String] {
        let cleaned = ipa
            .replacingOccurrences(of: "▁", with: "")
            .replacingOccurrences(of: "ˈ", with: "")
            .replacingOccurrences(of: "ˌ", with: "")
            .replacingOccurrences(of: " ", with: "")
        let chars = Array(cleaned)
        var phones: [String] = []
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if c == "t", i + 1 < chars.count, chars[i + 1] == "ʃ" {
                phones.append("tʃ")
                i += 2
                continue
            }
            var seg = String(c)
            // Attach trailing modifier diacritics (aspiration ʰ, length ː, rhoticity ˞)
            while i + 1 < chars.count && ["ʰ", "ʲ", "ʷ", "ː", "˞", "̃", "̪"].contains(chars[i + 1]) {
                i += 1
                seg.append(chars[i])
            }
            phones.append(seg)
            i += 1
        }
        return phones
    }

    /// Computes optimal articulatory alignment and overall pronunciation accuracy in [0.0, 1.0].
    public func align(expectedIPA: String, observedIPA: String) -> (accuracy: Double, cells: [PhonemeAlignmentCell]) {
        let exp = segmentIPA(expectedIPA)
        let obs = segmentIPA(observedIPA)
        guard !exp.isEmpty else { return (1.0, []) }
        guard !obs.isEmpty else {
            let missed = exp.enumerated().map {
                PhonemeAlignmentCell(id: $0.offset, expectedPhone: $0.element, observedPhone: "∅", distance: 1.0, coachingHint: "Sound omitted")
            }
            return (0.0, missed)
        }

        let m = exp.count
        let n = obs.count
        let gapCost = 0.85
        var dp = Array(repeating: Array(repeating: 0.0, count: n + 1), count: m + 1)
        for i in 0...m { dp[i][0] = Double(i) * gapCost }
        for j in 0...n { dp[0][j] = Double(j) * gapCost }

        for i in 1...m {
            for j in 1...n {
                let sub = dp[i - 1][j - 1] + substitutionCost(expected: exp[i - 1], observed: obs[j - 1])
                let del = dp[i - 1][j] + gapCost
                let ins = dp[i][j - 1] + gapCost
                dp[i][j] = min(sub, min(del, ins))
            }
        }

        // Backtrack
        var reversedCells: [(exp: String, obs: String, dist: Double)] = []
        var i = m, j = n
        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                let cost = substitutionCost(expected: exp[i - 1], observed: obs[j - 1])
                if abs(dp[i][j] - (dp[i - 1][j - 1] + cost)) < 1e-6 {
                    reversedCells.append((exp[i - 1], obs[j - 1], cost))
                    i -= 1; j -= 1
                    continue
                }
            }
            if i > 0 && abs(dp[i][j] - (dp[i - 1][j] + gapCost)) < 1e-6 {
                reversedCells.append((exp[i - 1], "∅", gapCost))
                i -= 1
            } else {
                reversedCells.append(("∅", obs[j - 1], gapCost))
                j -= 1
            }
        }

        let ordered = reversedCells.reversed().enumerated().map { idx, pair in
            PhonemeAlignmentCell(
                id: idx,
                expectedPhone: pair.exp,
                observedPhone: pair.obs,
                distance: min(1.0, pair.dist),
                coachingHint: coachingHint(expected: pair.exp, observed: pair.obs, distance: pair.dist)
            )
        }

        let maxPossible = Double(max(m, n))
        let accuracy = max(0.0, min(1.0, 1.0 - (dp[m][n] / maxPossible)))
        return (accuracy, ordered)
    }

    private func substitutionCost(expected: String, observed: String) -> Double {
        if expected == observed { return 0.0 }
        // Accept Castilian/LatAm regional variants cleanly (θ <-> s, ʎ <-> ʝ <-> ʃ)
        if (expected == "θ" && observed == "s") || (expected == "s" && observed == "θ") { return 0.05 }
        if ["ʎ", "ʝ", "ʃ", "ʒ"].contains(expected) && ["ʎ", "ʝ", "ʃ", "ʒ"].contains(observed) { return 0.08 }
        // Accept Spanish b/β, d/ð, g/ɣ allophones with small distance
        if (expected == "β" && observed == "b") || (expected == "b" && observed == "β") { return 0.15 }
        if (expected == "ð" && observed == "d") || (expected == "d" && observed == "ð") { return 0.15 }
        if (expected == "ɣ" && (observed == "ɡ" || observed == "g")) { return 0.15 }
        // English aspiration on voiceless stops [pʰ, tʰ, kʰ]
        if observed.hasSuffix("ʰ") && String(observed.dropLast()) == expected { return 0.30 }

        let baseExp = String(expected.prefix(expected.hasPrefix("tʃ") ? 2 : 1))
        let baseObs = String(observed.prefix(observed.hasPrefix("tʃ") ? 2 : 1))
        guard let f1 = Self.featureTable[baseExp], let f2 = Self.featureTable[baseObs] else {
            return baseExp == baseObs ? 0.15 : 0.90
        }
        if f1.isVowel != f2.isVowel { return 1.0 }
        var dist = 0.0
        if f1.voiced != f2.voiced { dist += 0.28 }
        if f1.place != f2.place { dist += 0.36 }
        if f1.manner != f2.manner { dist += 0.36 }
        return min(1.0, dist)
    }

    private func coachingHint(expected: String, observed: String, distance: Double) -> String? {
        guard distance >= 0.25 else { return nil }
        if observed.hasSuffix("ʰ") {
            return "Unaspirate /\(expected)/ — keep zero puff of air (softer than English)."
        }
        if expected == "r" && (observed == "ɹ" || observed == "ɾ") {
            return "Roll Spanish /r/ with alveolar tongue-tip vibration."
        }
        if expected == "ɾ" && observed == "ɹ" {
            return "Tap /ɾ/ lightly once against the ridge behind upper teeth (like 'tt' in 'butter')."
        }
        if ["a", "e", "i", "o", "u"].contains(expected) && ["ə", "ʌ", "ɪ", "ʊ"].contains(observed) {
            return "Keep Spanish vowel /\(expected)/ crisp and pure—avoid English schwa reduction."
        }
        if expected == "ɲ" && observed == "n" {
            return "Press mid-tongue to the hard palate for /ɲ/ (ñ)."
        }
        if expected == "x" && (observed == "h" || observed == "k") {
            return "Give Spanish j/ge /x/ slightly more gentle velar friction."
        }
        if expected == "β" && observed == "v" {
            return "Keep lips together softly for /β/—don't touch upper teeth to lower lip."
        }
        if expected == "∅" { return "Extra sound /\(observed)/ inserted." }
        if observed == "∅" { return "Missed /\(expected)/." }
        return "Target /\(expected)/, heard /\(observed)/."
    }
}
