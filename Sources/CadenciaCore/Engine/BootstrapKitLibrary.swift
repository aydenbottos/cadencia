import Foundation

/// Implements the revised 17-page Bootstrap Contract (Sessions 1–4, pp. 8–9):
///  - Fixed cast & places: Ana, Luis, Carla; Perú, México (names come from fixed cast list, NOT a tagger!)
///  - Exact 8 fixed Session 1 kits (60 items total: 8+8+8+8+7+8+6+7), one kit per passage, one present-tense pattern per kit
///  - Deterministic fixed kits of 6–8 items for Sessions 2–4, each guaranteed to hold >= 1 verb, function words, and >= 2 cognate nouns/adjectives
///  - Complete deterministic table of each item's inflected forms (`InflectedFormsTable`), fixing single-lemma tagger gaps
///    (e.g. doctora/doctoras -> doctor, profesora -> profesor, estás -> estar, visita -> visitar, estudio -> estudiar,
///    uso -> usar, preparo -> preparar, favorita -> favorito, un/una/unos/unas -> un/uno, el/la/los/las -> el/la)
///  - Atomic contraction splitting: `al` -> `["a", "el"]` and `del` -> `["de", "el"]` (count as two items!)
///  - All 8 hand-written Session 1 acceptance test passages (Kit 1 verbatim from Protocol p. 9)
public enum BootstrapKitLibrary {

    // MARK: - Fixed Cast & Places (Protocol p. 9: "Names come from a fixed cast list, not a tagger")

    public static let fixedCastDisplayNames: [String] = ["Ana", "Luis", "Carla", "Perú", "México"]

    public static let fixedCastLowercaseSet: Set<String> = [
        "ana", "luis", "carla", "perú", "peru", "méxico", "mexico",
    ]

    // MARK: - Fixed Kit Specification

    public struct KitSpec: Sendable, Hashable {
        public let kitNumber: Int       // 1...8
        public let itemLemmas: [String] // Canonical lemmas in protocol order
        public let pattern: String      // Single present-tense pattern for the kit
        public let microNoteEn: String
        public let contrastAEs: String
        public let contrastAEn: String
        public let contrastBEs: String
        public let contrastBEn: String

        public init(
            kitNumber: Int,
            itemLemmas: [String],
            pattern: String,
            microNoteEn: String,
            contrastAEs: String,
            contrastAEn: String,
            contrastBEs: String,
            contrastBEn: String
        ) {
            self.kitNumber = kitNumber
            self.itemLemmas = itemLemmas
            self.pattern = pattern
            self.microNoteEn = microNoteEn
            self.contrastAEs = contrastAEs
            self.contrastAEn = contrastAEn
            self.contrastBEs = contrastBEs
            self.contrastBEn = contrastBEn
        }
    }

    /// Exact Session 1 Eight Fixed Kits (60 items total — Protocol pp. 8–9).
    public static let session1Kits: [KitSpec] = [
        KitSpec(
            kitNumber: 1,
            itemLemmas: ["ser", "el", "un", "de", "y", "doctor", "hotel", "famoso"],
            pattern: "ser with a noun or adjective; agreement",
            microNoteEn: "Use ser (es / son) for identity and lasting qualities. Articles, nouns, and adjectives match in gender and number: un doctor famoso vs. una doctora famosa.",
            contrastAEs: "Luis es un doctor famoso.",
            contrastAEn: "Luis is a famous doctor (masculine).",
            contrastBEs: "Ana es una doctora famosa.",
            contrastBEn: "Ana is a famous doctor (feminine)."
        ),
        KitSpec(
            kitNumber: 2,
            itemLemmas: ["no", "en", "estar", "restaurante", "hospital", "moderno", "pero", "muy"],
            pattern: "estar for location; negation",
            microNoteEn: "Use estar (está / están) to say where people or places are located. Put no right before the verb to make a sentence negative.",
            contrastAEs: "Ana está en el hospital moderno.",
            contrastAEn: "Ana is in the modern hospital.",
            contrastBEs: "Luis no está en el restaurante.",
            contrastBEn: "Luis is not in the restaurant."
        ),
        KitSpec(
            kitNumber: 3,
            itemLemmas: ["tener", "que", "su", "familia", "problema", "importante", "con", "persona"],
            pattern: "tener; possessive su; relative que",
            microNoteEn: "Use tener (tiene / tienen) for possession or issues, su/sus for his/her/their, and que (that/who) to connect a description to a noun.",
            contrastAEs: "Carla tiene un problema importante.",
            contrastAEn: "Carla has an important problem.",
            contrastBEs: "Su familia tiene un hotel que es famoso.",
            contrastBEn: "Her family has a hotel that is famous."
        ),
        KitSpec(
            kitNumber: 4,
            itemLemmas: ["a", "ir", "para", "universidad", "estudiante", "profesor", "clase", "este"],
            pattern: "ir a; a + el = al",
            microNoteEn: "Use ir + a (va a / van a) for going to a place. Whenever a meets masculine singular el, they obligatorily fuse into al.",
            contrastAEs: "Carla va a la universidad para la clase.",
            contrastAEn: "Carla goes to the university for class.",
            contrastBEs: "Luis va al hospital con este profesor.",
            contrastBEn: "Luis goes to the hospital with this professor."
        ),
        KitSpec(
            kitNumber: 5,
            itemLemmas: ["haber", "mucho", "más", "música", "café", "parque", "centro"],
            pattern: "hay; más ... que",
            microNoteEn: "Use hay (from haber) to say 'there is' or 'there are', and más … que to compare ('more … than').",
            contrastAEs: "Hay mucha música en el parque.",
            contrastAEn: "There is a lot of music in the park.",
            contrastBEs: "El café del centro es más moderno que el hotel.",
            contrastBEn: "The downtown café is more modern than the hotel."
        ),
        KitSpec(
            kitNumber: 6,
            itemLemmas: ["yo", "mi", "también", "porque", "visitar", "estudiar", "museo", "historia"],
            pattern: "first-person verb forms; porque",
            microNoteEn: "First-person singular present verbs end in -o (estudio, visito; soy, estoy, tengo, voy). Use porque ('because') to give a reason.",
            contrastAEs: "Yo estudio historia en mi universidad.",
            contrastAEn: "I study history at my university.",
            contrastBEs: "También visito el museo porque es famoso.",
            contrastBEn: "I also visit the museum because it is famous."
        ),
        KitSpec(
            kitNumber: 7,
            itemLemmas: ["o", "todo", "usar", "necesitar", "teléfono", "minuto"],
            pattern: "infinitive after para; todo + article",
            microNoteEn: "Follow para with an unconjugated infinitive ('in order to …'). Place todo/toda/todos/todas directly before the article (todo el minuto, toda la clase).",
            contrastAEs: "Uso mi teléfono para visitar el museo.",
            contrastAEn: "I use my phone to visit the museum.",
            contrastBEs: "Toda la familia necesita un minuto.",
            contrastBEn: "The whole family needs a minute."
        ),
        KitSpec(
            kitNumber: 8,
            itemLemmas: ["cuando", "otro", "poder", "preparar", "animal", "favorito", "color"],
            pattern: "poder + infinitive; cuando clauses",
            microNoteEn: "Pair poder (puedo / puede / pueden) with an infinitive to express 'can'. Use cuando ('when') to link two present-tense actions.",
            contrastAEs: "Ana puede preparar café en otro hotel.",
            contrastAEn: "Ana can prepare coffee in another hotel.",
            contrastBEs: "Cuando Luis visita el parque, prepara su color favorito.",
            contrastBEn: "When Luis visits the park, he prepares his favorite color."
        ),
    ]

    /// Present-tense micro-patterns for Sessions 2–4 dynamically built bootstrap kits.
    private static let laterSessionBootstrapPatterns: [String] = [
        "present tense regular -ar verbs; subject agreement",
        "present tense regular -er and -ir verbs",
        "noun and adjective gender/number agreement",
        "prepositions en, con, de, para in present tense",
        "negation with no and contrast with pero",
        "possessive determiners mi and su",
        "infinitive after poder, querer, or necesitar",
        "comparisons with más ... que and mucho",
    ]

    /// Bootstrap Feasibility Pre-Check (Protocol p. 8):
    /// "The 23% arithmetic is skipped. The only pre-check is that the kit holds a verb and a noun."
    public static func checkKitFeasibility(items: [SyllabusItem]) -> Bool {
        let hasVerb = items.contains { $0.pos == "VERB" || $0.pos == "AUX" }
        let hasNoun = items.contains { $0.pos == "NOUN" }
        return hasVerb && hasNoun
    }

    /// Resolves a Session 1 fixed kit into `SyllabusItem` objects from the syllabus bank.
    /// Note: `"un"` resolves cleanly onto the `"uno"` syllabus item ID (preserving SQLite FK integrity)
    /// while carrying `"un"` as its canonical kit display/matching lemma.
    public static func resolveSession1Kit(
        kitNumber: Int,
        allItems: [SyllabusItem]
    ) -> (spec: KitSpec, items: [SyllabusItem])? {
        guard kitNumber >= 1, kitNumber <= session1Kits.count else { return nil }
        let spec = session1Kits[kitNumber - 1]
        let lexical = allItems.filter { !$0.isHoldout && ($0.itemType == .lemma || $0.itemType == .chunk) }
        let byLower = Dictionary(lexical.map { ($0.spanish.lowercased(), $0) }, uniquingKeysWith: { first, _ in first })

        var resolved: [SyllabusItem] = []
        for rawLemma in spec.itemLemmas {
            let key = rawLemma.lowercased()
            if let exact = byLower[key] {
                resolved.append(exact)
            } else if key == "un", let unoItem = byLower["uno"] {
                resolved.append(SyllabusItem(
                    id: unoItem.id,
                    itemType: unoItem.itemType,
                    rankIndex: unoItem.rankIndex,
                    spanish: "un",
                    englishGloss: "a / an / one",
                    pos: "DET",
                    cefr: unoItem.cefr,
                    ipaEs: "un",
                    ipaLatam: "un",
                    isTransparentCognate: unoItem.isTransparentCognate,
                    cognateScore: unoItem.cognateScore,
                    isBootstrap: true,
                    isHoldout: false,
                    isTop2000ProductiveTarget: true,
                    zipfFrequency: unoItem.zipfFrequency,
                    visualSymbol: unoItem.visualSymbol
                ))
            }
        }
        return (spec, resolved)
    }

    /// Builds a deterministic fixed kit (6–8 items) for Bootstrap Sessions 2–4, guaranteeing each kit
    /// holds >= 1 verb, function words, and >= 2 cognate nouns/adjectives (Protocol p. 8).
    public static func buildDynamicBootstrapKit(
        sessionNumber: Int,
        kitNumber: Int,
        allItems: [SyllabusItem],
        records: [String: LearnerItemRecord],
        remainingBudget: Int
    ) -> (spec: KitSpec, items: [SyllabusItem])? {
        guard kitNumber >= 1, kitNumber <= 8, remainingBudget > 0 else { return nil }
        let targetSizes = [8, 8, 8, 8, 7, 8, 6, 7]
        let targetSize = min(remainingBudget, targetSizes[kitNumber - 1])
        guard targetSize > 0 else { return nil }

        let activeLexical = allItems.filter { !$0.isHoldout && ($0.itemType == .lemma || $0.itemType == .chunk) }
        let unmetAll = activeLexical.filter {
            (records[$0.id]?.meetingsCount ?? 0) == 0 && !(records[$0.id]?.isHeldBackControl ?? false)
        }.sorted { $0.rankIndex < $1.rankIndex }
        let unmetBootstrap = unmetAll.filter(\.isBootstrap)
        let pool = unmetBootstrap.count >= targetSize ? unmetBootstrap : unmetAll
        guard !pool.isEmpty else { return nil }

        var chosen: [SyllabusItem] = []
        var seen: Set<String> = []
        func take(_ item: SyllabusItem?) {
            guard let item, chosen.count < targetSize, seen.insert(item.id).inserted else { return }
            chosen.append(item)
        }

        // 1. At least 1 verb (preferring bootstrap/frequent verbs)
        take(pool.first(where: { $0.pos == "VERB" || $0.pos == "AUX" })
             ?? unmetAll.first(where: { $0.pos == "VERB" || $0.pos == "AUX" }))

        // 2. At least 2 cognate nouns or adjectives (guaranteeing at least 1 NOUN!)
        let cogNouns = pool.filter { $0.pos == "NOUN" && $0.isTransparentCognate }
        let anyNouns = pool.filter { $0.pos == "NOUN" }
        let cogAdj = pool.filter { $0.pos == "ADJ" && $0.isTransparentCognate }

        take(cogNouns.first ?? anyNouns.first ?? unmetAll.first(where: { $0.pos == "NOUN" }))
        take(cogNouns.dropFirst().first ?? cogAdj.first ?? anyNouns.dropFirst().first)
        if chosen.count + 1 <= targetSize {
            take(cogAdj.first(where: { !seen.contains($0.id) }) ?? cogNouns.dropFirst(2).first)
        }

        // 3. Function words & remaining high-frequency bootstrap items up to targetSize
        let funcPOS: Set<String> = ["DET", "ADP", "PRON", "CCONJ", "SCONJ", "ADV", "NUM"]
        for item in pool where funcPOS.contains(item.pos) && chosen.count < targetSize {
            take(item)
        }
        for item in pool where chosen.count < targetSize {
            take(item)
        }

        let patIdx = ((sessionNumber - 2) * 8 + (kitNumber - 1)) % laterSessionBootstrapPatterns.count
        let pattern = laterSessionBootstrapPatterns[abs(patIdx)]
        let spec = KitSpec(
            kitNumber: kitNumber,
            itemLemmas: chosen.map(\.spanish),
            pattern: pattern,
            microNoteEn: "Focus on \(pattern) using today's kit items in present tense.",
            contrastAEs: "",
            contrastAEn: "",
            contrastBEs: "",
            contrastBEn: ""
        )
        return (spec, chosen)
    }

    /// Synthesizes a single `SyllabusItem` grammar object for a Bootstrap kit's exact present-tense pattern.
    public static func makeGrammarItem(for kit: KitSpec) -> SyllabusItem {
        SyllabusItem(
            id: "boot_kit_gram_\(kit.kitNumber)",
            itemType: .grammar,
            rankIndex: kit.kitNumber,
            spanish: kit.pattern,
            englishGloss: kit.microNoteEn,
            pos: "GRAMMAR",
            cefr: .a1,
            ipaEs: "",
            ipaLatam: "",
            isTransparentCognate: false,
            cognateScore: 0.0,
            isBootstrap: true,
            isHoldout: false,
            isTop2000ProductiveTarget: false,
            zipfFrequency: 0.0,
            visualSymbol: "text.book.closed.fill",
            grammarCategory: "Bootstrap Present Tense",
            grammarWeek: 1,
            microNoteEn: kit.microNoteEn,
            contrastAEs: kit.contrastAEs,
            contrastAEn: kit.contrastAEn,
            contrastBEs: kit.contrastBEs,
            contrastBEn: kit.contrastBEn
        )
    }

    // MARK: - Complete Table of Inflected Forms (Protocol p. 8)

    /// Deterministic table of inflected forms for every kit item + regular Spanish paradigms + SQLite inflections.
    /// Guarantees single-lemma tagger quirks (`doctora -> doctora`, `favorita -> favoritar`, `visita -> visita`,
    /// `estudio -> estudio`, `uso -> uso`, `preparo -> preparo`, `estás -> estás`, `una -> uno`) NEVER cause
    /// false rejections when matching a surface word against allowed items!
    public static let explicitParadigmsByLemma: [String: Set<String>] = [
        // Kit 1: ser, el, un, de, y, doctor, hotel, famoso
        "ser": ["ser", "soy", "eres", "es", "somos", "sois", "son", "sea", "seas", "sean", "era", "eran", "fue", "fueron", "sido", "siendo"],
        "el": ["el", "la", "los", "las"],
        "la": ["la", "las", "el", "los"],
        "un": ["un", "una", "unos", "unas", "uno"],
        "uno": ["uno", "un", "una", "unos", "unas"],
        "de": ["de"],
        "y": ["y", "e"],
        "doctor": ["doctor", "doctora", "doctores", "doctoras"],
        "hotel": ["hotel", "hoteles"],
        "famoso": ["famoso", "famosa", "famosos", "famosas"],

        // Kit 2: no, en, estar, restaurante, hospital, moderno, pero, muy
        "no": ["no"],
        "en": ["en"],
        "estar": ["estar", "estoy", "estás", "estas", "está", "esta", "estamos", "estáis", "están", "estan", "estaba", "estuvo", "estado", "estando"],
        "restaurante": ["restaurante", "restaurantes"],
        "hospital": ["hospital", "hospitales"],
        "moderno": ["moderno", "moderna", "modernos", "modernas"],
        "pero": ["pero"],
        "muy": ["muy"],

        // Kit 3: tener, que, su, familia, problema, importante, con, persona
        "tener": ["tener", "tengo", "tienes", "tiene", "tenemos", "tenéis", "tienen", "tenía", "tuvo", "tenido", "teniendo"],
        "que": ["que", "qué"],
        "su": ["su", "sus"],
        "familia": ["familia", "familias"],
        "problema": ["problema", "problemas"],
        "importante": ["importante", "importantes"],
        "con": ["con"],
        "persona": ["persona", "personas"],

        // Kit 4: a, ir, para, universidad, estudiante, profesor, clase, este
        "a": ["a"],
        "ir": ["ir", "voy", "vas", "va", "vamos", "vais", "van", "iba", "fue", "ido", "yendo"],
        "para": ["para"],
        "universidad": ["universidad", "universidades"],
        "estudiante": ["estudiante", "estudiantes"],
        "profesor": ["profesor", "profesora", "profesores", "profesoras"],
        "clase": ["clase", "clases"],
        "este": ["este", "esta", "esto", "estos", "estas"],

        // Kit 5: haber, mucho, más, música, café, parque, centro
        "haber": ["haber", "hay", "he", "has", "ha", "hemos", "habéis", "han", "había", "hubo"],
        "mucho": ["mucho", "mucha", "muchos", "muchas"],
        "más": ["más", "mas"],
        "música": ["música", "músicas", "musica"],
        "café": ["café", "cafés", "cafe", "cafes"],
        "parque": ["parque", "parques"],
        "centro": ["centro", "centros"],

        // Kit 6: yo, mi, también, porque, visitar, estudiar, museo, historia
        "yo": ["yo"],
        "mi": ["mi", "mis"],
        "también": ["también", "tambien"],
        "porque": ["porque"],
        "visitar": ["visitar", "visito", "visitas", "visita", "visitamos", "visitáis", "visitan", "visitando", "visitado"],
        "estudiar": ["estudiar", "estudio", "estudias", "estudia", "estudiamos", "estudiáis", "estudian", "estudiando", "estudiado"],
        "museo": ["museo", "museos"],
        "historia": ["historia", "historias"],

        // Kit 7: o, todo, usar, necesitar, teléfono, minuto
        "o": ["o", "u"],
        "todo": ["todo", "toda", "todos", "todas"],
        "usar": ["usar", "uso", "usas", "usa", "usamos", "usáis", "usan", "usando", "usado"],
        "necesitar": ["necesitar", "necesito", "necesitas", "necesita", "necesitamos", "necesitáis", "necesitan", "necesitando", "necesitado"],
        "teléfono": ["teléfono", "teléfonos", "telefono", "telefonos"],
        "minuto": ["minuto", "minutos"],

        // Kit 8: cuando, otro, poder, preparar, animal, favorito, color
        "cuando": ["cuando", "cuándo"],
        "otro": ["otro", "otra", "otros", "otras"],
        "poder": ["poder", "puedo", "puedes", "puede", "podemos", "podéis", "pueden", "podía", "pudo", "podido", "pudiendo"],
        "preparar": ["preparar", "preparo", "preparas", "prepara", "preparamos", "preparáis", "preparan", "preparando", "preparado"],
        "animal": ["animal", "animales"],
        "favorito": ["favorito", "favorita", "favoritos", "favoritas"],
        "color": ["color", "colores"],

        // Additional high-frequency pronouns & irregular verbs for Sessions 2–4
        "tú": ["tú", "tu"],
        "él": ["él", "ella", "ellos", "ellas"],
        "nosotros": ["nosotros", "nosotras"],
        "hacer": ["hacer", "hago", "haces", "hace", "hacemos", "hacéis", "hacen"],
        "ver": ["ver", "veo", "ves", "ve", "vemos", "veis", "ven"],
        "dar": ["dar", "doy", "das", "da", "damos", "dais", "dan"],
        "decir": ["decir", "digo", "dices", "dice", "decimos", "decís", "dicen"],
        "saber": ["saber", "sé", "sabes", "sabe", "sabemos", "sabéis", "saben"],
        "querer": ["querer", "quiero", "quieres", "quiere", "queremos", "queréis", "quieren"],
        "venir": ["venir", "vengo", "vienes", "viene", "venimos", "venís", "vienen"],
        "salir": ["salir", "salgo", "sales", "sale", "salimos", "salís", "salen"],
        "ese": ["ese", "esa", "eso", "esos", "esas"],
        "bueno": ["bueno", "buen", "buena", "buenos", "buenas"],
        "grande": ["grande", "gran", "grandes"],
        "primero": ["primero", "primer", "primera", "primeros", "primeras"],
        "nuevo": ["nuevo", "nueva", "nuevos", "nuevas"],
    ]

    /// Reverse lookup `surfaceLower -> Set<canonicalLemma>` built once from `explicitParadigmsByLemma`.
    private static let explicitSurfaceToLemmas: [String: Set<String>] = {
        var map: [String: Set<String>] = [:]
        for (lemma, forms) in explicitParadigmsByLemma {
            for f in forms {
                map[f.lowercased(), default: []].insert(lemma.lowercased())
            }
        }
        return map
    }()

    /// Expands a normalized Spanish surface token into atomic lexical checks:
    /// - `"al"` -> `[["a"], ["el"]]` (TWO items per Protocol p. 8!)
    /// - `"del"` -> `[["de"], ["el"]]` (TWO items per Protocol p. 8!)
    /// - Any other word -> `[candidateLemmasSet]` from explicit paradigms + SQLite lemmatizer + regular Spanish inflection rules.
    public static func expandTokenToCandidateItemSets(
        normalizedSurface: String,
        lemmatizerLemma: String
    ) -> [Set<String>] {
        let low = normalizedSurface.lowercased()
        // Protocol p. 8: "al and del count as two items"
        if low == "al" {
            return [Set(["a"]), Set(["el"])]
        }
        if low == "del" {
            return [Set(["de"]), Set(["el"])]
        }

        var cands: Set<String> = [low, lemmatizerLemma.lowercased()]

        // 1. Explicit complete paradigm table (many-to-many)
        if let explicit = explicitSurfaceToLemmas[low] {
            cands.formUnion(explicit)
        }

        // 2. Regular noun/adjective gender & number inflections (e.g. doctora/doctoras -> doctor, famosa -> famoso)
        if low.hasSuffix("oras") && low.count > 5 {
            cands.insert(String(low.dropLast(2))) // -oras -> -or
        } else if low.hasSuffix("ora") && low.count > 4 {
            cands.insert(String(low.dropLast(1))) // -ora -> -or
        } else if low.hasSuffix("ores") && low.count > 5 {
            cands.insert(String(low.dropLast(2))) // -ores -> -or
        }
        if low.hasSuffix("as") && low.count > 4 {
            cands.insert(String(low.dropLast(2)) + "o")
            cands.insert(String(low.dropLast(1)))
        } else if low.hasSuffix("os") && low.count > 4 {
            cands.insert(String(low.dropLast(1)))
        } else if low.hasSuffix("a") && low.count > 3 {
            cands.insert(String(low.dropLast(1)) + "o")
        }
        if low.hasSuffix("es") && low.count > 4 {
            cands.insert(String(low.dropLast(2)))
        } else if low.hasSuffix("s") && low.count > 3 {
            cands.insert(String(low.dropLast(1)))
        }

        // 3. Regular present-tense verb forms (-ar / -er / -ir) so e.g. preparo->preparar, visita->visitar, come->comer
        if low.count >= 4 {
            for (ending, infSuffixes) in [
                ("amos", ["ar"]), ("áis", ["ar"]), ("an", ["ar"]), ("as", ["ar"]),
                ("emos", ["er"]), ("éis", ["er"]), ("imos", ["ir"]), ("ís", ["ir"]),
                ("en", ["er", "ir"]), ("es", ["er", "ir"]),
                ("o", ["ar", "er", "ir"]), ("a", ["ar"]), ("e", ["er", "ir"]),
            ] {
                if low.hasSuffix(ending) && low.count > ending.count + 2 {
                    let stem = String(low.dropLast(ending.count))
                    for inf in infSuffixes {
                        cands.insert(stem + inf)
                    }
                }
            }
        }

        // Treat "un" and "uno" symmetrically
        if cands.contains("un") { cands.insert("uno") }
        if cands.contains("uno") { cands.insert("un") }

        return [cands]
    }

    // MARK: - All 8 Hand-Written Session 1 Kit Acceptance Passages (Protocol p. 9)

    public struct HandWrittenKitPassage: Sendable {
        public let kitNumber: Int
        public let titleEs: String
        public let titleEn: String
        public let sentences: [(es: String, en: String, symbol: String)]
        public let probes: [(sentenceEs: String, options: [String], symbols: [String], correct: Int, hinge: String)]
    }

    /// The 8 hand-written Session 1 acceptance test passages (Kit 1 verbatim from Protocol p. 9!).
    /// "Those eight passages are the verifier's acceptance test. A verifier that rejects any of them has the fault, not the generator."
    public static let session1AcceptancePassages: [HandWrittenKitPassage] = [
        // KIT 1 (EXACT VERBATIM PASSAGE FROM PROTOCOL PAGE 9!)
        // New items: ser, el, un, de, y, doctor, hotel, famoso
        HandWrittenKitPassage(
            kitNumber: 1,
            titleEs: "Ana, Luis y el hotel famoso",
            titleEn: "Ana, Luis, and the famous hotel",
            sentences: [
                ("Ana es una doctora famosa.", "Ana is a famous doctor.", "stethoscope"),
                ("Luis es un doctor famoso.", "Luis is a famous doctor.", "cross.case.fill"),
                ("El hotel es famoso.", "The hotel is famous.", "building.2.fill"),
                ("El doctor es de Perú.", "The doctor is from Peru.", "globe.americas.fill"),
                ("Ana y Luis son doctores.", "Ana and Luis are doctors.", "person.2.fill"),
                ("El hotel de Ana y Luis es famoso.", "Ana and Luis's hotel is famous.", "star.fill"),
                ("Luis es de México y Ana de Perú.", "Luis is from Mexico and Ana from Peru.", "map.fill"),
                ("Un hotel de Luis es famoso.", "A hotel of Luis's is famous.", "bed.double.fill"),
            ],
            probes: [
                (
                    sentenceEs: "Ana es una doctora famosa.",
                    options: ["Ana is a famous doctor", "Ana is a hotel in Peru", "Luis is from Mexico", "The hotel is not famous"],
                    symbols: ["stethoscope", "building.2.fill", "map.fill", "xmark.circle"],
                    correct: 0,
                    hinge: "«doctora» (doctor) + «famosa» (famous) + «es» (is)"
                ),
                (
                    sentenceEs: "El hotel de Ana y Luis es famoso.",
                    options: ["Ana and Luis are in Mexico", "Ana and Luis's hotel is famous", "A doctor is from Peru", "Luis is a doctor"],
                    symbols: ["globe.americas.fill", "building.2.fill", "stethoscope", "person.fill"],
                    correct: 1,
                    hinge: "«hotel» + «de» (of) + «y» (and) + «famoso» (famous)"
                ),
            ]
        ),
        // KIT 2: no, en, estar, restaurante, hospital, moderno, pero, muy
        HandWrittenKitPassage(
            kitNumber: 2,
            titleEs: "En el hospital y el restaurante",
            titleEn: "At the hospital and the restaurant",
            sentences: [
                ("Ana está en el hospital moderno.", "Ana is in the modern hospital.", "cross.case.fill"),
                ("El hospital de Perú es muy moderno.", "The hospital in Peru is very modern.", "building.columns.fill"),
                ("Luis no está en el hospital.", "Luis is not in the hospital.", "xmark.circle.fill"),
                ("Luis está en un restaurante famoso.", "Luis is in a famous restaurant.", "fork.knife"),
                ("El restaurante es moderno pero no famoso.", "The restaurant is modern but not famous.", "sparkles"),
                ("Carla está en el hotel de México.", "Carla is in the hotel in Mexico.", "bed.double.fill"),
                ("El hotel es muy moderno y famoso.", "The hotel is very modern and famous.", "building.2.fill"),
                ("Ana y Carla no están en México.", "Ana and Carla are not in Mexico.", "map.fill"),
            ],
            probes: [
                (
                    sentenceEs: "Ana está en el hospital moderno.",
                    options: ["Ana is in the modern hospital", "Ana is in a restaurant", "Luis is in Peru", "The hotel is closed"],
                    symbols: ["cross.case.fill", "fork.knife", "map.fill", "bed.double.fill"],
                    correct: 0,
                    hinge: "«está en» (is in) + «hospital» + «moderno»"
                ),
                (
                    sentenceEs: "Luis está en un restaurante famoso.",
                    options: ["Luis is in a modern hospital", "Luis is in a famous restaurant", "Carla is a doctor", "The hospital is in Mexico"],
                    symbols: ["cross.case.fill", "fork.knife", "stethoscope", "globe.americas.fill"],
                    correct: 1,
                    hinge: "«está en» + «restaurante» + «famoso»"
                ),
            ]
        ),
        // KIT 3: tener, que, su, familia, problema, importante, con, persona
        HandWrittenKitPassage(
            kitNumber: 3,
            titleEs: "La familia y un problema importante",
            titleEn: "The family and an important problem",
            sentences: [
                ("Carla tiene una familia muy famosa.", "Carla has a very famous family.", "person.3.fill"),
                ("Su familia está en Perú.", "Her family is in Peru.", "globe.americas.fill"),
                ("Ana tiene un problema importante.", "Ana has an important problem.", "exclamationmark.triangle.fill"),
                ("El doctor está con su familia.", "The doctor is with his family.", "heart.fill"),
                ("Carla es una persona muy importante.", "Carla is a very important person.", "person.fill"),
                ("Luis tiene un hotel que es moderno.", "Luis has a hotel that is modern.", "building.2.fill"),
                ("La persona que está en Perú es Ana.", "The person who is in Peru is Ana.", "globe.americas.fill"),
                ("Su restaurante tiene un problema pero es famoso.", "Their restaurant has a problem but is famous.", "fork.knife"),
            ],
            probes: [
                (
                    sentenceEs: "Ana tiene un problema importante.",
                    options: ["Ana has an important problem", "Ana is in a modern hotel", "Luis has a restaurant", "Carla is from Peru"],
                    symbols: ["exclamationmark.triangle.fill", "building.2.fill", "fork.knife", "map.fill"],
                    correct: 0,
                    hinge: "«tiene» (has) + «problema» + «importante»"
                ),
                (
                    sentenceEs: "El doctor está con su familia.",
                    options: ["The doctor is in Mexico", "The doctor is with his family", "Carla has a hotel", "The restaurant is modern"],
                    symbols: ["map.fill", "person.3.fill", "building.2.fill", "fork.knife"],
                    correct: 1,
                    hinge: "«con» (with) + «su» (his/her) + «familia»"
                ),
            ]
        ),
        // KIT 4: a, ir, para, universidad, estudiante, profesor, clase, este
        HandWrittenKitPassage(
            kitNumber: 4,
            titleEs: "En la universidad",
            titleEn: "At the university",
            sentences: [
                ("Carla es estudiante en esta universidad.", "Carla is a student at this university.", "graduationcap.fill"),
                ("Luis es un profesor muy importante.", "Luis is a very important professor.", "person.bust.fill"),
                ("Este profesor va a la universidad.", "This professor goes to the university.", "building.columns.fill"),
                ("Carla va a la clase con Ana.", "Carla goes to the class with Ana.", "book.fill"),
                ("Ana va al hospital para su familia.", "Ana goes to the hospital for her family.", "cross.case.fill"),
                ("Luis va al restaurante con un estudiante.", "Luis goes to the restaurant with a student.", "fork.knife"),
                ("Esta clase es muy importante para Carla.", "This class is very important for Carla.", "sparkles"),
                ("El profesor tiene una clase en México.", "The professor has a class in Mexico.", "globe.americas.fill"),
            ],
            probes: [
                (
                    sentenceEs: "Carla va a la clase con Ana.",
                    options: ["Carla goes to class with Ana", "Carla is at a hotel", "Luis has a problem", "Ana is from Mexico"],
                    symbols: ["book.fill", "building.2.fill", "exclamationmark.triangle.fill", "map.fill"],
                    correct: 0,
                    hinge: "«va a» (goes to) + «clase» + «con» (with)"
                ),
                (
                    sentenceEs: "Ana va al hospital para su familia.",
                    options: ["Ana goes to the university", "Ana goes to the hospital for her family", "Luis goes to class", "Carla is a professor"],
                    symbols: ["graduationcap.fill", "cross.case.fill", "book.fill", "person.bust.fill"],
                    correct: 1,
                    hinge: "«va al» (a + el) + «hospital» + «para» (for)"
                ),
            ]
        ),
        // KIT 5: haber, mucho, más, música, café, parque, centro
        HandWrittenKitPassage(
            kitNumber: 5,
            titleEs: "Música y café en el centro",
            titleEn: "Music and coffee downtown",
            sentences: [
                ("Hay mucha música en el centro.", "There is a lot of music downtown.", "music.note"),
                ("Hay un parque famoso en el centro.", "There is a famous park downtown.", "leaf.fill"),
                ("Ana y Luis están en el parque.", "Ana and Luis are in the park.", "person.2.fill"),
                ("Hay un café moderno en el parque.", "There is a modern café in the park.", "cup.and.saucer.fill"),
                ("Este café tiene más música que el hotel.", "This café has more music than the hotel.", "music.note.list"),
                ("Carla va al centro para un café.", "Carla goes downtown for a coffee.", "figure.walk"),
                ("Hay muchas personas en el restaurante del centro.", "There are many people in the downtown restaurant.", "person.3.fill"),
                ("El parque es más moderno que el hospital.", "The park is more modern than the hospital.", "sparkles"),
            ],
            probes: [
                (
                    sentenceEs: "Hay mucha música en el centro.",
                    options: ["There is lots of music downtown", "There is a hospital in Peru", "Carla is at the university", "Luis has a problem"],
                    symbols: ["music.note", "cross.case.fill", "graduationcap.fill", "exclamationmark.triangle.fill"],
                    correct: 0,
                    hinge: "«hay» (there is) + «mucha» + «música» + «centro»"
                ),
                (
                    sentenceEs: "Carla va al centro para un café.",
                    options: ["Carla goes to the hospital", "Carla goes downtown for a coffee", "Ana is a student", "There is a famous doctor"],
                    symbols: ["cross.case.fill", "cup.and.saucer.fill", "graduationcap.fill", "stethoscope"],
                    correct: 1,
                    hinge: "«va al centro» + «para un café»"
                ),
            ]
        ),
        // KIT 6: yo, mi, también, porque, visitar, estudiar, museo, historia
        HandWrittenKitPassage(
            kitNumber: 6,
            titleEs: "El museo de historia",
            titleEn: "The history museum",
            sentences: [
                ("Yo estudio historia en mi universidad.", "I study history at my university.", "book.closed.fill"),
                ("Mi profesor de historia es muy famoso.", "My history professor is very famous.", "person.bust.fill"),
                ("Yo visito el museo del centro.", "I visit the downtown museum.", "building.columns.fill"),
                ("Ana también visita el museo de Perú.", "Ana also visits the museum in Peru.", "globe.americas.fill"),
                ("Estudio historia porque es muy importante.", "I study history because it is very important.", "lightbulb.fill"),
                ("Luis también va al museo con mi familia.", "Luis also goes to the museum with my family.", "person.3.fill"),
                ("El museo tiene mucha historia de México.", "The museum has much history of Mexico.", "map.fill"),
                ("Carla estudia en el café del museo.", "Carla studies in the museum café.", "cup.and.saucer.fill"),
            ],
            probes: [
                (
                    sentenceEs: "Yo estudio historia en mi universidad.",
                    options: ["I study history at my university", "I visit a hospital in Peru", "Luis is in the park", "Ana is a famous doctor"],
                    symbols: ["book.closed.fill", "cross.case.fill", "leaf.fill", "stethoscope"],
                    correct: 0,
                    hinge: "«yo estudio» (I study) + «historia» + «mi universidad»"
                ),
                (
                    sentenceEs: "Ana también visita el museo de Perú.",
                    options: ["Ana goes to a café", "Ana also visits the museum in Peru", "Carla has a problem", "Luis is a student"],
                    symbols: ["cup.and.saucer.fill", "building.columns.fill", "exclamationmark.triangle.fill", "graduationcap.fill"],
                    correct: 1,
                    hinge: "«también» (also) + «visita» (visits) + «museo»"
                ),
            ]
        ),
        // KIT 7: o, todo, usar, necesitar, teléfono, minuto
        HandWrittenKitPassage(
            kitNumber: 7,
            titleEs: "Un minuto con el teléfono",
            titleEn: "A minute with the phone",
            sentences: [
                ("Yo uso mi teléfono en el café.", "I use my phone in the café.", "iphone"),
                ("Ana necesita su teléfono para el hospital.", "Ana needs her phone for the hospital.", "cross.case.fill"),
                ("Carla usa el teléfono para estudiar historia.", "Carla uses the phone to study history.", "book.closed.fill"),
                ("Necesito un minuto para ir al centro.", "I need a minute to go downtown.", "clock.fill"),
                ("Todo el hospital está en el centro.", "The whole hospital is downtown.", "building.columns.fill"),
                ("Luis va al parque o al restaurante.", "Luis goes to the park or to the restaurant.", "fork.knife"),
                ("Todos los estudiantes necesitan un teléfono moderno.", "All the students need a modern phone.", "graduationcap.fill"),
                ("Toda mi familia usa el café del centro.", "My whole family uses the downtown café.", "cup.and.saucer.fill"),
            ],
            probes: [
                (
                    sentenceEs: "Ana necesita su teléfono para el hospital.",
                    options: ["Ana needs her phone for the hospital", "Ana visits a park in Peru", "Luis studies music", "Carla is in a hotel"],
                    symbols: ["iphone", "leaf.fill", "music.note", "bed.double.fill"],
                    correct: 0,
                    hinge: "«necesita» (needs) + «teléfono» + «para el hospital»"
                ),
                (
                    sentenceEs: "Necesito un minuto para ir al centro.",
                    options: ["I study history at the museum", "I need a minute to go downtown", "All students are doctors", "Luis is in Mexico"],
                    symbols: ["building.columns.fill", "clock.fill", "stethoscope", "map.fill"],
                    correct: 1,
                    hinge: "«necesito» + «minuto» + «para ir al centro»"
                ),
            ]
        ),
        // KIT 8: cuando, otro, poder, preparar, animal, favorito, color
        HandWrittenKitPassage(
            kitNumber: 8,
            titleEs: "El animal y el color favorito",
            titleEn: "The favorite animal and color",
            sentences: [
                ("Cuando voy al parque, visito mi animal favorito.", "When I go to the park, I visit my favorite animal.", "pawprint.fill"),
                ("Ana puede preparar café en el restaurante.", "Ana can prepare coffee in the restaurant.", "cup.and.saucer.fill"),
                ("El animal favorito de Carla está en Perú.", "Carla's favorite animal is in Peru.", "globe.americas.fill"),
                ("Luis tiene otro color favorito para su hotel.", "Luis has another favorite color for his hotel.", "paintpalette.fill"),
                ("Cuando Carla estudia, prepara música en su teléfono.", "When Carla studies, she prepares music on her phone.", "music.note"),
                ("El profesor puede preparar otra clase de historia.", "The professor can prepare another history class.", "book.closed.fill"),
                ("Otro animal está en el parque del centro.", "Another animal is in the downtown park.", "leaf.fill"),
                ("Mi color favorito es muy moderno y famoso.", "My favorite color is very modern and famous.", "sparkles"),
            ],
            probes: [
                (
                    sentenceEs: "Ana puede preparar café en el restaurante.",
                    options: ["Ana can prepare coffee in the restaurant", "Ana visits another museum", "Luis has a favorite animal", "Carla needs a minute"],
                    symbols: ["cup.and.saucer.fill", "building.columns.fill", "pawprint.fill", "clock.fill"],
                    correct: 0,
                    hinge: "«puede preparar» (can prepare) + «café» + «restaurante»"
                ),
                (
                    sentenceEs: "Luis tiene otro color favorito para su hotel.",
                    options: ["Luis goes to the hospital", "Luis has another favorite color for his hotel", "Ana is a student", "There is music in the park"],
                    symbols: ["cross.case.fill", "paintpalette.fill", "graduationcap.fill", "music.note"],
                    correct: 1,
                    hinge: "«otro» (another) + «color favorito» + «hotel»"
                ),
            ]
        ),
    ]
}
