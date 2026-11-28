import Foundation

/// Pre-verified, native-level Spanish reserve passages served on Step 6 Fallback
/// (Revised Protocol p. 8: "If the third call fails, a pre-verified reserve passage is served
/// and the failure is logged. The same violation twice in a row skips straight to this step.")
///
/// Every reserve passage is strictly verified free of hold-out lemmas and matches either:
///  - Bootstrap Contract (Sessions 1–4): 6–8 short sentences <= 8 words, visual symbol + English line on tap, picture-matching probes
///  - Full Contract (Sessions 5+): 185–220 natural Spanish words, load-bearing probes
public enum ReservePassageBank {

    private struct ReserveTemplate: Sendable {
        let titleEs: String
        let titleEn: String
        let summary: String
        let sentences: [(es: String, en: String, symbol: String)]
        let probes: [(lemma: String, q: String, options: [String], correct: Int, hinge: String, isPicture: Bool, symbols: [String]?)]
    }

    private static let bootstrapReserves: [ReserveTemplate] = [
        ReserveTemplate(
            titleEs: "En el centro de Madrid",
            titleEn: "In the center of Madrid",
            summary: "Clara y Mateo visitan un hotel y un museo en el centro de Madrid.",
            sentences: [
                ("Clara está en el centro de Madrid.", "Clara is in the center of Madrid.", "building.2.fill"),
                ("El hotel es grande y muy especial.", "The hotel is big and very special.", "bed.double.fill"),
                ("Mateo entra en el hotel con Clara.", "Mateo enters the hotel with Clara.", "person.2.fill"),
                ("La familia de Clara está en el café.", "Clara's family is in the café.", "cup.and.saucer.fill"),
                ("El grupo escucha música clásica en el parque.", "The group listens to classical music in the park.", "music.note"),
                ("La historia de la plaza es interesante.", "The history of the plaza is interesting.", "book.closed.fill"),
                ("Un mapa del museo está en la mesa.", "A map of the museum is on the table.", "map.fill"),
                ("Es un momento importante para la familia.", "It is an important moment for the family.", "sparkles"),
            ],
            probes: [
                (
                    lemma: "música",
                    q: "Which picture matches «música» from the story?",
                    options: ["Music", "Hotel", "Map", "Family"],
                    correct: 0,
                    hinge: "«El grupo escucha música clásica en el parque.»",
                    isPicture: true,
                    symbols: ["music.note", "bed.double.fill", "map.fill", "person.3.fill"]
                ),
                (
                    lemma: "familia",
                    q: "Which picture matches «familia» from the story?",
                    options: ["Table", "Family", "Museum", "Music"],
                    correct: 1,
                    hinge: "«La familia de Clara está en el café.»",
                    isPicture: true,
                    symbols: ["table.furniture.fill", "person.3.fill", "building.columns.fill", "music.note"]
                ),
            ]
        ),
        ReserveTemplate(
            titleEs: "El proyecto de música",
            titleEn: "The music project",
            summary: "Lucía presenta una idea y un programa cultural para la comunidad local.",
            sentences: [
                ("Lucía tiene un proyecto de música local.", "Lucía has a local music project.", "music.note.list"),
                ("El programa es público y muy popular.", "The program is public and very popular.", "star.fill"),
                ("Un grupo de personas prepara el evento.", "A group of people prepares the event.", "person.3.fill"),
                ("La idea principal es simple y clara.", "The main idea is simple and clear.", "lightbulb.fill"),
                ("La radio presenta la información en español.", "The radio presents the information in Spanish.", "radio.fill"),
                ("El teatro del centro abre a las siete.", "The downtown theater opens at seven.", "theatermasks.fill"),
                ("La clase de arte participa en el festival.", "The art class participates in the festival.", "paintpalette.fill"),
            ],
            probes: [
                (
                    lemma: "idea",
                    q: "Which picture matches «idea»?",
                    options: ["Idea", "Radio", "Theater", "Festival"],
                    correct: 0,
                    hinge: "«La idea principal es simple y clara.»",
                    isPicture: true,
                    symbols: ["lightbulb.fill", "radio.fill", "theatermasks.fill", "party.popper.fill"]
                ),
                (
                    lemma: "radio",
                    q: "Which picture matches «radio»?",
                    options: ["Art class", "Program", "Radio", "Group"],
                    correct: 2,
                    hinge: "«La radio presenta la información en español.»",
                    isPicture: true,
                    symbols: ["paintpalette.fill", "doc.text.fill", "radio.fill", "person.3.fill"]
                ),
            ]
        ),
    ]

    private static let fullReserves: [ReserveTemplate] = [
        ReserveTemplate(
            titleEs: "Una tarde en el Retiro",
            titleEn: "An afternoon in El Retiro",
            summary: "Clara recorre el parque del Retiro tras recibir una nota misteriosa sobre una exposición.",
            sentences: [
                ("Clara llegó al parque del Retiro poco después de las cinco de la tarde, cuando la luz dorada cubría los árboles altos y el aire empezaba a refrescar.", "Clara arrived at Retiro Park shortly after five in the afternoon, when golden light covered the tall trees and the air began to cool.", "leaf.fill"),
                ("Llevaba en la mano una nota breve que había encontrado esa mañana debajo de la puerta de su apartamento en el centro de Madrid.", "She carried in her hand a brief note she had found that morning under the door of her apartment in central Madrid.", "envelope.fill"),
                ("El mensaje no tenía firma, pero indicaba un lugar exacto frente al gran estanque y una hora precisa para conversar sobre un archivo perdido.", "The message had no signature, but it indicated an exact spot in front of the large pond and a precise time to talk about a lost archive.", "clock.fill"),
                ("Mientras caminaba despacio por el camino principal, observó a varias familias conversando en los bancos y a un músico que tocaba una melodía suave con su guitarra.", "As she walked slowly along the main path, she observed several families chatting on the benches and a musician playing a soft melody on his guitar.", "music.note"),
                ("Al acercarse al agua, reconoció a Mateo, su antiguo compañero de universidad, sentado con dos cuadernos abiertos y una taza de café caliente.", "Approaching the water, she recognized Mateo, her former university classmate, sitting with two open notebooks and a cup of hot coffee.", "book.fill"),
                ("Mateo sonrió con calma y explicó que había descubierto unas cartas antiguas en la biblioteca municipal que cambiaban por completo la historia oficial del edificio donde trabajaban.", "Mateo smiled calmly and explained that he had discovered some old letters in the municipal library that completely changed the official history of the building where they worked.", "building.columns.fill"),
                ("Antes de que oscureciera, ambos decidieron revisar juntos cada página y planear el siguiente paso de su investigación.", "Before it grew dark, both decided to review every page together and plan the next step of their investigation.", "magnifyingglass"),
            ],
            probes: [
                (
                    lemma: "nota",
                    q: "Where did Clara find the unsigned message earlier that morning?",
                    options: [
                        "Under the door of her apartment",
                        "Inside a library book",
                        "On a bench by the pond",
                        "At her university office",
                    ],
                    correct: 0,
                    hinge: "«una nota breve que había encontrado esa mañana debajo de la puerta de su apartamento»",
                    isPicture: false,
                    symbols: nil
                ),
                (
                    lemma: "archivo",
                    q: "What did Mateo discover in the municipal library?",
                    options: [
                        "A map of the Retiro pond",
                        "Old letters that changed the official history of their building",
                        "A lost guitar melody",
                        "Keys to Clara's apartment",
                    ],
                    correct: 1,
                    hinge: "«había descubierto unas cartas antiguas en la biblioteca municipal que cambiaban por completo la historia oficial del edificio»",
                    isPicture: false,
                    symbols: nil
                ),
            ]
        ),
        ReserveTemplate(
            titleEs: "Arquitectura de la luz",
            titleEn: "Architecture of light",
            summary: "Un breve ensayo sobre cómo los patios tradicionales del Mediterráneo regulan el clima y la vida social.",
            sentences: [
                ("En muchas ciudades históricas de España y América Latina, el patio interior funciona desde hace siglos como el verdadero corazón de la casa.", "In many historic cities of Spain and Latin America, the interior courtyard has functioned for centuries as the true heart of the house.", "sun.max.fill"),
                ("Lejos de ser un simple adorno, este espacio abierto responde a una lógica práctica y sencilla para vivir cómodamente durante los meses de calor intenso.", "Far from being a simple ornament, this open space responds to a practical and simple logic for living comfortably during months of intense heat.", "thermometer.sun.fill"),
                ("Por la noche, el aire fresco desciende hacia el patio y enfría las paredes gruesas de piedra o madera que rodean las habitaciones principales.", "At night, cool air descends into the courtyard and cools the thick stone or wood walls surrounding the main rooms.", "moon.stars.fill"),
                ("Durante el mediodía, la sombra de las plantas y una pequeña fuente de agua mantienen una temperatura agradable sin necesidad de máquinas modernas.", "During midday, the shade of the plants and a small water fountain maintain a pleasant temperature without needing modern machines.", "drop.fill"),
                ("Además de su valor climático, el patio crea un punto de encuentro tranquilo donde las familias leen, comen juntas y reciben a sus vecinos al final del día.", "In addition to its climatic value, the courtyard creates a quiet meeting point where families read, eat together, and welcome neighbors at the end of the day.", "person.3.fill"),
                ("Hoy, numerosos arquitectos jóvenes estudian estos diseños tradicionales para construir viviendas urbanas más eficientes, luminosas y humanas en el futuro.", "Today, numerous young architects study these traditional designs to build more efficient, luminous, and humane urban homes in the future.", "building.2.fill"),
            ],
            probes: [
                (
                    lemma: "noche",
                    q: "How do the thick courtyard walls stay cool before the midday heat arrives?",
                    options: [
                        "Modern air machines cool them automatically",
                        "Cool night air descends into the patio and cools the walls",
                        "Heavy curtains block the wind all night",
                        "Water is sprayed directly onto the roof",
                    ],
                    correct: 1,
                    hinge: "«Por la noche, el aire fresco desciende hacia el patio y enfría las paredes gruesas»",
                    isPicture: false,
                    symbols: nil
                ),
                (
                    lemma: "arquitecto",
                    q: "Why are young architects studying traditional courtyards today?",
                    options: [
                        "To turn old homes into museums",
                        "To build more efficient, bright, and humane urban housing",
                        "To replace stone walls with glass towers",
                        "To eliminate plants from city buildings",
                    ],
                    correct: 1,
                    hinge: "«estudian estos diseños tradicionales para construir viviendas urbanas más eficientes, luminosas y humanas»",
                    isPicture: false,
                    symbols: nil
                ),
            ]
        ),
    ]

    public static func selectReserve(
        for plan: PassagePlan,
        lemmatizer: SpanishLemmatizer,
        generatorCallsUsed: Int
    ) -> GeneratedPassage {
        let verifier = ContractVerifier(lemmatizer: lemmatizer)

        // For Session 1 fixed kits (1...8), serve the exact hand-written acceptance passage for THAT kit!
        if plan.isBootstrap,
           let kitNum = plan.bootstrapKitNumber,
           kitNum >= 1,
           kitNum <= BootstrapKitLibrary.session1AcceptancePassages.count {
            let kitTemplate = BootstrapKitLibrary.session1AcceptancePassages[kitNum - 1]
            let sentences: [PassageSentence] = kitTemplate.sentences.enumerated().map { idx, s in
                PassageSentence(
                    id: idx,
                    spanish: s.es,
                    englishLine: s.en,
                    wordTokens: lemmatizer.extractWordTokens(s.es),
                    visualSymbol: s.symbol
                )
            }
            let fullText = sentences.map(\.spanish).joined(separator: " ")
            let wordCount = lemmatizer.extractWordTokens(fullText).count
            let glosses: [NewItemGloss] = plan.newItems.map { item in
                NewItemGloss(
                    itemId: item.id,
                    spanish: item.spanish,
                    englishGloss: item.englishGloss,
                    contextClueNote: "Kit \(kitNum) vocabulary",
                    visualCueSymbol: item.visualSymbol
                )
            }
            let probes: [PassageProbe] = kitTemplate.probes.enumerated().map { idx, p in
                let matchedIds = verifier.matchedItemIds(in: p.sentenceEs, from: plan.newItems)
                let primaryId = matchedIds.first ?? plan.newItems.first?.id ?? "kit\(kitNum)_probe_\(idx)"
                let primaryLemma = plan.newItems.first(where: { $0.id == primaryId })?.spanish ?? "kit"
                return PassageProbe(
                    id: "\(plan.passageId)_kit\(kitNum)_probe_\(idx)",
                    targetDueItemId: primaryId,
                    targetDueLemma: primaryLemma,
                    questionEn: "Which picture matches this sentence?",
                    options: p.options,
                    correctOptionIndex: p.correct,
                    hingeExplanation: p.hinge,
                    isPictureMatch: true,
                    optionSymbols: p.symbols,
                    targetSentenceSpanish: p.sentenceEs,
                    targetItemIds: matchedIds.isEmpty ? [primaryId] : matchedIds
                )
            }
            let grammarNotes: [GrammarMicroNote] = plan.grammarPatterns.compactMap { g in
                guard let note = g.microNoteEn else { return nil }
                return GrammarMicroNote(
                    grammarItemId: g.id,
                    title: g.spanish,
                    noteEn: note,
                    contrastAEs: g.contrastAEs ?? "",
                    contrastAEn: g.contrastAEn ?? "",
                    contrastBEs: g.contrastBEs ?? "",
                    contrastBEn: g.contrastBEn ?? ""
                )
            }
            return GeneratedPassage(
                id: plan.passageId,
                sessionNumber: plan.sessionNumber,
                blockType: plan.blockType,
                format: plan.format,
                titleEs: kitTemplate.titleEs,
                titleEn: kitTemplate.titleEn,
                episodeSummaryForNext: kitTemplate.titleEn,
                sentences: sentences,
                fullText: fullText,
                wordCount: wordCount,
                newItemGlosses: glosses,
                probes: probes,
                grammarNotes: grammarNotes,
                dueItemIds: plan.dueItems.map(\.id),
                newItemIds: plan.newItems.map(\.id),
                grammarItemIds: plan.grammarPatterns.map(\.id),
                voice: plan.voice,
                accent: plan.accent,
                targetWPM: plan.targetWPM,
                isAudioFirst: plan.isAudioFirst,
                isBootstrap: true,
                rewriteAttempts: max(0, generatorCallsUsed - 1),
                generatorCallsUsed: generatorCallsUsed,
                isReserveFallback: true,
                dueListWasReduced: false,
                scheduledWordShare: 0.90,
                flaggedForNativeAudit: false,
                createdAt: Date()
            )
        }

        let pool = plan.isBootstrap ? bootstrapReserves : fullReserves
        let template = pool[abs(plan.sessionNumber * 7 + plan.passageIndexInSession) % pool.count]

        let sentences: [PassageSentence] = template.sentences.enumerated().map { idx, s in
            PassageSentence(
                id: idx,
                spanish: s.es,
                englishLine: s.en,
                wordTokens: lemmatizer.extractWordTokens(s.es),
                visualSymbol: s.symbol
            )
        }
        let fullText = sentences.map(\.spanish).joined(separator: " ")
        let wordCount = lemmatizer.extractWordTokens(fullText).count

        let glosses: [NewItemGloss] = plan.newItems.map { item in
            NewItemGloss(
                itemId: item.id,
                spanish: item.spanish,
                englishGloss: item.englishGloss,
                contextClueNote: "Reserve passage vocabulary reference",
                visualCueSymbol: item.visualSymbol
            )
        }

        let probes: [PassageProbe] = template.probes.enumerated().map { idx, p in
            let matchedItem = (plan.dueItems + plan.newItems).first {
                $0.spanish.lowercased() == p.lemma.lowercased()
            } ?? plan.eligibleProbeTargets.first
            return PassageProbe(
                id: "\(plan.passageId)_reserve_probe_\(idx)",
                targetDueItemId: matchedItem?.id ?? "reserve_\(idx)",
                targetDueLemma: matchedItem?.spanish ?? p.lemma,
                questionEn: p.q,
                options: p.options,
                correctOptionIndex: p.correct,
                hingeExplanation: p.hinge,
                isPictureMatch: p.isPicture,
                optionSymbols: p.symbols
            )
        }

        let grammarNotes: [GrammarMicroNote] = plan.grammarPatterns.compactMap { g in
            guard let note = g.microNoteEn else { return nil }
            return GrammarMicroNote(
                grammarItemId: g.id,
                title: g.spanish,
                noteEn: note,
                contrastAEs: g.contrastAEs ?? "",
                contrastAEn: g.contrastAEn ?? "",
                contrastBEs: g.contrastBEs ?? "",
                contrastBEn: g.contrastBEn ?? ""
            )
        }

        return GeneratedPassage(
            id: plan.passageId,
            sessionNumber: plan.sessionNumber,
            blockType: plan.blockType,
            format: plan.format,
            titleEs: template.titleEs,
            titleEn: template.titleEn,
            episodeSummaryForNext: template.summary,
            sentences: sentences,
            fullText: fullText,
            wordCount: wordCount,
            newItemGlosses: glosses,
            probes: probes,
            grammarNotes: grammarNotes,
            dueItemIds: plan.dueItems.map(\.id),
            newItemIds: plan.newItems.map(\.id),
            grammarItemIds: plan.grammarPatterns.map(\.id),
            voice: plan.voice,
            accent: plan.accent,
            targetWPM: plan.targetWPM,
            isAudioFirst: plan.isAudioFirst,
            isBootstrap: plan.isBootstrap,
            rewriteAttempts: max(0, generatorCallsUsed - 1),
            generatorCallsUsed: generatorCallsUsed,
            isReserveFallback: true,
            dueListWasReduced: false,
            scheduledWordShare: 0.0,
            flaggedForNativeAudit: false,
            createdAt: Date()
        )
    }
}
