import CadenciaCore
import CadenciaSpeech
import Charts
import SwiftUI

public enum MemoryFilterMode: String, CaseIterable, Identifiable {
    case all = "All (4.5k)"
    case due = "Due Now"
    case learning = "Learning"
    case known = "Known"
    case productive = "Productive"
    case chunks = "Chunks"
    case grammar = "Grammar"

    public var id: String { rawValue }
}

/// Interactive 4,500-Item Memory Constellation & Lexical/Grammar Ledger:
///  - Pinch & pan GPU-instanced Metal Constellation (all 4,500 active syllabus items colored by mastery state & due pulse)
///  - Instant search + filter chips across Lemmas, 200 Chunks, 300 Cervantes A1–B2 Grammar Patterns
///  - Item Inspector sheet with live Swift Charts $p = 2^{-\Delta/h}$ forgetting curve, regional IPA,
///    prior Rule 4 fresh-context sentences, and one-tap Phonon-40m-es native Spanish pronunciation playback!
public struct MemoryAtlasView: View {
    @Bindable var appState: AppState

    @State private var filterMode: MemoryFilterMode = .all
    @State private var searchText: String = ""
    @State private var zoom: Float = 1.0
    @State private var baseZoom: Float = 1.0
    @State private var pan: SIMD2<Float> = .zero
    @State private var basePan: SIMD2<Float> = .zero
    @State private var selectedItem: SyllabusItem?

    public init(appState: AppState) {
        self.appState = appState
    }

    private var filteredItems: [SyllabusItem] {
        let params = appState.profile?.parameters ?? .protocolDefaults
        let eng = LearnerModelEngine(parameters: params)
        let now = Date()
        let q = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        return appState.allSyllabusItems.filter { item in
            let rec = appState.learnerRecords[item.id]
            let matchesMode: Bool
            switch filterMode {
            case .all: matchesMode = true
            case .due: matchesMode = rec.map { eng.isDue(record: $0, at: now) } ?? false
            case .learning: matchesMode = rec?.state == .learning
            case .known: matchesMode = rec?.state == .known
            case .productive: matchesMode = rec?.state == .productive
            case .chunks: matchesMode = item.itemType == .chunk
            case .grammar: matchesMode = item.itemType == .grammar
            }
            guard matchesMode else { return false }
            if q.isEmpty { return true }
            return item.spanish.lowercased().contains(q)
                || item.englishGloss.lowercased().contains(q)
                || (item.grammarCategory?.lowercased().contains(q) ?? false)
        }
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Top GPU Constellation Stage
            ZStack(alignment: .topLeading) {
                MemoryConstellationMetalView(
                    stars: appState.constellationStars,
                    zoom: zoom,
                    pan: pan
                )
                .frame(height: 250)
                .clipShape(RoundedRectangle(cornerRadius: 28, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.14), lineWidth: 1)
                )
                .gesture(
                    SimultaneousGesture(
                        MagnificationGesture()
                            .onChanged { val in
                                zoom = min(5.0, max(0.7, baseZoom * Float(val)))
                            }
                            .onEnded { _ in baseZoom = zoom },
                        DragGesture()
                            .onChanged { g in
                                let dx = Float(g.translation.width) / 220.0
                                let dy = -Float(g.translation.height) / 220.0
                                pan = SIMD2<Float>(basePan.x + dx, basePan.y + dy)
                            }
                            .onEnded { _ in basePan = pan }
                    )
                )

                // Overlay Header & Legend
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("MEMORY · CONSTELLATION")
                                .font(CadenciaFont.mono(10, weight: .bold))
                                .foregroundStyle(CadenciaPalette.saffronGold)
                                .tracking(1.3)
                            Text("\(appState.allSyllabusItems.count) Active Items")
                                .font(CadenciaFont.displaySerif(22, weight: .bold))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                        }
                        Spacer()
                        Button {
                            withAnimation(.spring(response: 0.4)) {
                                zoom = 1.0; baseZoom = 1.0
                                pan = .zero; basePan = .zero
                            }
                        } label: {
                            Image(systemName: "scope")
                                .font(.system(size: 13, weight: .bold))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                                .padding(8)
                                .cadenciaGlassPill()
                        }
                    }

                    Spacer()

                    HStack(spacing: 12) {
                        legendDot("Learning", CadenciaPalette.terracotta, count: appState.telemetry?.stateCounts[.learning] ?? 0)
                        legendDot("Known", CadenciaPalette.bioluminescentCobalt, count: appState.telemetry?.stateCounts[.known] ?? 0)
                        legendDot("Productive", CadenciaPalette.saffronGold, count: appState.telemetry?.stateCounts[.productive] ?? 0)
                        legendDot("Due", CadenciaPalette.crimsonAlert, count: appState.dueNowCount)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .cadenciaGlassPill(interactive: false)
                }
                .padding(14)
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)

            // Search & Filter Bar
            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                    TextField("Search Spanish lemma, English gloss, or Cervantes grammar…", text: $searchText)
                        .textFieldStyle(.plain)
                        .font(CadenciaFont.uiCaption(14))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                    if !searchText.isEmpty {
                        Button { searchText = "" } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(CadenciaPalette.mutedSilver)
                        }
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color.black.opacity(0.32), in: Capsule())
                .overlay(Capsule().strokeBorder(Color.white.opacity(0.14)))
                .padding(.horizontal, 16)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(MemoryFilterMode.allCases) { mode in
                            let active = filterMode == mode
                            Button {
                                filterMode = mode
                                HapticDirector.shared.playGlossTap()
                            } label: {
                                Text(mode.rawValue)
                                    .font(CadenciaFont.mono(11, weight: active ? .bold : .medium))
                                    .foregroundStyle(active ? CadenciaPalette.obsidian : CadenciaPalette.ivoryText)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 7)
                                    .background(active ? CadenciaPalette.saffronGold : Color.white.opacity(0.08), in: Capsule())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 16)
                }
            }
            .padding(.top, 12)
            .padding(.bottom, 8)

            // Virtualized Syllabus List
            let results = filteredItems
            ScrollView {
                LazyVStack(spacing: 10) {
                    if results.isEmpty {
                        Text("No items match this filter yet.")
                            .font(CadenciaFont.uiCaption(14))
                            .foregroundStyle(CadenciaPalette.mutedSilver)
                            .padding(.top, 40)
                    } else {
                        ForEach(results.prefix(250)) { item in
                            Button {
                                selectedItem = item
                                HapticDirector.shared.playGlossTap()
                            } label: {
                                itemRow(item)
                                    .contentShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                            }
                            .buttonStyle(.plain)
                        }
                        if results.count > 250 {
                            Text("Showing first 250 of \(results.count) matching items — refine search to inspect any item.")
                                .font(CadenciaFont.mono(11))
                                .foregroundStyle(CadenciaPalette.mutedSilver)
                                .padding(.vertical, 14)
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 24)
            }
        }
        .sheet(item: $selectedItem) { item in
            ItemForgettingDetailSheet(item: item, appState: appState)
                .presentationDetents([.large])
        }
    }

    private func legendDot(_ title: String, _ color: Color, count: Int) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 7, height: 7)
            Text("\(title) \(count)")
                .font(CadenciaFont.mono(10, weight: .semibold))
                .foregroundStyle(CadenciaPalette.ivoryText)
        }
    }

    private func itemRow(_ item: SyllabusItem) -> some View {
        let rec = appState.learnerRecords[item.id]
        let state = rec?.state ?? .new
        let recall = rec.map { $0.meetingsCount > 0 ? $0.predictedRecall() : 0.0 } ?? 0.0
        let hDays = rec?.halfLifeDays ?? (item.isTransparentCognate ? 1.0 : 15.0 / 1440.0)

        return HStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 4)
                .fill(CadenciaPalette.color(for: state))
                .frame(width: 4, height: 42)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(item.spanish)
                        .font(CadenciaFont.displaySerif(18, weight: .semibold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                    if item.isTransparentCognate {
                        Text("COG")
                            .font(CadenciaFont.mono(9, weight: .bold))
                            .foregroundStyle(CadenciaPalette.mediterraneanJade)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(CadenciaPalette.mediterraneanJade.opacity(0.16), in: Capsule())
                    }
                    Text(item.cefr.rawValue)
                        .font(CadenciaFont.mono(9, weight: .bold))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
                Text(item.englishGloss)
                    .font(CadenciaFont.uiCaption(13))
                    .foregroundStyle(CadenciaPalette.mutedSilver)
                    .lineLimit(1)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 3) {
                Text(state.displayName.uppercased())
                    .font(CadenciaFont.mono(9, weight: .bold))
                    .foregroundStyle(CadenciaPalette.color(for: state))
                if (rec?.meetingsCount ?? 0) > 0 {
                    Text("p=\(Int((recall * 100).rounded()))% · h=\(formatHalfLife(hDays))")
                        .font(CadenciaFont.mono(10))
                        .foregroundStyle(CadenciaPalette.ivoryText.opacity(0.85))
                } else {
                    Text("#\(item.rankIndex) · Unmet")
                        .font(CadenciaFont.mono(10))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .cadenciaGlassCard(cornerRadius: 16)
    }

    private func formatHalfLife(_ days: Double) -> String {
        if days < 1.0 / 24.0 {
            return "\(Int((days * 1440).rounded()))m"
        } else if days < 1.0 {
            return String(format: "%.1fh", days * 24)
        } else {
            return String(format: "%.1fd", days)
        }
    }
}

// MARK: - Item Forgetting-Curve Inspector Sheet

private struct ItemForgettingDetailSheet: View {
    let item: SyllabusItem
    @Bindable var appState: AppState

    @State private var priorSentences: [String] = []
    @State private var isSpeaking: Bool = false

    private struct CurvePoint: Identifiable {
        let id: Int
        let day: Double
        let recall: Double
    }

    private var record: LearnerItemRecord? {
        appState.learnerRecords[item.id]
    }

    private var halfLifeDays: Double {
        record?.halfLifeDays ?? (item.isTransparentCognate ? 1.0 : 15.0 / 1440.0)
    }

    private var curvePoints: [CurvePoint] {
        let horizon = max(1.0, min(60.0, halfLifeDays * 4.0))
        return (0...30).map { i in
            let d = horizon * Double(i) / 30.0
            return CurvePoint(id: i, day: d, recall: LearnerModelEngine.predictedRecall(deltaDays: d, halfLifeDays: halfLifeDays))
        }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                CadenciaPalette.obsidian.ignoresSafeArea()
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        // Title + Phonon-40m-es Speak Button
                        HStack(alignment: .top) {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("\(item.itemType.rawValue.uppercased()) · #\(item.rankIndex) · \(item.cefr.rawValue)")
                                    .font(CadenciaFont.mono(11, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.saffronGold)
                                Text(item.spanish)
                                    .font(CadenciaFont.displaySerif(34, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.ivoryText)
                                Text(item.englishGloss)
                                    .font(CadenciaFont.uiTitle(18))
                                    .foregroundStyle(CadenciaPalette.mutedSilver)
                            }
                            Spacer()
                            Button {
                                Task {
                                    isSpeaking = true
                                    defer { isSpeaking = false }
                                    try? await appState.ensurePhononLoaded()
                                    if let clip = try? await appState.phononEngine.synthesizeSentence(
                                        text: item.spanish, voice: .celia, accent: .castilian
                                    ) {
                                        try? await appState.audioPlayer.play(clip: clip, targetWPM: 140)
                                    }
                                }
                            } label: {
                                Image(systemName: isSpeaking ? "waveform" : "speaker.wave.2.fill")
                                    .font(.system(size: 18, weight: .semibold))
                                    .foregroundStyle(CadenciaPalette.obsidian)
                                    .padding(14)
                                    .background(CadenciaPalette.saffronGold, in: Circle())
                            }
                        }

                        // Regional IPA Card
                        HStack(spacing: 14) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("CASTILIAN IPA")
                                    .font(CadenciaFont.mono(9, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.mutedSilver)
                                Text("/\(item.ipaEs)/")
                                    .font(CadenciaFont.mono(13))
                                    .foregroundStyle(CadenciaPalette.ivoryText)
                            }
                            Spacer()
                            VStack(alignment: .leading, spacing: 2) {
                                Text("LATIN AMERICAN IPA")
                                    .font(CadenciaFont.mono(9, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.mutedSilver)
                                Text("/\(item.ipaLatam)/")
                                    .font(CadenciaFont.mono(13))
                                    .foregroundStyle(CadenciaPalette.ivoryText)
                            }
                        }
                        .padding(14)
                        .cadenciaGlassCard(cornerRadius: 16)

                        // Swift Charts Forgetting Curve: p = 2^(-\Delta / h)
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text("HALF-LIFE FORGETTING CURVE · p = 2^(−Δ/h)")
                                    .font(CadenciaFont.mono(10, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                                Spacer()
                                Text(String(format: "h = %.2f d", halfLifeDays))
                                    .font(CadenciaFont.mono(11, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.saffronGold)
                            }

                            Chart {
                                ForEach(curvePoints) { pt in
                                    AreaMark(
                                        x: .value("Days", pt.day),
                                        y: .value("Recall", pt.recall * 100)
                                    )
                                    .foregroundStyle(
                                        LinearGradient(
                                            colors: [CadenciaPalette.bioluminescentCobalt.opacity(0.45), .clear],
                                            startPoint: .top,
                                            endPoint: .bottom
                                        )
                                    )
                                    LineMark(
                                        x: .value("Days", pt.day),
                                        y: .value("Recall", pt.recall * 100)
                                    )
                                    .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                                    .lineStyle(StrokeStyle(lineWidth: 2.5))
                                }

                                // 85% Due Threshold RuleMark
                                RuleMark(y: .value("Due Threshold", 85))
                                    .foregroundStyle(CadenciaPalette.terracotta.opacity(0.8))
                                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 4]))
                                    .annotation(position: .top, alignment: .trailing) {
                                        Text("Due ≤ 85%")
                                            .font(CadenciaFont.mono(9, weight: .bold))
                                            .foregroundStyle(CadenciaPalette.terracotta)
                                    }
                            }
                            .chartYScale(domain: 0...100)
                            .chartXAxisLabel("Days since meeting (Δ)")
                            .chartYAxisLabel("Predicted Recall (%)")
                            .frame(height: 185)
                        }
                        .padding(16)
                        .cadenciaGlassCard(cornerRadius: 20)

                        // State & Evidence Counters
                        if let rec = record {
                            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                                statChip("State", rec.state.displayName)
                                statChip("Meetings", "\(rec.meetingsCount)")
                                statChip("Successes", "\(rec.successesCount)")
                                statChip("Ear-Only", "\(rec.earSuccessesCount)")
                                statChip("Productive", "\(rec.productiveUsesCount)")
                                statChip("Lookups", "\(rec.lookupsCount)")
                            }
                        }

                        // Grammar Micro-Note & Contrasting Pair (for the 300 Cervantes patterns)
                        if let note = item.microNoteEn {
                            VStack(alignment: .leading, spacing: 10) {
                                Text("CERVANTES GRAMMAR MICRO-NOTE (20s)")
                                    .font(CadenciaFont.mono(10, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.saffronGold)
                                Text(note)
                                    .font(CadenciaFont.uiCaption(14))
                                    .foregroundStyle(CadenciaPalette.ivoryText)
                                if let aEs = item.contrastAEs, let bEs = item.contrastBEs {
                                    VStack(alignment: .leading, spacing: 6) {
                                        Text("A: \(aEs) — \(item.contrastAEn ?? "")")
                                            .font(CadenciaFont.proseSerif(15))
                                            .foregroundStyle(CadenciaPalette.mediterraneanJade)
                                        Text("B: \(bEs) — \(item.contrastBEn ?? "")")
                                            .font(CadenciaFont.proseSerif(15))
                                            .foregroundStyle(CadenciaPalette.terracotta)
                                    }
                                    .padding(12)
                                    .background(Color.black.opacity(0.3), in: RoundedRectangle(cornerRadius: 12))
                                }
                            }
                            .padding(16)
                            .cadenciaGlassCard(cornerRadius: 20, tint: CadenciaPalette.saffronGold)
                        }

                        // Rule 4 Fresh-Context History
                        if !priorSentences.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("RULE 4 FRESH-CONTEXT HISTORY (\(priorSentences.count) SENTENCES)")
                                    .font(CadenciaFont.mono(10, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.mutedSilver)
                                ForEach(priorSentences, id: \.self) { s in
                                    Text("• \(s)")
                                        .font(CadenciaFont.proseSerif(15))
                                        .foregroundStyle(CadenciaPalette.ivoryText.opacity(0.9))
                                }
                            }
                            .padding(16)
                            .cadenciaGlassCard(cornerRadius: 20)
                        }
                    }
                    .padding(20)
                }
            }
            .navigationTitle("Item Ledger")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
        }
        .task {
            if let lid = appState.profile?.learnerId {
                priorSentences = (try? await appState.database.loadPriorSentences(learnerId: lid, itemId: item.id, limit: 8)) ?? []
            }
        }
    }

    private func statChip(_ title: String, _ value: String) -> some View {
        VStack(spacing: 4) {
            Text(title.uppercased())
                .font(CadenciaFont.mono(9, weight: .bold))
                .foregroundStyle(CadenciaPalette.mutedSilver)
            Text(value)
                .font(CadenciaFont.mono(14, weight: .bold))
                .foregroundStyle(CadenciaPalette.ivoryText)
        }
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity)
        .cadenciaGlassCard(cornerRadius: 14)
    }
}
