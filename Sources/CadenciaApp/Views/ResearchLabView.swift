import CadenciaCore
import CadenciaSpeech
import Charts
import SwiftUI

private enum ActiveLabSheet: String, Identifiable {
    case weeklyAudit
    case vocabSizeTest
    case spanishEIT
    case nativeListening
    case baselineSpeaking
    case stage0BenchTest

    var id: String { rawValue }
}

/// Complete Signal-Starvation Research Lab & External Measures Suite (PDF pp. 1–3, 6, 9–14):
///  - Live Verdicts for Claim 1 (Density), Claim 2 (Capacity 30/60/120), Claim 3 (Conversion / Top-2000)
///  - Weekly 60-Item Calibration Audit with GPT-6.1 Terra grading + Automated Online MLE Multiplier Refit
///  - Vocabulary Size Test (Yes/No + Pseudowords + 400-Item Hold-Out >15% Guard + 100-Item Typed Recall)
///  - Spanish Elicited Imitation Task (30 rising-length Ortega sentences via Magpie + Parakeet Ultra + ZIPA)
///  - Native-Speed Listening Test (Hours 30 & 60 unscripted passages + comprehension check)
///  - Baseline 10-Min Speaking Recorder (Solo Pilot Step 2)
///  - Stage 0 Bench Test Runner (Simulated Hours 2, 10, 20, 35, 50)
///  - Blind JSONL Append-Only Evidence Export & 2% Native-Speaker Rater Markdown Pack
public struct ResearchLabView: View {
    @Bindable var appState: AppState

    @State private var activeSheet: ActiveLabSheet?
    @State private var exportedJSONLURL: URL?
    @State private var exportedNativePackURL: URL?
    @State private var exportNotice: String?

    public init(appState: AppState) {
        self.appState = appState
    }

    public var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerSection
                hypothesisVerdictsSection
                if !appState.audits.isEmpty {
                    calibrationChartSection
                }
                researchBatterySection
                blindExportSection
            }
            .padding(.horizontal, 18)
            .padding(.top, 16)
            .padding(.bottom, 24)
        }
        .sheet(item: $activeSheet) { sheet in
            switch sheet {
            case .weeklyAudit:
                WeeklyAuditRunnerSheet(appState: appState)
            case .vocabSizeTest:
                VocabSizeTestSheet(appState: appState)
            case .spanishEIT:
                SpanishEITRunnerSheet(appState: appState)
            case .nativeListening:
                NativeListeningTestSheet(appState: appState)
            case .baselineSpeaking:
                BaselineSpeakingRecorderSheet(appState: appState)
            case .stage0BenchTest:
                Stage0BenchTestSheet(appState: appState)
            }
        }
    }

    private var headerSection: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("RESEARCH LAB · PROTOCOL TELEMETRY")
                    .font(CadenciaFont.mono(11, weight: .bold))
                    .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                    .tracking(1.3)
                Text("Hypothesis & Measures")
                    .font(CadenciaFont.displaySerif(30, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
            }
            Spacer()
            if let code = appState.profile?.codedGroupLabel {
                Text(code)
                    .font(CadenciaFont.mono(11, weight: .bold))
                    .foregroundStyle(CadenciaPalette.saffronGold)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .cadenciaGlassPill(tint: CadenciaPalette.saffronGold)
            }
        }
    }

    // MARK: - Live Claim 1 / Claim 2 / Claim 3 Cards

    private var hypothesisVerdictsSection: some View {
        let tel = appState.telemetry
        return VStack(spacing: 12) {
            claimCard(
                code: "CLAIM 1 · DENSITY",
                headline: String(format: "%.1f usable events / input min", tel?.usableEventsPerInputMinute ?? 0),
                detail: String(
                    format: "Probe accuracy %.0f%% · Target ≥ 15.0/min & ≥ 85%% (%d total usable events)",
                    (tel?.comprehensionProbeAccuracy ?? 1.0) * 100,
                    tel?.usableEventsTotal ?? 0
                ),
                statusLabel: (tel?.sessionsCompleted ?? 0) == 0 ? "PENDING" : (tel?.claim1DensityPassing == true ? "PASSING" : "BELOW GATE"),
                color: (tel?.claim1DensityPassing == true) ? CadenciaPalette.mediterraneanJade : CadenciaPalette.saffronGold
            )

            let ratioStr = tel?.claim2CapacityRatio120To30.map { String(format: "%.0f%% of 30/session", $0 * 100) } ?? "Awaiting 7-day delayed tests"
            claimCard(
                code: "CLAIM 2 · CAPACITY (30 / 60 / 120)",
                headline: ratioStr,
                detail: "Target: 7-day retention at 120 new/session ≥ 80% of 30 new/session. Logged buckets: \(tel?.capacityStats.count ?? 0)/3.",
                statusLabel: tel?.claim2CapacityPassing.map { $0 ? "PASSING" : "BELOW 80%" } ?? "STAGE 1",
                color: (tel?.claim2CapacityPassing == true) ? CadenciaPalette.mediterraneanJade : CadenciaPalette.bioluminescentCobalt
            )

            let prodCount = tel?.top2000ProductiveCount ?? 0
            let prodPct = Int(((tel?.top2000ProductiveRatio ?? 0) * 100).rounded())
            claimCard(
                code: "CLAIM 3 · CONVERSION & COVERAGE",
                headline: "\(prodCount) / 2,000 Top-Freq Productive (\(prodPct)%)",
                detail: "Known: \(tel?.stateCounts[.known] ?? 0) · Learning: \(tel?.stateCounts[.learning] ?? 0) · Logged: \(String(format: "%.1f", tel?.totalLoggedHours ?? 0)) / 60.0h",
                statusLabel: "TRACKING",
                color: CadenciaPalette.terracotta
            )
        }
    }

    private func claimCard(code: String, headline: String, detail: String, statusLabel: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(code)
                    .font(CadenciaFont.mono(10, weight: .bold))
                    .foregroundStyle(color)
                    .tracking(1.1)
                Spacer()
                Text(statusLabel)
                    .font(CadenciaFont.mono(9, weight: .bold))
                    .foregroundStyle(color)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(color.opacity(0.16), in: Capsule())
            }
            Text(headline)
                .font(CadenciaFont.displaySerif(22, weight: .bold))
                .foregroundStyle(CadenciaPalette.ivoryText)
            Text(detail)
                .font(CadenciaFont.uiCaption(12))
                .foregroundStyle(CadenciaPalette.mutedSilver)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .cadenciaGlassCard(cornerRadius: 22, tint: color)
    }

    // MARK: - Weekly Calibration Audit Chart

    private var calibrationChartSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("WEEKLY 60-ITEM CALIBRATION (PREDICTED vs OBSERVED)")
                    .font(CadenciaFont.mono(10, weight: .bold))
                    .foregroundStyle(CadenciaPalette.saffronGold)
                Spacer()
                if let gap = appState.audits.first?.absoluteCalibrationError {
                    Text(String(format: "Δ %.1fpp", gap * 100))
                        .font(CadenciaFont.mono(11, weight: .bold))
                        .foregroundStyle(gap > 0.10 ? CadenciaPalette.crimsonAlert : CadenciaPalette.mediterraneanJade)
                }
            }

            Chart {
                ForEach(appState.audits.reversed()) { audit in
                    BarMark(
                        x: .value("Week", "W\(audit.week)"),
                        y: .value("Predicted", audit.meanPredictedRecall * 100)
                    )
                    .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                    .position(by: .value("Series", "Predicted"))

                    BarMark(
                        x: .value("Week", "W\(audit.week)"),
                        y: .value("Observed", audit.meanObservedRecall * 100)
                    )
                    .foregroundStyle(CadenciaPalette.saffronGold)
                    .position(by: .value("Series", "Observed"))
                }
            }
            .chartYScale(domain: 0...100)
            .frame(height: 165)
        }
        .padding(16)
        .cadenciaGlassCard(cornerRadius: 22)
    }

    // MARK: - Research Battery Launchers (All Outside 60h Timer!)

    private var researchBatterySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("EXTERNAL MEASURES & AUDITS (OUTSIDE 60H TIMER)")
                .font(CadenciaFont.mono(10, weight: .bold))
                .foregroundStyle(CadenciaPalette.mutedSilver)
                .tracking(1.1)

            labActionRow(
                icon: "checklist.checked",
                title: "Weekly 60-Item Calibration Audit",
                subtitle: "60 no-context typed prompts · Auto-refits half-life multipliers if |pred−obs| > 10pp",
                tint: CadenciaPalette.saffronGold
            ) { activeSheet = .weeklyAudit }

            labActionRow(
                icon: "textformat.abc",
                title: "Vocabulary Size Test + Hold-Out Guard",
                subtitle: "Yes/No real + pseudowords + 400 hold-out inflation check (>15%) + 100-item recall",
                tint: CadenciaPalette.mediterraneanJade
            ) { activeSheet = .vocabSizeTest }

            labActionRow(
                icon: "waveform.and.mic",
                title: "Spanish Elicited Imitation Task (EIT)",
                subtitle: "30 Ortega sentences (7–17 syllables) · Magpie audio -> Parakeet Ultra + ZIPA 0–4 scoring",
                tint: CadenciaPalette.bioluminescentCobalt
            ) { activeSheet = .spanishEIT }

            labActionRow(
                icon: "headphones",
                title: "Native-Speed Listening Test (Hours 30 & 60)",
                subtitle: "Unscripted 145–155 WPM regional passages + comprehension checks",
                tint: CadenciaPalette.terracotta
            ) { activeSheet = .nativeListening }

            labActionRow(
                icon: "record.circle",
                title: "Baseline 10-Min Speaking Recorder",
                subtitle: "Solo Pilot Step 2 · Save baseline oral sample + Parakeet/ZIPA diagnostics",
                tint: CadenciaPalette.saffronGold
            ) { activeSheet = .baselineSpeaking }

            labActionRow(
                icon: "flask.fill",
                title: "Stage 0 Bench Test (Hours 2 / 10 / 20 / 35 / 50)",
                subtitle: "Stress-test GPT-6.1 Terra Generator + Verifier contract gate (≥95% pass, ≥20% share)",
                tint: CadenciaPalette.bioluminescentCobalt
            ) { activeSheet = .stage0BenchTest }
        }
    }

    private func labActionRow(icon: String, title: String, subtitle: String, tint: Color, action: @escaping () -> Void) -> some View {
        Button {
            HapticDirector.shared.playGlossTap()
            action()
        } label: {
            HStack(spacing: 14) {
                Image(systemName: icon)
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(tint)
                    .frame(width: 42, height: 42)
                    .background(tint.opacity(0.16), in: RoundedRectangle(cornerRadius: 12))

                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(CadenciaFont.uiTitle(15, weight: .semibold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                    Text(subtitle)
                        .font(CadenciaFont.uiCaption(11))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                        .multilineTextAlignment(.leading)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(CadenciaPalette.mutedSilver)
            }
            .padding(14)
            .cadenciaGlassCard(cornerRadius: 20)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Blind JSONL & 2% Native-Rater Export

    private var blindExportSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("BLIND RESEARCH EXPORTS")
                .font(CadenciaFont.mono(10, weight: .bold))
                .foregroundStyle(CadenciaPalette.mutedSilver)
                .tracking(1.1)

            HStack(spacing: 10) {
                Button {
                    Task { await prepareExports() }
                } label: {
                    Label("Build Export Packs", systemImage: "square.and.arrow.up")
                        .font(CadenciaFont.uiCaption(13, weight: .bold))
                        .foregroundStyle(CadenciaPalette.obsidian)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(CadenciaPalette.ivoryText, in: Capsule())
                }

                if let url = exportedJSONLURL {
                    ShareLink(item: url) {
                        Label("Evidence JSONL", systemImage: "doc.text")
                            .font(CadenciaFont.mono(11, weight: .bold))
                            .foregroundStyle(CadenciaPalette.mediterraneanJade)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .cadenciaGlassPill(tint: CadenciaPalette.mediterraneanJade)
                    }
                }

                if let mdURL = exportedNativePackURL {
                    ShareLink(item: mdURL) {
                        Label("2% Rater MD", systemImage: "checklist")
                            .font(CadenciaFont.mono(11, weight: .bold))
                            .foregroundStyle(CadenciaPalette.saffronGold)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .cadenciaGlassPill(tint: CadenciaPalette.saffronGold)
                    }
                }
            }

            if let note = exportNotice {
                Text(note)
                    .font(CadenciaFont.mono(11))
                    .foregroundStyle(CadenciaPalette.mutedSilver)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .cadenciaGlassCard(cornerRadius: 22)
    }

    private func prepareExports() async {
        guard let profile = appState.profile else { return }
        do {
            let events = try await appState.database.loadEvidenceEvents(learnerId: profile.learnerId)
            let jsonlData = appState.analyticsEngine.exportBlindEvidenceJSONL(profile: profile, events: events)
            let tmpDir = FileManager.default.temporaryDirectory
            let jsonlFile = tmpDir.appendingPathComponent("cadencia_blind_evidence_\(profile.codedGroupLabel).jsonl")
            try jsonlData.write(to: jsonlFile)
            exportedJSONLURL = jsonlFile

            let flagged = try await appState.database.loadNativeAuditPassages(learnerId: profile.learnerId)
            let md = appState.analyticsEngine.exportNativeRaterAuditMarkdown(passages: flagged)
            let mdFile = tmpDir.appendingPathComponent("cadencia_2pct_native_rater_\(profile.codedGroupLabel).md")
            try Data(md.utf8).write(to: mdFile)
            exportedNativePackURL = mdFile

            exportNotice = "Prepared \(events.count) immutable events & \(flagged.count) flagged 2% native-audit passages."
        } catch {
            exportNotice = "Export error: \(error.localizedDescription)"
        }
    }
}

// MARK: - 1. Weekly 60-Item Calibration Audit Runner + Auto MLE Refit

private struct WeeklyAuditRunnerSheet: View {
    @Bindable var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var prompts: [AuditPromptItem] = []
    @State private var currentIndex: Int = 0
    @State private var typedText: String = ""
    @State private var typedMap: [String: String] = [:]
    @State private var startedAt: Date = Date()
    @State private var isGrading: Bool = false
    @State private var completedResult: WeeklyAuditResult?
    @State private var errorMsg: String?

    var body: some View {
        NavigationStack {
            ZStack {
                CadenciaPalette.obsidian.ignoresSafeArea()
                VStack(spacing: 20) {
                    if let res = completedResult {
                        VStack(spacing: 16) {
                            Image(systemName: res.exceededTenPointThreshold ? "slider.horizontal.below.rectangle" : "checkmark.seal.fill")
                                .font(.system(size: 46))
                                .foregroundStyle(res.exceededTenPointThreshold ? CadenciaPalette.saffronGold : CadenciaPalette.mediterraneanJade)
                            Text("Week \(res.week) Audit Complete")
                                .font(CadenciaFont.displaySerif(28, weight: .bold))
                                .foregroundStyle(CadenciaPalette.ivoryText)

                            HStack(spacing: 16) {
                                scorePill("PREDICTED", "\(Int((res.meanPredictedRecall * 100).rounded()))%", CadenciaPalette.bioluminescentCobalt)
                                scorePill("OBSERVED", "\(Int((res.meanObservedRecall * 100).rounded()))%", CadenciaPalette.saffronGold)
                                scorePill("GAP", String(format: "%.1fpp", res.absoluteCalibrationError * 100), res.exceededTenPointThreshold ? CadenciaPalette.crimsonAlert : CadenciaPalette.mediterraneanJade)
                            }

                            if res.didRefitModel {
                                Text("Calibration gap exceeded 10pp — Learner Model multipliers were automatically refit via online MLE and saved to local SQLite.")
                                    .font(CadenciaFont.uiCaption(13))
                                    .foregroundStyle(CadenciaPalette.saffronGold)
                                    .multilineTextAlignment(.center)
                                    .padding()
                                    .cadenciaGlassCard(cornerRadius: 16, tint: CadenciaPalette.saffronGold)
                            }

                            Button("Done") { dismiss() }
                                .font(CadenciaFont.uiTitle(16, weight: .bold))
                                .foregroundStyle(CadenciaPalette.obsidian)
                                .padding(.horizontal, 36)
                                .padding(.vertical, 14)
                                .background(CadenciaPalette.ivoryText, in: Capsule())
                        }
                        .padding(24)
                    } else if isGrading {
                        VStack(spacing: 16) {
                            ProgressView().scaleEffect(1.4).tint(CadenciaPalette.saffronGold)
                            Text("Grading \(prompts.count) typed responses via GPT-6.1 Terra & checking calibration…")
                                .font(CadenciaFont.uiCaption(14))
                                .foregroundStyle(CadenciaPalette.mutedSilver)
                                .multilineTextAlignment(.center)
                        }
                        .padding(30)
                    } else if prompts.isEmpty {
                        VStack(spacing: 14) {
                            Text("Complete at least one session first so there are met items to audit.")
                                .font(CadenciaFont.uiCaption(15))
                                .foregroundStyle(CadenciaPalette.mutedSilver)
                                .multilineTextAlignment(.center)
                            Button("Close") { dismiss() }
                                .foregroundStyle(CadenciaPalette.saffronGold)
                        }
                        .padding(24)
                    } else if currentIndex < prompts.count {
                        let p = prompts[currentIndex]
                        VStack(spacing: 20) {
                            ProgressView(value: Double(currentIndex + 1), total: Double(prompts.count))
                                .tint(CadenciaPalette.saffronGold)
                            Text("ITEM \(currentIndex + 1) OF \(prompts.count) · NO CONTEXT")
                                .font(CadenciaFont.mono(11, weight: .bold))
                                .foregroundStyle(CadenciaPalette.mutedSilver)

                            Text(p.item.spanish)
                                .font(CadenciaFont.displaySerif(40, weight: .bold))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                                .multilineTextAlignment(.center)

                            Text("/\(p.item.ipaEs)/")
                                .font(CadenciaFont.mono(14))
                                .foregroundStyle(CadenciaPalette.mutedSilver)

                            TextField("Type English meaning (leave blank if unknown)…", text: $typedText)
                                .textFieldStyle(.plain)
                                .font(CadenciaFont.uiTitle(17))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                                .padding(16)
                                .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 16))
                                .onSubmit { advanceAuditItem() }

                            HStack(spacing: 12) {
                                Button("Don't Know") {
                                    typedText = ""
                                    advanceAuditItem()
                                }
                                .font(CadenciaFont.uiCaption(14, weight: .semibold))
                                .foregroundStyle(CadenciaPalette.mutedSilver)
                                .padding(.horizontal, 18)
                                .padding(.vertical, 12)
                                .cadenciaGlassPill()

                                Button(currentIndex + 1 == prompts.count ? "Finish & Grade" : "Next") {
                                    advanceAuditItem()
                                }
                                .font(CadenciaFont.uiTitle(15, weight: .bold))
                                .foregroundStyle(CadenciaPalette.obsidian)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 12)
                                .background(CadenciaPalette.saffronGold, in: Capsule())
                            }
                        }
                        .padding(24)
                    }

                    if let err = errorMsg {
                        Text(err)
                            .font(CadenciaFont.mono(11))
                            .foregroundStyle(CadenciaPalette.crimsonAlert)
                            .padding(.horizontal)
                    }
                }
            }
            .navigationTitle("Weekly 60-Item Audit")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
        .onAppear {
            guard let prof = appState.profile else { return }
            let sched = SessionScheduler(parameters: prof.parameters)
            prompts = sched.selectWeeklyAuditItems(allItems: appState.allSyllabusItems, records: appState.learnerRecords, count: 60)
            startedAt = Date()
        }
    }

    private func advanceAuditItem() {
        let p = prompts[currentIndex]
        typedMap[p.item.id] = typedText.trimmingCharacters(in: .whitespacesAndNewlines)
        typedText = ""
        if currentIndex + 1 < prompts.count {
            currentIndex += 1
        } else {
            Task { await gradeAndPersistAudit() }
        }
    }

    private func gradeAndPersistAudit() async {
        guard var prof = appState.profile else { return }
        isGrading = true
        defer { isGrading = false }

        do {
            let batchInput = prompts.map { p in
                (itemId: p.item.id, spanish: p.item.spanish, expectedGloss: p.item.englishGloss, typedEnglish: typedMap[p.item.id] ?? "")
            }
            let grades = try await appState.terraClient.gradeTypedMeaningBatch(items: batchInput)
            let responses: [AuditItemResponse] = prompts.map { p in
                let typed = typedMap[p.item.id] ?? ""
                let g = grades[p.item.id] ?? (false, "No grade returned")
                return AuditItemResponse(
                    itemId: p.item.id,
                    spanish: p.item.spanish,
                    expectedGloss: p.item.englishGloss,
                    learnerTypedMeaning: typed,
                    predictedRecall: p.predictedRecallAtAudit,
                    observedCorrect: typed.isEmpty ? false : g.correct,
                    gradingNote: g.note
                )
            }

            let meanPred = responses.reduce(0.0) { $0 + $1.predictedRecall } / Double(max(1, responses.count))
            let meanObs = Double(responses.filter(\.observedCorrect).count) / Double(max(1, responses.count))
            let absErr = abs(meanPred - meanObs)
            var didRefit = false

            // Auto-refit multipliers via coordinate MLE when |pred - obs| > 10pp and parameters are unfrozen (PDF p. 6)!
            if absErr > 0.10 && !prof.parameters.isFrozen {
                let eng = LearnerModelEngine(parameters: prof.parameters)
                let now = Date()
                let obs: [LearnerModelEngine.CalibrationObservation] = prompts.compactMap { p in
                    guard let rec = appState.learnerRecords[p.item.id], let last = rec.lastMeetingAt else { return nil }
                    let resp = responses.first(where: { $0.itemId == p.item.id })
                    return LearnerModelEngine.CalibrationObservation(
                        itemId: p.item.id,
                        deltaDays: max(0.001, now.timeIntervalSince(last) / 86_400.0),
                        strongCount: rec.productiveUsesCount,
                        weakCount: max(0, rec.successesCount - rec.productiveUsesCount),
                        failCount: rec.failuresCount,
                        isCognate: p.item.isTransparentCognate,
                        observedCorrect: resp?.observedCorrect ?? false
                    )
                }
                let refitParams = eng.refitMultipliers(observations: obs)
                if refitParams != prof.parameters {
                    prof.parameters = refitParams
                    try await appState.database.saveProfile(prof)
                    appState.profile = prof
                    didRefit = true
                    try? await appState.database.appendEvidence(EvidenceEvent(
                        learnerId: prof.learnerId,
                        eventType: .parametersRefit,
                        numericValue: absErr,
                        payloadJSON: "{\"strong\":\(refitParams.strongSuccessMultiplier),\"weak\":\(refitParams.weakSuccessMultiplier),\"fail\":\(refitParams.failureMultiplier)}"
                    ))
                }
            }

            let result = WeeklyAuditResult(
                id: UUID().uuidString,
                week: prof.currentWeek,
                completedAt: Date(),
                itemCount: responses.count,
                meanPredictedRecall: meanPred,
                meanObservedRecall: meanObs,
                absoluteCalibrationError: absErr,
                exceededTenPointThreshold: absErr > 0.10,
                didRefitModel: didRefit,
                durationSeconds: Date().timeIntervalSince(startedAt),
                responses: responses
            )
            try await appState.database.saveWeeklyAudit(result, learnerId: prof.learnerId)
            try? await appState.database.appendEvidence(EvidenceEvent(
                learnerId: prof.learnerId,
                eventType: .weeklyAuditCompleted,
                numericValue: absErr
            ))
            try? await appState.refreshLearnerData()
            completedResult = result
        } catch {
            errorMsg = error.localizedDescription
        }
    }

    private func scorePill(_ label: String, _ value: String, _ color: Color) -> some View {
        VStack(spacing: 4) {
            Text(label).font(CadenciaFont.mono(9, weight: .bold)).foregroundStyle(color)
            Text(value).font(CadenciaFont.displaySerif(24, weight: .bold)).foregroundStyle(CadenciaPalette.ivoryText)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .cadenciaGlassCard(cornerRadius: 16, tint: color)
    }
}

// MARK: - 2. Vocabulary Size Test + Hold-Out >15% Guard

private struct VocabSizeTestSheet: View {
    @Bindable var appState: AppState
    @Environment(\.dismiss) private var dismiss

    private enum ProbeKind { case real, pseudo, holdout }
    private struct ProbeCard: Identifiable {
        let id = UUID()
        let word: String
        let kind: ProbeKind
    }

    @State private var checkpointLabel: String = "Hour 20"
    @State private var cards: [ProbeCard] = []
    @State private var index: Int = 0
    @State private var realYes = 0
    @State private var pseudoYes = 0
    @State private var holdoutYes = 0
    @State private var result: VocabSizeTestResult?

    var body: some View {
        NavigationStack {
            ZStack {
                CadenciaPalette.obsidian.ignoresSafeArea()
                VStack(spacing: 22) {
                    if let res = result {
                        VStack(spacing: 16) {
                            Text("Estimated Vocabulary Size")
                                .font(CadenciaFont.mono(11, weight: .bold))
                                .foregroundStyle(CadenciaPalette.mediterraneanJade)
                            Text("~\(res.correctedEstimatedVocabSize) words")
                                .font(CadenciaFont.displaySerif(42, weight: .bold))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                            Text("False alarms: \(res.pseudowordsFalseAlarms)/\(res.pseudowordsTested) · Hold-out corrected rate: \(Int((res.holdoutCorrectedClaimRate * 100).rounded()))%")
                                .font(CadenciaFont.mono(12))
                                .foregroundStyle(CadenciaPalette.mutedSilver)

                            if res.isInflatedWarning {
                                Label("HOLD-OUT INFLATION GUARD TRIGGERED (>15%) — scores may reflect guessing or outside exposure (PDF p. 13).", systemImage: "exclamationmark.triangle.fill")
                                    .font(CadenciaFont.uiCaption(13))
                                    .foregroundStyle(CadenciaPalette.crimsonAlert)
                                    .padding()
                                    .cadenciaGlassCard(cornerRadius: 16, tint: CadenciaPalette.crimsonAlert)
                            }

                            Button("Save & Close") { dismiss() }
                                .font(CadenciaFont.uiTitle(16, weight: .bold))
                                .foregroundStyle(CadenciaPalette.obsidian)
                                .padding(.horizontal, 36)
                                .padding(.vertical, 14)
                                .background(CadenciaPalette.mediterraneanJade, in: Capsule())
                        }
                        .padding(24)
                    } else if index < cards.count {
                        let card = cards[index]
                        VStack(spacing: 24) {
                            Picker("Checkpoint", selection: $checkpointLabel) {
                                ForEach(["Baseline", "Hour 20", "Hour 40", "Hour 60", "8-Week Retest"], id: \.self) {
                                    Text($0).tag($0)
                                }
                            }
                            .pickerStyle(.segmented)

                            ProgressView(value: Double(index + 1), total: Double(cards.count))
                                .tint(CadenciaPalette.mediterraneanJade)

                            Text("WORD \(index + 1) OF \(cards.count)")
                                .font(CadenciaFont.mono(11, weight: .bold))
                                .foregroundStyle(CadenciaPalette.mutedSilver)

                            Text(card.word)
                                .font(CadenciaFont.displaySerif(44, weight: .bold))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                                .padding(.vertical, 24)

                            Text("Do you know the meaning of this Spanish word? (Real words, pseudowords, and 400-word hold-out controls are mixed.)")
                                .font(CadenciaFont.uiCaption(12))
                                .foregroundStyle(CadenciaPalette.mutedSilver)
                                .multilineTextAlignment(.center)

                            HStack(spacing: 16) {
                                Button {
                                    recordAnswer(yes: false)
                                } label: {
                                    Text("No / Unsure")
                                        .font(CadenciaFont.uiTitle(16, weight: .semibold))
                                        .foregroundStyle(CadenciaPalette.ivoryText)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 16)
                                        .cadenciaGlassPill(tint: CadenciaPalette.terracotta)
                                }

                                Button {
                                    recordAnswer(yes: true)
                                } label: {
                                    Text("Yes, I know it")
                                        .font(CadenciaFont.uiTitle(16, weight: .bold))
                                        .foregroundStyle(CadenciaPalette.obsidian)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 16)
                                        .background(CadenciaPalette.mediterraneanJade, in: Capsule())
                                }
                            }
                        }
                        .padding(24)
                    }
                }
            }
            .navigationTitle("Vocabulary Size & Hold-Out Guard")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
        }
        .task {
            let reals = appState.allSyllabusItems.filter { $0.itemType == .lemma }.shuffled().prefix(60).map { ProbeCard(word: $0.spanish, kind: .real) }
            let pseudos = ((try? await appState.database.loadPseudowords()) ?? []).shuffled().prefix(30).map { ProbeCard(word: $0, kind: .pseudo) }
            let holdouts = appState.holdoutItems.shuffled().prefix(30).map { ProbeCard(word: $0.spanish, kind: .holdout) }
            cards = (reals + pseudos + holdouts).shuffled()
        }
    }

    private func recordAnswer(yes: Bool) {
        let card = cards[index]
        if yes {
            switch card.kind {
            case .real: realYes += 1
            case .pseudo: pseudoYes += 1
            case .holdout: holdoutYes += 1
            }
        }
        HapticDirector.shared.playGlossTap()
        if index + 1 < cards.count {
            index += 1
        } else {
            let rTot = cards.filter { $0.kind == .real }.count
            let pTot = cards.filter { $0.kind == .pseudo }.count
            let hTot = cards.filter { $0.kind == .holdout }.count
            let scored = appState.analyticsEngine.scoreVocabSizeTest(
                checkpointLabel: checkpointLabel,
                realWordsTested: rTot,
                realWordsClaimed: realYes,
                pseudowordsTested: pTot,
                pseudowordsFalseAlarms: pseudoYes,
                holdoutWordsTested: hTot,
                holdoutWordsClaimed: holdoutYes,
                typedRecall100Score: nil
            )
            result = scored
            if let lid = appState.profile?.learnerId {
                Task {
                    try? await appState.database.saveResearchAssessment(
                        id: scored.id, learnerId: lid, type: "vocab_size_yes_no",
                        checkpointLabel: checkpointLabel, primaryScore: Double(scored.correctedEstimatedVocabSize), payload: scored
                    )
                    try? await appState.database.appendEvidence(EvidenceEvent(
                        learnerId: lid, eventType: .vocabSizeCheck, numericValue: Double(scored.correctedEstimatedVocabSize)
                    ))
                }
            }
        }
    }
}

// MARK: - 3. Spanish Elicited Imitation Task (30 Ortega Sentences, 0–4 Scoring)

private struct SpanishEITRunnerSheet: View {
    @Bindable var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var prompts: [EITPrompt] = []
    @State private var index: Int = 0
    @State private var hasPlayedCurrent: Bool = false
    @State private var isPlaying: Bool = false
    @State private var isRecording: Bool = false
    @State private var isScoring: Bool = false
    @State private var scores: [EITItemScore] = []
    @State private var sessionResult: EITSessionResult?
    @State private var statusLine: String = "Listen once, then repeat aloud in Spanish."

    var body: some View {
        NavigationStack {
            ZStack {
                CadenciaPalette.obsidian.ignoresSafeArea()
                VStack(spacing: 22) {
                    if let res = sessionResult {
                        VStack(spacing: 14) {
                            Text("SPANISH EIT COMPLETE")
                                .font(CadenciaFont.mono(11, weight: .bold))
                                .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                            Text("\(res.totalScore) / \(res.maxScore)")
                                .font(CadenciaFont.displaySerif(44, weight: .bold))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                            Text(String(format: "%.1f%% Ortega Standardized Score", res.normalizedPercent))
                                .font(CadenciaFont.mono(13))
                                .foregroundStyle(CadenciaPalette.saffronGold)
                            Button("Done") { dismiss() }
                                .font(CadenciaFont.uiTitle(16, weight: .bold))
                                .foregroundStyle(CadenciaPalette.obsidian)
                                .padding(.horizontal, 36)
                                .padding(.vertical, 14)
                                .background(CadenciaPalette.ivoryText, in: Capsule())
                        }
                    } else if index < prompts.count {
                        let p = prompts[index]
                        VStack(spacing: 18) {
                            ProgressView(value: Double(index + 1), total: Double(prompts.count))
                                .tint(CadenciaPalette.bioluminescentCobalt)
                            Text("SENTENCE \(index + 1) / \(prompts.count) · \(p.syllables) SYLLABLES · \(p.cefr.rawValue)")
                                .font(CadenciaFont.mono(11, weight: .bold))
                                .foregroundStyle(CadenciaPalette.bioluminescentCobalt)

                            ResonanceOrbView(
                                mode: isRecording ? .listening : (isPlaying ? .speaking : (isScoring ? .thinking : .idle)),
                                energy: appState.liveAudioEnergy
                            )
                            .frame(width: 190, height: 190)

                            Text(statusLine)
                                .font(CadenciaFont.uiCaption(14))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                                .multilineTextAlignment(.center)

                            HStack(spacing: 14) {
                                Button {
                                    Task { await playPromptOnce(p) }
                                } label: {
                                    Label(hasPlayedCurrent ? "Played" : "Play Once", systemImage: "play.fill")
                                        .font(CadenciaFont.uiTitle(15, weight: .bold))
                                        .foregroundStyle(CadenciaPalette.obsidian)
                                        .padding(.horizontal, 20)
                                        .padding(.vertical, 14)
                                        .background(hasPlayedCurrent ? CadenciaPalette.mutedSilver : CadenciaPalette.saffronGold, in: Capsule())
                                }
                                .disabled(isPlaying || isRecording || hasPlayedCurrent)

                                Button {
                                    Task { await toggleRecordAndScore(p) }
                                } label: {
                                    Label(isRecording ? "Stop & Score" : "Repeat Aloud", systemImage: isRecording ? "stop.fill" : "mic.fill")
                                        .font(CadenciaFont.uiTitle(15, weight: .bold))
                                        .foregroundStyle(CadenciaPalette.obsidian)
                                        .padding(.horizontal, 20)
                                        .padding(.vertical, 14)
                                        .background(isRecording ? CadenciaPalette.crimsonAlert : CadenciaPalette.mediterraneanJade, in: Capsule())
                                }
                                .disabled(isPlaying || isScoring)
                            }

                            if let last = scores.last {
                                Text("Last item score: \(last.score0To4)/4 · Heard: «\(last.learnerTranscript)»")
                                    .font(CadenciaFont.mono(11))
                                    .foregroundStyle(CadenciaPalette.mutedSilver)
                                    .lineLimit(2)
                            }
                        }
                        .padding(24)
                    }
                }
            }
            .navigationTitle("Spanish EIT (30 Items)")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
        }
        .task {
            prompts = (try? await appState.database.loadEITPrompts()) ?? []
        }
    }

    private func playPromptOnce(_ prompt: EITPrompt) async {
        isPlaying = true
        statusLine = "Playing sentence via Magpie v2607…"
        defer { isPlaying = false; hasPlayedCurrent = true }
        if await appState.magpieEngine.isLoaded,
           let clip = try? await appState.magpieEngine.synthesizeSentence(text: prompt.spanish, voice: .sofia, accent: .castilian) {
            try? await appState.magpieEngine.play(clip: clip, targetWPM: 130)
            statusLine = "Now tap Repeat Aloud and say the exact sentence."
        } else {
            statusLine = "Magpie not installed — read once & repeat: «\(prompt.spanish)»"
        }
    }

    private func toggleRecordAndScore(_ prompt: EITPrompt) async {
        if !isRecording {
            do {
                try await appState.speechRecognizer.startListening()
                isRecording = true
                statusLine = "Listening… speak clearly then tap Stop & Score."
            } catch {
                statusLine = error.localizedDescription
            }
        } else {
            isRecording = false
            isScoring = true
            statusLine = "Scoring with Parakeet Ultra INT8 + ZIPA CR Large FP16…"
            defer { isScoring = false }
            do {
                let rec = try await appState.speechRecognizer.stopAndRecognize(expectedSpanishText: prompt.spanish, accent: .castilian)
                let itemScore = appState.analyticsEngine.scoreEITResponse(
                    prompt: prompt,
                    learnerTranscript: rec.transcriptEs,
                    learnerIPA: rec.learnerIPA,
                    phonemeAccuracy: rec.phonemeAccuracy
                )
                scores.append(itemScore)
                if let lid = appState.profile?.learnerId {
                    try? await appState.database.appendEvidence(EvidenceEvent(
                        learnerId: lid, eventType: .eitItemScored, numericValue: Double(itemScore.score0To4)
                    ))
                }
                hasPlayedCurrent = false
                if index + 1 < prompts.count {
                    index += 1
                    statusLine = "Sentence \(index + 1) ready. Tap Play Once."
                } else {
                    let tot = scores.reduce(0) { $0 + $1.score0To4 }
                    let mx = prompts.count * 4
                    let res = EITSessionResult(
                        id: UUID().uuidString,
                        checkpointLabel: "Session \(appState.profile?.currentSessionNumber ?? 0)",
                        completedAt: Date(),
                        totalScore: tot,
                        maxScore: mx,
                        normalizedPercent: Double(tot) / Double(max(1, mx)) * 100.0,
                        items: scores
                    )
                    sessionResult = res
                    if let lid = appState.profile?.learnerId {
                        try? await appState.database.saveResearchAssessment(
                            id: res.id, learnerId: lid, type: "spanish_eit",
                            checkpointLabel: res.checkpointLabel, primaryScore: res.normalizedPercent, payload: res
                        )
                    }
                }
            } catch {
                statusLine = error.localizedDescription
            }
        }
    }
}

// MARK: - 4. Native-Speed Listening Test (Hours 30 & 60)

private struct NativeListeningTestSheet: View {
    @Bindable var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var passages: [NativeListeningPassage] = []
    @State private var selectedIdx: Int = 0
    @State private var isPlaying: Bool = false
    @State private var showTranscript: Bool = false
    @State private var answers: [Int: Int] = [:]
    @State private var submittedScore: String?

    var body: some View {
        NavigationStack {
            ZStack {
                CadenciaPalette.obsidian.ignoresSafeArea()
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        if !passages.isEmpty {
                            let p = passages[selectedIdx]
                            Picker("Passage", selection: $selectedIdx) {
                                ForEach(Array(passages.enumerated()), id: \.offset) { i, item in
                                    Text("H\(item.checkpointHour): \(item.title)").tag(i)
                                }
                            }
                            .pickerStyle(.menu)
                            .tint(CadenciaPalette.saffronGold)
                            .onChange(of: selectedIdx) { _, _ in
                                answers.removeAll(); submittedScore = nil; showTranscript = false
                            }

                            VStack(alignment: .leading, spacing: 10) {
                                Text("HOUR \(p.checkpointHour) · \(p.accent.displayName.uppercased()) · \(p.wpm) WPM")
                                    .font(CadenciaFont.mono(10, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.terracotta)
                                Text(p.title)
                                    .font(CadenciaFont.displaySerif(24, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.ivoryText)

                                HStack(spacing: 12) {
                                    Button {
                                        Task {
                                            guard await appState.magpieEngine.isLoaded else { return }
                                            isPlaying = true
                                            defer { isPlaying = false }
                                            let sentences = p.transcript.components(separatedBy: ". ").filter { !$0.isEmpty }
                                            if let clip = try? await appState.magpieEngine.synthesizePassage(sentences: sentences, voice: p.speaker, accent: p.accent) {
                                                try? await appState.magpieEngine.play(clip: clip, targetWPM: p.wpm)
                                            }
                                        }
                                    } label: {
                                        Label(isPlaying ? "Playing Native Audio…" : "Play at \(p.wpm) WPM", systemImage: "headphones")
                                            .font(CadenciaFont.uiTitle(14, weight: .bold))
                                            .foregroundStyle(CadenciaPalette.obsidian)
                                            .padding(.horizontal, 16)
                                            .padding(.vertical, 11)
                                            .background(CadenciaPalette.saffronGold, in: Capsule())
                                    }
                                    .disabled(isPlaying)

                                    Button(showTranscript ? "Hide Script" : "Reveal Script") {
                                        showTranscript.toggle()
                                    }
                                    .font(CadenciaFont.mono(11, weight: .semibold))
                                    .foregroundStyle(CadenciaPalette.mutedSilver)
                                }

                                if showTranscript {
                                    Text(p.transcript)
                                        .font(CadenciaFont.proseSerif(16))
                                        .foregroundStyle(CadenciaPalette.ivoryText.opacity(0.9))
                                        .padding(.top, 6)
                                }
                            }
                            .padding(16)
                            .cadenciaGlassCard(cornerRadius: 20)

                            ForEach(Array(p.questions.enumerated()), id: \.offset) { qIdx, q in
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("\(qIdx + 1). \(q.q)")
                                        .font(CadenciaFont.uiTitle(15, weight: .semibold))
                                        .foregroundStyle(CadenciaPalette.ivoryText)
                                    ForEach(Array(q.options.enumerated()), id: \.offset) { oIdx, opt in
                                        let picked = answers[qIdx] == oIdx
                                        Button {
                                            answers[qIdx] = oIdx
                                        } label: {
                                            HStack {
                                                Image(systemName: picked ? "largecircle.fill.circle" : "circle")
                                                    .foregroundStyle(picked ? CadenciaPalette.saffronGold : CadenciaPalette.mutedSilver)
                                                Text(opt)
                                                    .font(CadenciaFont.uiCaption(14))
                                                    .foregroundStyle(CadenciaPalette.ivoryText)
                                                Spacer()
                                            }
                                            .padding(10)
                                            .background(picked ? CadenciaPalette.saffronGold.opacity(0.16) : Color.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
                                        }
                                        .buttonStyle(.plain)
                                    }
                                }
                                .padding(14)
                                .cadenciaGlassCard(cornerRadius: 18)
                            }

                            Button("Submit Listening Score") {
                                let correct = p.questions.enumerated().filter { answers[$0.offset] == $0.element.correct }.count
                                let pct = Double(correct) / Double(max(1, p.questions.count)) * 100.0
                                submittedScore = "Score: \(correct)/\(p.questions.count) (\(Int(pct.rounded()))%)"
                                if let lid = appState.profile?.learnerId {
                                    Task {
                                        try? await appState.database.appendEvidence(EvidenceEvent(
                                            learnerId: lid, eventType: .nativeListeningAnswered, numericValue: pct
                                        ))
                                    }
                                }
                            }
                            .font(CadenciaFont.uiTitle(15, weight: .bold))
                            .foregroundStyle(CadenciaPalette.obsidian)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .background(CadenciaPalette.mediterraneanJade, in: Capsule())

                            if let s = submittedScore {
                                Text(s)
                                    .font(CadenciaFont.mono(13, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.mediterraneanJade)
                            }
                        }
                    }
                    .padding(18)
                }
            }
            .navigationTitle("Native-Speed Listening")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
        }
        .task {
            passages = (try? await appState.database.loadNativeListeningPassages()) ?? []
        }
    }
}

// MARK: - 5. Baseline 10-Min Speaking Recorder (Solo Pilot Step 2)

private struct BaselineSpeakingRecorderSheet: View {
    @Bindable var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var isRecording: Bool = false
    @State private var elapsedSeconds: Int = 0
    @State private var timerTask: Task<Void, Never>?
    @State private var lastOutput: DualRecognitionOutput?
    @State private var status: String = "Speak freely in Spanish for up to 10 minutes (picture description + personal routine + opinion) for your blind baseline recording."

    var body: some View {
        NavigationStack {
            ZStack {
                CadenciaPalette.obsidian.ignoresSafeArea()
                VStack(spacing: 22) {
                    Text(String(format: "%02d:%02d / 10:00", elapsedSeconds / 60, elapsedSeconds % 60))
                        .font(CadenciaFont.mono(34, weight: .bold))
                        .foregroundStyle(CadenciaPalette.saffronGold)

                    ResonanceOrbView(mode: isRecording ? .listening : .idle, energy: appState.liveAudioEnergy)
                        .frame(width: 200, height: 200)

                    Text(status)
                        .font(CadenciaFont.uiCaption(13))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)

                    Button {
                        Task { await toggle() }
                    } label: {
                        Label(isRecording ? "Stop & Transcribe Baseline" : "Start Baseline Recording", systemImage: isRecording ? "stop.fill" : "mic.fill")
                            .font(CadenciaFont.uiTitle(16, weight: .bold))
                            .foregroundStyle(CadenciaPalette.obsidian)
                            .padding(.horizontal, 28)
                            .padding(.vertical, 16)
                            .background(isRecording ? CadenciaPalette.crimsonAlert : CadenciaPalette.saffronGold, in: Capsule())
                    }

                    if let out = lastOutput {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("TRANSCRIPT (PARAKEET ULTRA INT8)")
                                    .font(CadenciaFont.mono(10, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.mediterraneanJade)
                                Text(out.transcriptEs)
                                    .font(CadenciaFont.proseSerif(16))
                                    .foregroundStyle(CadenciaPalette.ivoryText)
                                Text("IPA (ZIPA 300M FP16): /\(out.learnerIPA)/")
                                    .font(CadenciaFont.mono(11))
                                    .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                            }
                            .padding(14)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .cadenciaGlassCard(cornerRadius: 16)
                        }
                        .frame(maxHeight: 200)
                    }
                }
                .padding(24)
            }
            .navigationTitle("Baseline Speaking")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
        }
        .onDisappear { timerTask?.cancel() }
    }

    private func toggle() async {
        if !isRecording {
            do {
                try await appState.speechRecognizer.startListening()
                isRecording = true
                elapsedSeconds = 0
                status = "Recording baseline Spanish sample…"
                timerTask = Task {
                    while !Task.isCancelled {
                        try? await Task.sleep(nanoseconds: 1_000_000_000)
                        await MainActor.run { elapsedSeconds += 1 }
                    }
                }
            } catch { status = error.localizedDescription }
        } else {
            timerTask?.cancel()
            isRecording = false
            status = "Running Parakeet Ultra INT8 + ZIPA CR Large FP16…"
            do {
                let res = try await appState.speechRecognizer.stopAndRecognize()
                lastOutput = res
                status = String(format: "Saved baseline (%.0f ms Parakeet · %.0f ms ZIPA)", res.parakeetLatencyMs, res.zipaLatencyMs)
                if let lid = appState.profile?.learnerId {
                    try? await appState.database.appendEvidence(EvidenceEvent(
                        learnerId: lid, eventType: .spokenUtterance, numericValue: res.phonemeAccuracy,
                        payloadJSON: "{\"baseline\":true,\"durationSec\":\(elapsedSeconds)}"
                    ))
                }
            } catch { status = error.localizedDescription }
        }
    }
}

// MARK: - 6. In-App Stage 0 Bench Test Runner

private struct Stage0BenchTestSheet: View {
    @Bindable var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var passagesPerHour: Int = 2
    @State private var isRunning: Bool = false
    @State private var progressLog: String = "Tests GPT-6.1 Terra Generator + Stage A/B Verifier across simulated learner states at hours 2, 10, 20, 35, and 50 (Stage 0 Gate: ≥95% pass within 3 rewrites, ≥20% scheduled share)."
    @State private var summary: BenchTestSummary?

    var body: some View {
        NavigationStack {
            ZStack {
                CadenciaPalette.obsidian.ignoresSafeArea()
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        Stepper("Passages per simulated hour: \(passagesPerHour) (Total: \(passagesPerHour * 5))", value: $passagesPerHour, in: 1...20)
                            .font(CadenciaFont.uiCaption(14, weight: .semibold))
                            .foregroundStyle(CadenciaPalette.ivoryText)
                            .padding(14)
                            .cadenciaGlassCard(cornerRadius: 16)

                        Button {
                            Task { await runBench() }
                        } label: {
                            HStack {
                                if isRunning { ProgressView().tint(CadenciaPalette.obsidian) }
                                Text(isRunning ? "Running Stage 0 Bench Test…" : "Run Live Stage 0 Bench Test")
                                    .font(CadenciaFont.uiTitle(15, weight: .bold))
                            }
                            .foregroundStyle(CadenciaPalette.obsidian)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .background(CadenciaPalette.bioluminescentCobalt, in: Capsule())
                        }
                        .disabled(isRunning)

                        Text(progressLog)
                            .font(CadenciaFont.mono(11))
                            .foregroundStyle(CadenciaPalette.mutedSilver)
                            .padding(14)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 14))

                        if let sum = summary {
                            VStack(alignment: .leading, spacing: 10) {
                                HStack {
                                    Text(sum.gatePassed ? "STAGE 0 GATE PASSED" : "STAGE 0 GATE FAILED")
                                        .font(CadenciaFont.mono(12, weight: .bold))
                                        .foregroundStyle(sum.gatePassed ? CadenciaPalette.mediterraneanJade : CadenciaPalette.crimsonAlert)
                                    Spacer()
                                    Text(String(format: "Pass %.0f%% · Share %.1f%%", sum.overallPassRate * 100, sum.overallScheduledShare * 100))
                                        .font(CadenciaFont.mono(11, weight: .bold))
                                        .foregroundStyle(CadenciaPalette.ivoryText)
                                }
                                ForEach(sum.perHourStates) { st in
                                    HStack {
                                        Text("Hour \(st.simulatedHour) (\(st.knownCount) known)")
                                            .font(CadenciaFont.mono(11))
                                            .foregroundStyle(CadenciaPalette.ivoryText)
                                        Spacer()
                                        Text(String(format: "%d/%d pass · %.1f%% share", st.passedWithinThreeRewrites, st.passagesAttempted, st.averageScheduledShare * 100))
                                            .font(CadenciaFont.mono(11))
                                            .foregroundStyle(CadenciaPalette.saffronGold)
                                    }
                                }
                            }
                            .padding(16)
                            .cadenciaGlassCard(cornerRadius: 18, tint: sum.gatePassed ? CadenciaPalette.mediterraneanJade : CadenciaPalette.crimsonAlert)
                        }
                    }
                    .padding(18)
                }
            }
            .navigationTitle("Stage 0 Bench Test")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
            }
        }
    }

    private func runBench() async {
        isRunning = true
        defer { isRunning = false }
        await appState.ensureLinguisticResourcesLoaded()

        let hours = [2, 10, 20, 35, 50]
        let scheduler = SessionScheduler(parameters: .protocolDefaults)
        let verifier = ContractVerifier(lemmatizer: appState.lemmatizer, parameters: .protocolDefaults)
        let serial = appState.serialGenres.first ?? SerialGenre(
            id: "madrid_noir", titleEs: "La Sombra del Retiro", titleEn: "The Shadow of Retiro",
            tagline: "", setting: "Madrid", premise: "", accentPalette: "", protagonist: "Clara"
        )

        var reports: [BenchTestStateReport] = []
        var totalPassed = 0
        var totalAttempted = 0
        var shareSum = 0.0

        for h in hours {
            let (simProfile, simRecs) = appState.analyticsEngine.buildSimulatedLearnerState(atHour: h, allItems: appState.allSyllabusItems)
            var passed = 0
            var rewritesSum = 0
            var hourShareSum = 0.0
            var closedSum = 0.0

            for pIdx in 0..<passagesPerHour {
                progressLog = "Simulating Hour \(h) — Passage \(pIdx + 1)/\(passagesPerHour) via GPT-6.1 Terra…"
                let plan = scheduler.planNextPassage(
                    profile: simProfile,
                    blockType: .input1,
                    passageIndexInSession: pIdx,
                    newItemsRemainingInSession: 60,
                    allItems: appState.allSyllabusItems,
                    holdoutItems: appState.holdoutItems,
                    records: simRecs,
                    priorContextByItem: [:],
                    serialGenre: serial,
                    serialEpisodeNumber: pIdx,
                    storySummarySoFar: ""
                )
                totalAttempted += 1
                do {
                    let (passage, rep) = try await appState.terraClient.generateVerifiedPassage(
                        plan: plan, verifier: verifier, priorContextByItem: [:]
                    )
                    if rep.passed && passage.rewriteAttempts <= 3 { passed += 1; totalPassed += 1 }
                    rewritesSum += passage.rewriteAttempts
                    hourShareSum += passage.scheduledWordShare
                    shareSum += passage.scheduledWordShare
                    closedSum += rep.closedVocabCompliance
                } catch {
                    progressLog = "Hour \(h) P\(pIdx + 1) failed: \(error.localizedDescription)"
                }
            }

            reports.append(BenchTestStateReport(
                simulatedHour: h,
                knownCount: simRecs.values.filter { $0.meetingsCount > 0 }.count,
                passagesAttempted: passagesPerHour,
                passedWithinThreeRewrites: passed,
                averageRewrites: Double(rewritesSum) / Double(max(1, passagesPerHour)),
                averageScheduledShare: hourShareSum / Double(max(1, passagesPerHour)),
                averageClosedVocabCompliance: closedSum / Double(max(1, passagesPerHour))
            ))
        }

        let overallPass = Double(totalPassed) / Double(max(1, totalAttempted))
        let overallShare = shareSum / Double(max(1, totalAttempted))
        let sum = BenchTestSummary(
            runAt: Date(),
            totalPassages: totalAttempted,
            overallPassRate: overallPass,
            overallScheduledShare: overallShare,
            gatePassed: overallPass >= 0.95 && overallShare >= 0.20,
            perHourStates: reports
        )
        summary = sum
        progressLog = "Bench test finished: \(totalPassed)/\(totalAttempted) passed."
        if let lid = appState.profile?.learnerId {
            try? await appState.database.appendEvidence(EvidenceEvent(
                learnerId: lid, eventType: .benchTestRun, numericValue: overallPass
            ))
        }
    }
}
