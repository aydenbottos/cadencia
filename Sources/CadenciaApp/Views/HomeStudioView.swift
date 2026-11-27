import CadenciaCore
import SwiftUI

/// "Studio" — Primary Home Dashboard displaying today's 75-minute session hero card,
/// live GPU Memory Constellation preview, real-time Signal Density Gauge (Claim 1: >=15 events/min),
/// due-item urgency counter, and weekly pace/intake trajectory.
public struct HomeStudioView: View {
    @Bindable var appState: AppState
    @Binding var selectedTab: Int

    public var body: some View {
        ZStack {
            MediterraneanAuroraView(mood: .studio, audioEnergy: appState.liveAudioEnergy)
                .ignoresSafeArea()

            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 22) {
                    studioHeader
                    todaySessionHeroCard
                    liveMetricsGrid
                    constellationPreviewCard
                    weeklyTrajectoryCard
                }
                .padding(.horizontal, 20)
                .padding(.top, 14)
                .padding(.bottom, 110)
            }
        }
    }

    private var studioHeader: some View {
        HStack(alignment: .bottom) {
            VStack(alignment: .leading, spacing: 4) {
                Text(appState.profile?.studyMode.title.uppercased() ?? "CADENCIA")
                    .font(CadenciaFont.mono(11, weight: .bold))
                    .tracking(2.2)
                    .foregroundStyle(CadenciaPalette.saffronGold)
                Text("Studio")
                    .font(CadenciaFont.displaySerif(36, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
            }
            Spacer()
            if let profile = appState.profile {
                VStack(alignment: .trailing, spacing: 2) {
                    Text(String(format: "%.1f / 60.0 h", profile.totalLoggedSeconds / 3600.0))
                        .font(CadenciaFont.mono(14, weight: .bold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                    Text("WEEK \(profile.currentWeek) · SESSION \(profile.currentSessionNumber)/48")
                        .font(CadenciaFont.mono(10))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .cadenciaGlassPill(interactive: false)
            }
        }
    }

    private var todaySessionHeroCard: some View {
        let profile = appState.profile
        let week = profile?.currentWeek ?? 1
        let spec = WeeklyPlanSpec.forWeek(week)
        let scheduler = SessionScheduler(parameters: profile?.parameters ?? .protocolDefaults)
        let newBudget = profile.map {
            scheduler.targetNewItemsForSession(
                sessionNumber: $0.currentSessionNumber,
                studyMode: $0.studyMode,
                learnerSeed: abs($0.learnerId.hashValue)
            )
        } ?? 60
        let isBootstrap = profile?.isBootstrapSession ?? true

        return VStack(alignment: .leading, spacing: 18) {
            HStack {
                Label(
                    isBootstrap ? "Bootstrap Session \(profile?.currentSessionNumber ?? 1)/4" : "Session \(profile?.currentSessionNumber ?? 1) of 48",
                    systemImage: "flame.fill"
                )
                .font(CadenciaFont.mono(12, weight: .bold))
                .foregroundStyle(CadenciaPalette.saffronGold)
                Spacer()
                Text("75 MIN · 50 IN / 25 OUT")
                    .font(CadenciaFont.mono(11, weight: .semibold))
                    .foregroundStyle(CadenciaPalette.mutedSilver)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text(appState.currentSerialGenre?.titleEs ?? "La Sombra del Retiro")
                    .font(CadenciaFont.displaySerif(26, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                Text(isBootstrap
                     ? "Short <=8-word illustrated bootstrap sentences introducing high-frequency function words & transparent cognates."
                     : "High-density passages at \(spec.paceWPM) WPM · \(Int(spec.audioFirstFraction * 100))% audio-first · \(newBudget) new items + \(appState.dueNowCount) due.")
                    .font(.subheadline)
                    .foregroundStyle(CadenciaPalette.mutedSilver)
                    .lineSpacing(3)
            }

            HStack(spacing: 10) {
                pillBadge(icon: "waveform", text: "\(spec.paceWPM) WPM", color: CadenciaPalette.bioluminescentCobalt)
                pillBadge(icon: "sparkles", text: "+\(newBudget) new", color: CadenciaPalette.saffronGold)
                pillBadge(icon: "clock.arrow.circlepath", text: "\(appState.dueNowCount) due", color: CadenciaPalette.terracotta)
            }

            Button {
                HapticDirector.shared.playStrongSuccess()
                appState.startTodaySession()
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "play.fill")
                    Text("Begin 75-Minute Immersion")
                        .font(.headline)
                    Spacer()
                    Image(systemName: "arrow.right")
                }
                .foregroundStyle(CadenciaPalette.obsidian)
                .padding(.horizontal, 22)
                .padding(.vertical, 16)
                .background(
                    LinearGradient(
                        colors: [CadenciaPalette.saffronGold, CadenciaPalette.terracotta],
                        startPoint: .leading,
                        endPoint: .trailing
                    ),
                    in: Capsule()
                )
            }
            .buttonStyle(.plain)
        }
        .padding(22)
        .cadenciaGlassCard(cornerRadius: 28, tint: CadenciaPalette.terracotta, interactive: false)
    }

    private var liveMetricsGrid: some View {
        let tel = appState.telemetry
        let density = tel?.usableEventsPerInputMinute ?? 0
        let probeAcc = (tel?.comprehensionProbeAccuracy ?? 1.0) * 100
        let knownTotal = (tel?.stateCounts[.known] ?? 0) + (tel?.stateCounts[.productive] ?? 0)
        let prod2000 = tel?.top2000ProductiveCount ?? 0

        return LazyVGrid(columns: [GridItem(.flexible(), spacing: 14), GridItem(.flexible(), spacing: 14)], spacing: 14) {
            metricTile(
                label: "SIGNAL DENSITY",
                value: String(format: "%.1f", density),
                unit: "/ min",
                target: "Claim 1 target ≥ 15.0",
                accent: density >= 15.0 ? CadenciaPalette.mediterraneanJade : CadenciaPalette.saffronGold
            )
            metricTile(
                label: "PROBE ACCURACY",
                value: String(format: "%.0f%%", probeAcc),
                unit: "",
                target: "Comprehension ≥ 85%",
                accent: probeAcc >= 85.0 ? CadenciaPalette.bioluminescentCobalt : CadenciaPalette.crimsonAlert
            )
            metricTile(
                label: "KNOWN & PRODUCTIVE",
                value: "\(knownTotal)",
                unit: "/ 4,500",
                target: "h ≥ 7d · ≥4 hits · ≥1 ear",
                accent: CadenciaPalette.bioluminescentCobalt
            )
            metricTile(
                label: "OUTPUT COVERAGE",
                value: "\(prod2000)",
                unit: "/ 2,000",
                target: "Top 2k lemmas used ≥2×",
                accent: CadenciaPalette.saffronGold
            )
        }
    }

    private var constellationPreviewCard: some View {
        Button {
            HapticDirector.shared.playGlossTap()
            selectedTab = 1
        } label: {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("MEMORY CONSTELLATION")
                            .font(CadenciaFont.mono(11, weight: .bold))
                            .foregroundStyle(CadenciaPalette.saffronGold)
                        Text("4,500 Spanish Items Live Half-Life Map")
                            .font(CadenciaFont.displaySerif(18, weight: .semibold))
                            .foregroundStyle(CadenciaPalette.ivoryText)
                    }
                    Spacer()
                    Image(systemName: "arrow.up.right.circle.fill")
                        .font(.title2)
                        .foregroundStyle(CadenciaPalette.saffronGold)
                }

                MemoryConstellationMetalView(stars: appState.constellationStars, zoom: 1.05, pan: .zero)
                    .frame(height: 175)
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
                    )

                HStack(spacing: 14) {
                    legendDot(color: CadenciaPalette.mutedSilver.opacity(0.5), label: "New")
                    legendDot(color: CadenciaPalette.terracotta, label: "Learning")
                    legendDot(color: CadenciaPalette.bioluminescentCobalt, label: "Known")
                    legendDot(color: CadenciaPalette.saffronGold, label: "Productive")
                }
                .font(CadenciaFont.mono(10))
                .foregroundStyle(CadenciaPalette.mutedSilver)
            }
            .padding(20)
            .cadenciaGlassCard(cornerRadius: 24, interactive: true)
        }
        .buttonStyle(.plain)
    }

    private var weeklyTrajectoryCard: some View {
        let currentWeek = appState.profile?.currentWeek ?? 1
        return VStack(alignment: .leading, spacing: 14) {
            Text("8-WEEK PROTOCOL SCHEDULE")
                .font(CadenciaFont.mono(11, weight: .bold))
                .foregroundStyle(CadenciaPalette.saffronGold)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(WeeklyPlanSpec.eightWeekPlan, id: \.week) { spec in
                        let isCurrent = spec.week == currentWeek
                        VStack(alignment: .leading, spacing: 6) {
                            Text("WK \(spec.week)")
                                .font(CadenciaFont.mono(11, weight: .bold))
                                .foregroundStyle(isCurrent ? CadenciaPalette.obsidian : CadenciaPalette.saffronGold)
                            Text("\(spec.paceWPM) WPM")
                                .font(.subheadline.weight(.bold))
                                .foregroundStyle(isCurrent ? CadenciaPalette.obsidian : CadenciaPalette.ivoryText)
                            Text("+\(spec.newItemsPerSession)/sess")
                                .font(CadenciaFont.mono(10))
                                .foregroundStyle(isCurrent ? CadenciaPalette.obsidian.opacity(0.8) : CadenciaPalette.mutedSilver)
                            Text("\(Int(spec.audioFirstFraction * 100))% ear")
                                .font(CadenciaFont.mono(10))
                                .foregroundStyle(isCurrent ? CadenciaPalette.obsidian.opacity(0.8) : CadenciaPalette.bioluminescentCobalt)
                        }
                        .padding(14)
                        .frame(width: 96, alignment: .leading)
                        .background(
                            isCurrent ? AnyShapeStyle(CadenciaPalette.saffronGold) : AnyShapeStyle(Color.white.opacity(0.06)),
                            in: RoundedRectangle(cornerRadius: 16, style: .continuous)
                        )
                    }
                }
            }
        }
        .padding(20)
        .cadenciaGlassCard(cornerRadius: 24)
    }

    private func metricTile(label: String, value: String, unit: String, target: String, accent: Color) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(CadenciaFont.mono(10, weight: .bold))
                .foregroundStyle(CadenciaPalette.mutedSilver)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value)
                    .font(CadenciaFont.displaySerif(28, weight: .bold))
                    .foregroundStyle(accent)
                if !unit.isEmpty {
                    Text(unit)
                        .font(CadenciaFont.mono(11))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
            }
            Text(target)
                .font(.caption2)
                .foregroundStyle(CadenciaPalette.mutedSilver)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .cadenciaGlassCard(cornerRadius: 20)
    }

    private func pillBadge(icon: String, text: String, color: Color) -> some View {
        Label(text, systemImage: icon)
            .font(CadenciaFont.mono(11, weight: .semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(color.opacity(0.14), in: Capsule())
    }

    private func legendDot(color: Color, label: String) -> some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 7, height: 7)
            Text(label)
        }
    }
}
