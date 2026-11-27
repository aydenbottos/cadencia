import CadenciaCore
import SwiftUI

/// Session Closing Ritual (PDF pp. 9, 12):
///  - 1–7 Fatigue & Enjoyment sliders
///  - Mandatory Daily "No Other Spanish" diary confirmation
///  - Live Session Telemetry Recap (Usable Events/min, Probe Accuracy, Promotions, Running Words)
///  - Finalizes SQLite session summary, 7-day delayed capacity test scheduling, and append-only events.
public struct SessionClosingView: View {
    @Bindable var controller: SessionController
    let onDismiss: () -> Void

    @State private var isSaving: Bool = false

    public init(controller: SessionController, onDismiss: @escaping () -> Void) {
        self.controller = controller
        self.onDismiss = onDismiss
    }

    public var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                // Header
                VStack(spacing: 8) {
                    Image(systemName: controller.completedSummary == nil ? "sun.horizon.fill" : "checkmark.seal.fill")
                        .font(.system(size: 42))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [CadenciaPalette.saffronGold, CadenciaPalette.terracotta],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .symbolEffect(.bounce, value: controller.completedSummary != nil)

                    Text("Session \(controller.profile.currentSessionNumber) Complete")
                        .font(CadenciaFont.displaySerif(30, weight: .bold))
                        .foregroundStyle(CadenciaPalette.ivoryText)

                    Text(controller.completedSummary == nil
                         ? "Record your end-of-session diary to seal today's immutable evidence log."
                         : "Session sealed into local SQLite. Every event is preserved.")
                        .font(CadenciaFont.uiCaption(14))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 12)

                // Telemetry Summary Grid
                let inputMins = max(controller.totalInputSeconds / 60.0, 0.01)
                let density = Double(controller.usableEventsThisSession) / inputMins
                let probePct = controller.probesTotal > 0
                    ? Int((Double(controller.probesCorrect) / Double(controller.probesTotal) * 100).rounded())
                    : 100

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 14) {
                    metricTile(
                        title: "USABLE EVENTS / MIN",
                        value: String(format: "%.1f", density),
                        subtitle: "Target ≥ 15.0 (Claim 1)",
                        accent: density >= 15.0 ? CadenciaPalette.mediterraneanJade : CadenciaPalette.saffronGold
                    )
                    metricTile(
                        title: "LOAD-BEARING PROBES",
                        value: "\(probePct)%",
                        subtitle: "\(controller.probesCorrect)/\(controller.probesTotal) correct (≥85%)",
                        accent: probePct >= 85 ? CadenciaPalette.bioluminescentCobalt : CadenciaPalette.terracotta
                    )
                    metricTile(
                        title: "WORDS & PASSAGES",
                        value: "\(controller.runningWordsProcessed)",
                        subtitle: "\(controller.passagesCompleted) passages · +\(controller.newItemsIntroducedCount) new",
                        accent: CadenciaPalette.terracotta
                    )
                    metricTile(
                        title: "STATE PROMOTIONS",
                        value: "+\(controller.promotionsToKnown + controller.promotionsToProductive)",
                        subtitle: "\(controller.promotionsToKnown) Known · \(controller.promotionsToProductive) Prod.",
                        accent: CadenciaPalette.saffronGold
                    )
                }

                if controller.completedSummary == nil {
                    // 1-7 Fatigue & Enjoyment Sliders
                    VStack(alignment: .leading, spacing: 18) {
                        Text("END-OF-SESSION RATINGS (1–7)")
                            .font(CadenciaFont.mono(11, weight: .bold))
                            .foregroundStyle(CadenciaPalette.saffronGold)
                            .tracking(1.2)

                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Label("Mental Fatigue", systemImage: "brain.head.profile")
                                    .font(CadenciaFont.uiTitle(15))
                                    .foregroundStyle(CadenciaPalette.ivoryText)
                                Spacer()
                                Text("\(controller.fatigueRating) / 7")
                                    .font(CadenciaFont.mono(16, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.terracotta)
                            }
                            Slider(
                                value: Binding(
                                    get: { Double(controller.fatigueRating) },
                                    set: { controller.fatigueRating = Int($0.rounded()) }
                                ),
                                in: 1...7,
                                step: 1
                            )
                            .tint(CadenciaPalette.terracotta)
                            HStack {
                                Text("1 · Fresh").font(CadenciaFont.uiCaption(11)).foregroundStyle(CadenciaPalette.mutedSilver)
                                Spacer()
                                Text("7 · Exhausted").font(CadenciaFont.uiCaption(11)).foregroundStyle(CadenciaPalette.mutedSilver)
                            }
                        }

                        Divider().overlay(Color.white.opacity(0.12))

                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Label("Immersion Enjoyment", systemImage: "sparkles")
                                    .font(CadenciaFont.uiTitle(15))
                                    .foregroundStyle(CadenciaPalette.ivoryText)
                                Spacer()
                                Text("\(controller.enjoymentRating) / 7")
                                    .font(CadenciaFont.mono(16, weight: .bold))
                                    .foregroundStyle(CadenciaPalette.mediterraneanJade)
                            }
                            Slider(
                                value: Binding(
                                    get: { Double(controller.enjoymentRating) },
                                    set: { controller.enjoymentRating = Int($0.rounded()) }
                                ),
                                in: 1...7,
                                step: 1
                            )
                            .tint(CadenciaPalette.mediterraneanJade)
                            HStack {
                                Text("1 · Tedious").font(CadenciaFont.uiCaption(11)).foregroundStyle(CadenciaPalette.mutedSilver)
                                Spacer()
                                Text("7 · Absorbed").font(CadenciaFont.uiCaption(11)).foregroundStyle(CadenciaPalette.mutedSilver)
                            }
                        }

                        Divider().overlay(Color.white.opacity(0.12))

                        // Mandatory Protocol Contamination Diary Check (PDF p. 12)
                        Toggle(isOn: $controller.noOtherSpanishConfirmed) {
                            VStack(alignment: .leading, spacing: 3) {
                                Text("Daily \"No Other Spanish\" Diary")
                                    .font(CadenciaFont.uiTitle(15))
                                    .foregroundStyle(CadenciaPalette.ivoryText)
                                Text("Confirm you had zero external Spanish study, flashcards, podcasts, or subtitles outside this 60-hour timer today.")
                                    .font(CadenciaFont.uiCaption(12))
                                    .foregroundStyle(CadenciaPalette.mutedSilver)
                            }
                        }
                        .tint(CadenciaPalette.mediterraneanJade)
                    }
                    .padding(20)
                    .cadenciaGlassCard(cornerRadius: 24)

                    if let err = controller.errorMessage {
                        Text(err)
                            .font(CadenciaFont.uiCaption(13))
                            .foregroundStyle(CadenciaPalette.crimsonAlert)
                            .padding()
                            .cadenciaGlassCard(cornerRadius: 16, tint: CadenciaPalette.crimsonAlert)
                    }

                    Button {
                        Task {
                            isSaving = true
                            await controller.finalizeSession()
                            isSaving = false
                        }
                    } label: {
                        HStack(spacing: 10) {
                            if isSaving {
                                ProgressView().tint(CadenciaPalette.obsidian)
                            } else {
                                Image(systemName: "lock.doc.fill")
                            }
                            Text(isSaving ? "Sealing Evidence Log…" : "Seal Session & Save to Local DB")
                                .font(CadenciaFont.uiTitle(16, weight: .bold))
                        }
                        .foregroundStyle(CadenciaPalette.obsidian)
                        .frame(maxWidth: .infinity)
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
                    .disabled(isSaving)
                } else {
                    // Sealed confirmation card
                    VStack(spacing: 12) {
                        Label("Append-Only Evidence Sealed", systemImage: "checkmark.shield.fill")
                            .font(CadenciaFont.uiTitle(16, weight: .bold))
                            .foregroundStyle(CadenciaPalette.mediterraneanJade)
                        Text("Active Input: \(Int(controller.totalInputSeconds / 60))m · Active Output: \(Int(controller.totalOutputSeconds / 60))m · Usable Events: \(controller.usableEventsThisSession)")
                            .font(CadenciaFont.mono(12))
                            .foregroundStyle(CadenciaPalette.mutedSilver)
                    }
                    .padding(20)
                    .frame(maxWidth: .infinity)
                    .cadenciaGlassCard(cornerRadius: 22, tint: CadenciaPalette.mediterraneanJade)

                    Button {
                        onDismiss()
                    } label: {
                        Text("Return to Studio")
                            .font(CadenciaFont.uiTitle(16, weight: .bold))
                            .foregroundStyle(CadenciaPalette.obsidian)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(CadenciaPalette.ivoryText, in: Capsule())
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 40)
        }
    }

    private func metricTile(title: String, value: String, subtitle: String, accent: Color) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(CadenciaFont.mono(10, weight: .bold))
                .foregroundStyle(accent)
                .tracking(0.8)
            Text(value)
                .font(CadenciaFont.displaySerif(28, weight: .bold))
                .foregroundStyle(CadenciaPalette.ivoryText)
            Text(subtitle)
                .font(CadenciaFont.uiCaption(11))
                .foregroundStyle(CadenciaPalette.mutedSilver)
                .lineLimit(2)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .cadenciaGlassCard(cornerRadius: 20, tint: accent)
    }
}
