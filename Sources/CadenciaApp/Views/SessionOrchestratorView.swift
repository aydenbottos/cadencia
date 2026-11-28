import CadenciaCore
import SwiftUI

/// Full-screen 75-minute immersive session container orchestrating all 5 blocks + Closing Ritual,
/// wrapped in a floating Liquid Glass precision session bar and adaptive 120Hz Metal Aurora backdrop.
public struct SessionOrchestratorView: View {
    @Bindable var controller: SessionController
    public let onDismiss: () -> Void
    @State private var showExitConfirm = false

    public init(controller: SessionController, onDismiss: @escaping () -> Void) {
        self.controller = controller
        self.onDismiss = onDismiss
    }

    private var auroraMood: AuroraMood {
        switch controller.currentBlock {
        case .warmup: return .warmup
        case .input1: return .input1
        case .output1: return .outputSpeaking
        case .input2: return .input2AudioFirst
        case .output2: return .outputSpeaking
        case .closing: return .studio
        }
    }

    public var body: some View {
        ZStack {
            MediterraneanAuroraView(mood: auroraMood, audioEnergy: controller.appState.liveAudioEnergy)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                if controller.currentBlock != .closing {
                    topLiquidGlassTimerBar
                        .padding(.horizontal, 16)
                        .padding(.top, 8)
                        .padding(.bottom, 6)
                        .zIndex(10)
                }

                Group {
                    switch controller.currentBlock {
                    case .warmup:
                        WarmupBlockView(controller: controller)
                    case .input1, .input2:
                        InputReaderView(controller: controller, onCancelAndExit: {
                            controller.cancelSession()
                            onDismiss()
                        })
                    case .output1:
                        SpeakingPartnerView(controller: controller)
                    case .output2:
                        if controller.isWritingDay {
                            WritingStudioView(controller: controller)
                        } else {
                            SpeakingPartnerView(controller: controller)
                        }
                    case .closing:
                        SessionClosingView(controller: controller, onDismiss: onDismiss)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .preferredColorScheme(.dark)
        .alert("End Session Early?", isPresented: $showExitConfirm) {
            Button("Finish & Log Progress") {
                Task { await controller.transitionToBlock(.closing) }
            }
            Button("Exit Without Saving", role: .destructive) {
                controller.cancelSession()
                onDismiss()
            }
            Button("Keep Going", role: .cancel) {}
        } message: {
            Text("Log your active minutes to SQLite, or exit straight back to Studio.")
        }
        .alert("Notice", isPresented: Binding(
            get: { controller.errorMessage != nil },
            set: { if !$0 { controller.errorMessage = nil } }
        )) {
            Button("OK") { controller.errorMessage = nil }
        } message: {
            Text(controller.errorMessage ?? "")
        }
    }

    private var topLiquidGlassTimerBar: some View {
        CadenciaGlassContainer(spacing: 10) {
            HStack(spacing: 10) {
                Button {
                    let hasLoggedProgress = controller.passagesCompleted > 0
                        || !controller.speakingTurns.isEmpty
                        || controller.usableEventsThisSession > 0
                    if controller.isGeneratingPassage || !hasLoggedProgress {
                        controller.cancelSession()
                        onDismiss()
                    } else {
                        showExitConfirm = true
                    }
                } label: {
                    Image(systemName: "xmark")
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                        .frame(width: 44, height: 44)
                        .contentShape(Circle())
                        .cadenciaGlassPill(interactive: false)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Close Session")

                // Block progress pill
                HStack(spacing: 8) {
                    Circle()
                        .fill(CadenciaPalette.color(for: controller.currentBlock))
                        .frame(width: 8, height: 8)
                    Text(controller.currentBlock.title)
                        .font(CadenciaFont.uiCaption(12, weight: .bold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                        .lineLimit(1)
                    Spacer(minLength: 4)
                    Text(formattedClock(controller.blockElapsedSeconds) + " / " + formattedClock(controller.currentBlock.durationSeconds))
                        .font(CadenciaFont.mono(11, weight: .semibold))
                        .foregroundStyle(CadenciaPalette.saffronGold)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .cadenciaGlassPill(tint: CadenciaPalette.color(for: controller.currentBlock), interactive: false)

                // Pause / Next Block controls
                Menu {
                    Button(controller.isTimerPaused ? "Resume Timer" : "Pause Timer") {
                        controller.isTimerPaused.toggle()
                    }
                    Divider()
                    ForEach(SessionBlockType.allCases, id: \.self) { blk in
                        Button("Jump to \(blk.title)") {
                            Task { await controller.transitionToBlock(blk) }
                        }
                    }
                } label: {
                    Image(systemName: controller.isTimerPaused ? "play.fill" : "ellipsis")
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                        .frame(width: 44, height: 44)
                        .contentShape(Circle())
                        .cadenciaGlassPill(interactive: false)
                }
            }
        }
    }

    private func formattedClock(_ seconds: TimeInterval) -> String {
        let s = max(0, Int(seconds))
        return String(format: "%02d:%02d", s / 60, s % 60)
    }
}
