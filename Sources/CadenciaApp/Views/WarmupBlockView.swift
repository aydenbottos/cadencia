import CadenciaCore
import SwiftUI

/// Minutes 0–3: 20 rapid meaning checks on the shakiest items from the previous session (PDF p. 8).
public struct WarmupBlockView: View {
    @Bindable var controller: SessionController

    public var body: some View {
        VStack(spacing: 24) {
            if controller.warmupIndex < controller.warmupItems.count {
                let item = controller.warmupItems[controller.warmupIndex]

                VStack(spacing: 6) {
                    Text("WARM-UP · ITEM \(controller.warmupIndex + 1) OF \(controller.warmupItems.count)")
                        .font(CadenciaFont.mono(11, weight: .bold))
                        .tracking(2.0)
                        .foregroundStyle(CadenciaPalette.saffronGold)
                    ProgressView(value: Double(controller.warmupIndex), total: Double(max(1, controller.warmupItems.count)))
                        .tint(CadenciaPalette.saffronGold)
                        .padding(.horizontal, 40)
                }
                .padding(.top, 20)

                Spacer()

                // Tactile editorial flashcard
                VStack(spacing: 18) {
                    Image(systemName: item.visualSymbol)
                        .font(.system(size: 32))
                        .foregroundStyle(CadenciaPalette.saffronGold)

                    Text(item.spanish)
                        .font(CadenciaFont.displaySerif(42, weight: .bold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                        .multilineTextAlignment(.center)

                    Text("/\(item.ipaEs)/ · \(item.pos)")
                        .font(CadenciaFont.mono(13))
                        .foregroundStyle(CadenciaPalette.mutedSilver)

                    if controller.warmupShowingAnswer {
                        Divider().overlay(Color.white.opacity(0.14))
                        Text(item.englishGloss)
                            .font(CadenciaFont.displaySerif(24, weight: .medium))
                            .foregroundStyle(CadenciaPalette.mediterraneanJade)
                            .multilineTextAlignment(.center)
                            .transition(.opacity.combined(with: .scale(scale: 0.96)))
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 44)
                .padding(.horizontal, 26)
                .cadenciaGlassCard(cornerRadius: 30, tint: CadenciaPalette.saffronGold, interactive: true)
                .padding(.horizontal, 24)
                .onTapGesture {
                    HapticDirector.shared.playGlossTap()
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                        controller.warmupShowingAnswer = true
                    }
                }

                Spacer()

                if controller.warmupShowingAnswer {
                    CadenciaGlassContainer(spacing: 14) {
                        HStack(spacing: 14) {
                            Button {
                                Task { await controller.submitWarmupSelfCheck(knewIt: false) }
                            } label: {
                                Label("Shaky (×0.5)", systemImage: "arrow.counterclockwise")
                                    .font(.headline)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 16)
                                    .foregroundStyle(CadenciaPalette.ivoryText)
                                    .cadenciaGlassPill(tint: CadenciaPalette.crimsonAlert)
                            }
                            .buttonStyle(.plain)

                            Button {
                                Task { await controller.submitWarmupSelfCheck(knewIt: true) }
                            } label: {
                                Label("Knew It (×1.6)", systemImage: "checkmark")
                                    .font(.headline)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 16)
                                    .foregroundStyle(CadenciaPalette.obsidian)
                                    .background(CadenciaPalette.mediterraneanJade, in: Capsule())
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(.horizontal, 24)
                        .padding(.bottom, 28)
                    }
                } else {
                    Button {
                        HapticDirector.shared.playGlossTap()
                        withAnimation(.spring(response: 0.35)) {
                            controller.warmupShowingAnswer = true
                        }
                    } label: {
                        Text("Tap Card to Reveal Meaning")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .foregroundStyle(CadenciaPalette.ivoryText)
                            .cadenciaGlassPill()
                    }
                    .buttonStyle(.plain)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 28)
                }
            } else {
                ProgressView("Transitioning to Input I…")
                    .tint(CadenciaPalette.saffronGold)
            }
        }
    }
}
