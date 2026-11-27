import CadenciaCore
import SwiftUI

/// Twice-weekly 13-minute Writing Studio (from Week 3 onward, PDF pp. 8–9) with inline recast feedback
/// (no grammar lectures) and automatic productive lemma & pattern failure logging.
public struct WritingStudioView: View {
    @Bindable var controller: SessionController

    private var wordCount: Int {
        controller.writingDraftText.split { $0.isWhitespace || $0.isNewline }.count
    }

    public var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label("OUTPUT II · 13-MINUTE WRITING STUDIO", systemImage: "pencil.and.outline")
                            .font(CadenciaFont.mono(10, weight: .bold))
                            .tracking(2)
                            .foregroundStyle(CadenciaPalette.saffronGold)
                        Spacer()
                        Text("\(wordCount) words")
                            .font(CadenciaFont.mono(11, weight: .bold))
                            .foregroundStyle(CadenciaPalette.ivoryText)
                    }
                    Text(controller.writingPromptEs)
                        .font(CadenciaFont.proseSerif(19))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                        .lineSpacing(4)
                }
                .padding(18)
                .cadenciaGlassCard(cornerRadius: 22, tint: CadenciaPalette.saffronGold)

                if let fb = controller.writingFeedback {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("RECAST REVIEW · \(fb.correctlyUsedItemIds.count) PRODUCTIVE CREDITS")
                            .font(CadenciaFont.mono(11, weight: .bold))
                            .foregroundStyle(CadenciaPalette.mediterraneanJade)

                        ForEach(Array(fb.recastSentences.enumerated()), id: \.offset) { _, pair in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(pair.original)
                                    .font(CadenciaFont.proseSerif(16))
                                    .foregroundStyle(pair.hadError ? CadenciaPalette.mutedSilver : CadenciaPalette.ivoryText)
                                    .strikethrough(pair.hadError, color: CadenciaPalette.terracotta)
                                if pair.hadError {
                                    Text("→ \(pair.corrected)")
                                        .font(CadenciaFont.proseSerif(17))
                                        .foregroundStyle(CadenciaPalette.mediterraneanJade)
                                }
                            }
                            .padding(14)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }

                        Button {
                            Task { await controller.transitionToBlock(.closing) }
                        } label: {
                            Text("Continue to Session Reflection →")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .foregroundStyle(CadenciaPalette.obsidian)
                                .background(CadenciaPalette.saffronGold, in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(18)
                    .cadenciaGlassCard(cornerRadius: 22)
                } else {
                    TextEditor(text: $controller.writingDraftText)
                        .scrollContentBackground(.hidden)
                        .font(CadenciaFont.proseSerif(18))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                        .frame(minHeight: 260)
                        .padding(16)
                        .cadenciaGlassCard(cornerRadius: 22)

                    Button {
                        Task { await controller.submitWritingDraft() }
                    } label: {
                        HStack {
                            if controller.isEvaluatingWriting {
                                ProgressView().tint(CadenciaPalette.obsidian)
                            }
                            Text(controller.isEvaluatingWriting ? "Evaluating with GPT-6.1 Terra…" : "Submit for Recast Review")
                                .font(.headline)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .foregroundStyle(CadenciaPalette.obsidian)
                        .background(CadenciaPalette.saffronGold, in: Capsule())
                    }
                    .disabled(wordCount < 10 || controller.isEvaluatingWriting)
                }
            }
            .padding(20)
        }
    }
}
