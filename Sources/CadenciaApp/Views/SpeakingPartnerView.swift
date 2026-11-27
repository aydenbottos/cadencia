import CadenciaCore
import CadenciaSpeech
import SwiftUI

/// Immersive Bidirectional Bilingual Voice Studio for Output 1 (Retell, min 28–40) and Output 2 (Open Talk / SIELE, min 62–75).
/// Features:
///  - Live 120Hz Raymarched Metal Resonance Orb responding to learner mic FFT & dual tutor audio
///  - Native Spanish voicing via **Gradium Phonon-40m-es** + English conversation coaching via **Kokoro-82M** (24 kHz stitched)
///  - Concurrent Parakeet Ultra 600M INT8 transcript + ZIPA CR Large 300M FP16 **IPA Phoneme Articulation Ribbon**
///  - Natural implicit Recast Corrections + toggleable **ES + EN Tutor** vs **Spanish Only** mode
public struct SpeakingPartnerView: View {
    @Bindable var controller: SessionController

    public var body: some View {
        VStack(spacing: 0) {
            // Top Tutor Mode Bar (ES + EN Tutor via Phonon + Kokoro vs Spanish Only)
            HStack(spacing: 10) {
                Picker("Tutor Language", selection: $controller.conversationTutorMode) {
                    ForEach(ConversationTutorMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                Text(controller.conversationTutorMode.allowsEnglish ? "PHONON + KOKORO" : "PHONON ES")
                    .font(CadenciaFont.mono(9, weight: .bold))
                    .foregroundStyle(CadenciaPalette.saffronGold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(CadenciaPalette.saffronGold.opacity(0.14), in: Capsule())
            }
            .padding(.horizontal, 18)
            .padding(.top, 8)
            .padding(.bottom, 6)

            ScrollViewReader { proxy in
                ScrollView(showsIndicators: false) {
                    VStack(spacing: 18) {
                        // Current Tutor Prompt Banner (Spanish + optional English Coaching)
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Label(
                                    controller.currentBlock == .output1 ? "OUTPUT I · RETELL" : "OUTPUT II · LIVE CONVERSATION",
                                    systemImage: "waveform.circle.fill"
                                )
                                .font(CadenciaFont.mono(10, weight: .bold))
                                .tracking(2)
                                .foregroundStyle(CadenciaPalette.mediterraneanJade)
                                Spacer()
                                Button {
                                    Task {
                                        let accent = controller.currentPassage?.accent ?? .castilian
                                        let voice: TutorVoice = controller.currentBlock == .output2 ? .leonel : .celia
                                        await controller.speakSpanishLine(controller.currentPartnerPromptEs, voice: voice, accent: accent)
                                    }
                                } label: {
                                    Label("ES", systemImage: "speaker.wave.2.fill")
                                        .font(CadenciaFont.mono(10, weight: .bold))
                                        .foregroundStyle(CadenciaPalette.mediterraneanJade)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(CadenciaPalette.mediterraneanJade.opacity(0.16), in: Capsule())
                                }
                                .buttonStyle(.plain)
                            }

                            Text(controller.currentPartnerPromptEs)
                                .font(CadenciaFont.proseSerif(21))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                                .lineSpacing(4)

                            if controller.conversationTutorMode.allowsEnglish,
                               let englishCoach = controller.currentPartnerReplyEn ?? controller.currentPartnerHintEn,
                               !englishCoach.isEmpty {
                                HStack(alignment: .top, spacing: 10) {
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text("ENGLISH TUTOR · KOKORO-82M")
                                            .font(CadenciaFont.mono(9, weight: .bold))
                                            .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                                        Text(englishCoach)
                                            .font(CadenciaFont.uiCaption(13))
                                            .foregroundStyle(CadenciaPalette.ivoryText.opacity(0.88))
                                    }
                                    Spacer()
                                    Button {
                                        Task { await controller.speakEnglishCoachLine(englishCoach) }
                                    } label: {
                                        Image(systemName: "speaker.wave.2.fill")
                                            .font(.caption)
                                            .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                                            .padding(8)
                                            .background(CadenciaPalette.bioluminescentCobalt.opacity(0.16), in: Circle())
                                    }
                                    .buttonStyle(.plain)
                                }
                                .padding(10)
                                .background(CadenciaPalette.bioluminescentCobalt.opacity(0.10), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                            } else if let hint = controller.currentPartnerHintEn {
                                Text(hint)
                                    .font(.caption)
                                    .foregroundStyle(CadenciaPalette.mutedSilver)
                            }
                        }
                        .padding(18)
                        .cadenciaGlassCard(cornerRadius: 22, tint: CadenciaPalette.mediterraneanJade)

                        // Conversation & Phoneme Alignment Turn Feed
                        ForEach(controller.speakingTurns) { turn in
                            turnEvaluationCard(turn)
                                .id(turn.id)
                        }
                    }
                    .padding(.horizontal, 18)
                    .padding(.vertical, 12)
                }
                .onChange(of: controller.speakingTurns.count) { _, _ in
                    if let last = controller.speakingTurns.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }

            // Bottom Raymarched Resonance Orb & Controls
            VStack(spacing: 10) {
                ZStack {
                    ResonanceOrbView(
                        mode: controller.orbMode,
                        energy: controller.appState.liveAudioEnergy
                    )
                    .frame(width: 175, height: 175)

                    if controller.isEvaluatingVoice {
                        ProgressView()
                            .tint(CadenciaPalette.ivoryText)
                            .scaleEffect(1.3)
                    } else {
                        Image(systemName: controller.isRecordingVoice ? "stop.fill" : "mic.fill")
                            .font(.system(size: 30, weight: .semibold))
                            .foregroundStyle(CadenciaPalette.ivoryText)
                            .shadow(color: .black.opacity(0.45), radius: 8)
                    }
                }
                .contentShape(Circle())
                .onTapGesture {
                    guard !controller.isEvaluatingVoice else { return }
                    Task { await controller.toggleVoiceRecording() }
                }

                HStack {
                    Text(statusCaption)
                        .font(CadenciaFont.mono(11, weight: .semibold))
                        .foregroundStyle(CadenciaPalette.saffronGold)
                    Spacer()
                    Button(controller.currentBlock == .output1 ? "Next: Input II →" : "Finish Session →") {
                        let next: SessionBlockType = controller.currentBlock == .output1 ? .input2 : .closing
                        Task { await controller.transitionToBlock(next) }
                    }
                    .font(CadenciaFont.mono(11, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .cadenciaGlassPill()
                }
                .padding(.horizontal, 22)
                .padding(.bottom, 16)
            }
        }
    }

    private var statusCaption: String {
        if controller.isRecordingVoice { return "LISTENING · TAP ORB TO SEND" }
        if controller.isEvaluatingVoice { return "PARAKEET INT8 + ZIPA FP16 + TERRA…" }
        if controller.orbMode == .speaking { return "\(controller.activeSpeakingEngineLabel) SPEAKING…" }
        return controller.conversationTutorMode.allowsEnglish
            ? "TAP ORB · SPEAK SPANISH (OR ASK IN ENGLISH)"
            : "TAP ORB TO SPEAK SPANISH"
    }

    private func turnEvaluationCard(_ turn: SpokenTurnEvaluation) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Learner spoken utterance (Parakeet Ultra 600M INT8)
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("YOU SAID")
                        .font(CadenciaFont.mono(10, weight: .bold))
                        .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                    Spacer()
                    Text(String(format: "Parakeet %.0fms · ZIPA %.0fms", turn.parakeetLatencyMs, turn.zipaLatencyMs))
                        .font(CadenciaFont.mono(9))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
                Text(turn.learnerTranscript)
                    .font(CadenciaFont.proseSerif(18))
                    .foregroundStyle(CadenciaPalette.ivoryText)
            }

            // Live ZIPA CR Large 300M FP16 Phoneme Articulation Ribbon
            if !turn.alignments.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("IPA PHONEME RIBBON (ZIPA 300M FP16)")
                            .font(CadenciaFont.mono(9, weight: .bold))
                            .foregroundStyle(CadenciaPalette.saffronGold)
                        Spacer()
                        Text(String(format: "%.0f%% phonetic match", turn.phonemeAccuracy * 100))
                            .font(CadenciaFont.mono(10, weight: .bold))
                            .foregroundStyle(turn.phonemeAccuracy >= 0.80 ? CadenciaPalette.mediterraneanJade : CadenciaPalette.saffronGold)
                    }
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 4) {
                            ForEach(turn.alignments.prefix(40)) { cell in
                                let col: Color = cell.distance <= 0.16
                                    ? CadenciaPalette.mediterraneanJade
                                    : (cell.distance <= 0.38 ? CadenciaPalette.saffronGold : CadenciaPalette.crimsonAlert)
                                VStack(spacing: 2) {
                                    Text(cell.observedPhone)
                                        .font(CadenciaFont.mono(12, weight: .bold))
                                        .foregroundStyle(col)
                                    Text(cell.expectedPhone)
                                        .font(CadenciaFont.mono(9))
                                        .foregroundStyle(CadenciaPalette.mutedSilver)
                                }
                                .padding(.horizontal, 6)
                                .padding(.vertical, 4)
                                .background(col.opacity(0.14), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                            }
                        }
                    }

                    if let hint = turn.alignments.first(where: { $0.coachingHint != nil })?.coachingHint {
                        Label(hint, systemImage: "mouth.fill")
                            .font(.caption2)
                            .foregroundStyle(CadenciaPalette.saffronGold)
                    }
                }
            }

            // Implicit Recast Correction
            if let recast = turn.partnerRecastEs, !recast.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "sparkles")
                        .foregroundStyle(CadenciaPalette.saffronGold)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("NATURAL RECAST")
                            .font(CadenciaFont.mono(9, weight: .bold))
                            .foregroundStyle(CadenciaPalette.saffronGold)
                        Text(recast)
                            .font(CadenciaFont.proseSerif(16))
                            .foregroundStyle(CadenciaPalette.ivoryText)
                    }
                    Spacer()
                    Button {
                        Task {
                            let accent = controller.currentPassage?.accent ?? .castilian
                            await controller.speakSpanishLine(recast, voice: .celia, accent: accent)
                        }
                    } label: {
                        Image(systemName: "speaker.wave.2.fill")
                            .font(.caption)
                            .foregroundStyle(CadenciaPalette.saffronGold)
                            .padding(6)
                            .background(CadenciaPalette.saffronGold.opacity(0.15), in: Circle())
                    }
                    .buttonStyle(.plain)
                }
                .padding(10)
                .background(CadenciaPalette.saffronGold.opacity(0.12), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }

            // Bilingual Tutor Reply: Spanish (Phonon-40m-es) + English Coaching (Kokoro-82M)
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("TUTOR · PHONON-40M-ES")
                        .font(CadenciaFont.mono(10, weight: .bold))
                        .foregroundStyle(CadenciaPalette.mediterraneanJade)
                    Spacer()
                    Button {
                        Task { await controller.replayConversationTurn(turn) }
                    } label: {
                        Label("Replay", systemImage: "play.circle.fill")
                            .font(CadenciaFont.mono(10, weight: .bold))
                            .foregroundStyle(CadenciaPalette.ivoryText)
                            .padding(.horizontal, 9)
                            .padding(.vertical, 4)
                            .background(Color.white.opacity(0.10), in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
                Text(turn.partnerReplyEs)
                    .font(CadenciaFont.proseSerif(17))
                    .foregroundStyle(CadenciaPalette.ivoryText.opacity(0.95))

                if let englishReply = turn.partnerReplyEn, !englishReply.isEmpty {
                    HStack(alignment: .top, spacing: 8) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("ENGLISH · KOKORO-82M")
                                .font(CadenciaFont.mono(9, weight: .bold))
                                .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                            Text(englishReply)
                                .font(CadenciaFont.uiCaption(14))
                                .foregroundStyle(CadenciaPalette.ivoryText.opacity(0.90))
                        }
                        Spacer()
                        Button {
                            Task { await controller.speakEnglishCoachLine(englishReply) }
                        } label: {
                            Image(systemName: "speaker.wave.2.fill")
                                .font(.caption)
                                .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                                .padding(6)
                                .background(CadenciaPalette.bioluminescentCobalt.opacity(0.16), in: Circle())
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(10)
                    .background(CadenciaPalette.bioluminescentCobalt.opacity(0.10), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
            }

            if !turn.correctlyUsedItemIds.isEmpty {
                Text("✓ +\(turn.correctlyUsedItemIds.count) target item(s) credited toward Productive mastery")
                    .font(CadenciaFont.mono(10, weight: .semibold))
                    .foregroundStyle(CadenciaPalette.mediterraneanJade)
            }
        }
        .padding(16)
        .cadenciaGlassCard(cornerRadius: 20)
    }
}
