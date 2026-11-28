import CadenciaCore
import SwiftUI

/// Immersive Editorial Input Reader & Audio-First Listening Theater covering BOTH:
/// - **Input 1 (Minutes 3–28)**: Synced 120Hz Karaoke Spanish prose + Phonon-40m-es native audio, tap-to-gloss
///   Liquid Glass inspector (with visual cue, IPA, audio), sentence tap-to-replay (logs Doubt),
///   20-second Grammar Micro-Note interstitial, and Load-Bearing Comprehension Probes.
/// - **Input 2 (Minutes 40–62)**: Full-bleed Midnight Indigo Audio-First Theater (text completely hidden
///   while listening + answering probes by ear -> dramatic Curtain Reveal AFTER probes, earning `ear_success`!).
public struct InputReaderView: View {
    @Bindable var controller: SessionController
    public var onCancelAndExit: (() -> Void)? = nil
    @State private var revealedBootstrapLineIds: Set<Int> = []

    public var body: some View {
        ZStack(alignment: .bottom) {
            if controller.isGeneratingPassage {
                generatingStateView
            } else if let note = controller.activeGrammarNote {
                grammarMicroNoteInterstitial(note)
            } else if let passage = controller.currentPassage {
                VStack(spacing: 0) {
                    if passage.isAudioFirst && !controller.audioFirstTextRevealed {
                        audioFirstTheater(passage)
                    } else {
                        syncedTextReader(passage)
                    }
                }
            } else {
                VStack(spacing: 16) {
                    Text("Ready for Next Passage")
                        .font(CadenciaFont.displaySerif(24))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                    Button("Generate Passage") {
                        Task { await controller.transitionToBlock(controller.currentBlock) }
                    }
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
                    .foregroundStyle(CadenciaPalette.obsidian)
                    .background(CadenciaPalette.saffronGold, in: Capsule())
                }
            }

            // Floating Tap-to-Gloss Liquid Glass Popover Card
            if let gloss = controller.selectedGlossItem {
                glossPopoverCard(item: gloss.item, clue: gloss.contextClue, surface: gloss.surface)
                    .padding(.horizontal, 18)
                    .padding(.bottom, 92)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
    }

    // MARK: - Loading / Contract Verification State

    private var generatingStateView: some View {
        VStack(spacing: 22) {
            ResonanceOrbView(mode: .thinking, energy: 0.35)
                .frame(width: 190, height: 190)
            Text(controller.generationStatusText)
                .font(CadenciaFont.displaySerif(22, weight: .medium))
                .foregroundStyle(CadenciaPalette.ivoryText)
                .multilineTextAlignment(.center)
            Text("Enforcing Closed Vocabulary · 3× Spaced New Items · Fresh Context · Load-Bearing Probes")
                .font(CadenciaFont.mono(11))
                .foregroundStyle(CadenciaPalette.mutedSilver)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 36)

            if let onCancelAndExit {
                Button {
                    onCancelAndExit()
                } label: {
                    Label("Cancel & Return to Studio", systemImage: "xmark.circle")
                        .font(CadenciaFont.uiCaption(13, weight: .semibold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 11)
                        .cadenciaGlassPill(interactive: false)
                }
                .buttonStyle(.plain)
                .padding(.top, 8)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Rule 6: 20-Second English Grammar Micro-Note with Contrasting Pair

    private func grammarMicroNoteInterstitial(_ note: GrammarMicroNote) -> some View {
        ScrollView(showsIndicators: false) {
            VStack(alignment: .leading, spacing: 20) {
                Label("20-SECOND GRAMMAR NOTE", systemImage: "text.book.closed.fill")
                    .font(CadenciaFont.mono(11, weight: .bold))
                    .tracking(2)
                    .foregroundStyle(CadenciaPalette.saffronGold)

                Text(note.title)
                    .font(CadenciaFont.displaySerif(28, weight: .bold))
                    .foregroundStyle(CadenciaPalette.ivoryText)

                Text(note.noteEn)
                    .font(.body)
                    .foregroundStyle(CadenciaPalette.ivoryText.opacity(0.92))
                    .lineSpacing(5)

                VStack(spacing: 12) {
                    contrastRow(badge: "A", es: note.contrastAEs, en: note.contrastAEn, accent: CadenciaPalette.bioluminescentCobalt)
                    if !note.contrastBEs.isEmpty {
                        contrastRow(badge: "B", es: note.contrastBEs, en: note.contrastBEn, accent: CadenciaPalette.terracotta)
                    }
                }

                Button {
                    HapticDirector.shared.playStrongSuccess()
                    Task { await controller.dismissGrammarNoteAndPlay() }
                } label: {
                    HStack {
                        Text("Got It · Begin Passage")
                            .font(.headline)
                        Spacer()
                        Image(systemName: "arrow.right")
                    }
                    .padding(.horizontal, 22)
                    .padding(.vertical, 16)
                    .foregroundStyle(CadenciaPalette.obsidian)
                    .background(CadenciaPalette.saffronGold, in: Capsule())
                }
                .buttonStyle(.plain)
                .padding(.top, 8)
            }
            .padding(24)
            .cadenciaGlassCard(cornerRadius: 28, tint: CadenciaPalette.saffronGold)
            .padding(.horizontal, 20)
            .padding(.vertical, 28)
        }
    }

    private func contrastRow(badge: String, es: String, en: String, accent: Color) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(badge)
                .font(CadenciaFont.mono(12, weight: .bold))
                .foregroundStyle(CadenciaPalette.obsidian)
                .frame(width: 24, height: 24)
                .background(accent, in: Circle())
            VStack(alignment: .leading, spacing: 4) {
                Text(es)
                    .font(CadenciaFont.proseSerif(18))
                    .foregroundStyle(CadenciaPalette.ivoryText)
                Text(en)
                    .font(.caption)
                    .foregroundStyle(CadenciaPalette.mutedSilver)
            }
            Spacer()
        }
        .padding(14)
        .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    // MARK: - Input 2: Audio-First Midnight Indigo Theater (Text Hidden Until After Probes!)

    private func audioFirstTheater(_ passage: GeneratedPassage) -> some View {
        ScrollView(showsIndicators: false) {
            VStack(spacing: 24) {
                VStack(spacing: 6) {
                    Label("AUDIO-FIRST · EAR ONLY", systemImage: "ear.fill")
                        .font(CadenciaFont.mono(11, weight: .bold))
                        .tracking(2.0)
                        .foregroundStyle(CadenciaPalette.bioluminescentCobalt)
                    Text(passage.titleEs)
                        .font(CadenciaFont.displaySerif(26, weight: .bold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                        .multilineTextAlignment(.center)
                    Text("\(passage.voice.rawValue) · \(passage.accent.displayName) · \(controller.currentWPMOverride) WPM")
                        .font(CadenciaFont.mono(11))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
                .padding(.top, 12)

                // Living Raymarched Acoustic Orb reacting to Phonon-40m-es native Spanish audio
                ResonanceOrbView(
                    mode: controller.isPlayingAudio ? .speaking : .idle,
                    energy: controller.appState.liveAudioEnergy
                )
                .frame(width: 240, height: 240)
                .onTapGesture {
                    HapticDirector.shared.playGlossTap()
                    Task { await controller.togglePlayPause() }
                }

                // Playback controls
                audioTransportBar(passage)

                if !controller.showingProbes {
                    Button {
                        HapticDirector.shared.playGlossTap()
                        controller.proceedToProbes()
                    } label: {
                        Label("Answer Probes by Ear to Reveal Text", systemImage: "checkmark.bubble.fill")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .foregroundStyle(CadenciaPalette.obsidian)
                            .background(CadenciaPalette.bioluminescentCobalt, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .padding(.horizontal, 22)
                } else {
                    probesSection(passage)
                        .padding(.horizontal, 18)
                }
            }
            .padding(.bottom, 40)
        }
    }

    // MARK: - Input 1 & Revealed Input 2: Editorial Synced Reader

    private func syncedTextReader(_ passage: GeneratedPassage) -> some View {
        let newLemmasSet = Set(passage.newItemGlosses.map { $0.spanish.lowercased() })
        let sentenceWordOffsets: [Int] = {
            var offsets: [Int] = []
            offsets.reserveCapacity(passage.sentences.count)
            var acc = 0
            for s in passage.sentences {
                offsets.append(acc)
                acc += s.wordTokens.count
            }
            return offsets
        }()

        return VStack(spacing: 0) {
            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 20) {
                    // Passage masthead
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(passage.isReserveFallback
                                 ? (passage.isBootstrap ? "BOOTSTRAP · RESERVE" : "RESERVE PASSAGE")
                                 : (passage.format == .serialEpisode ? "SERIAL EPISODE" : (passage.isBootstrap ? "BOOTSTRAP" : "NON-FICTION")))
                                .font(CadenciaFont.mono(10, weight: .bold))
                                .tracking(2)
                                .foregroundStyle(CadenciaPalette.saffronGold)
                            Spacer()
                            Text("\(passage.voice.rawValue) · \(passage.accent.displayName)")
                                .font(CadenciaFont.mono(10))
                                .foregroundStyle(CadenciaPalette.mutedSilver)
                        }
                        Text(passage.titleEs)
                            .font(CadenciaFont.displaySerif(28, weight: .bold))
                            .foregroundStyle(CadenciaPalette.ivoryText)
                        Text(String(format: "%d words · %.0f%% scheduled signal density · %d new",
                                    passage.wordCount, passage.scheduledWordShare * 100, passage.newItemIds.count))
                            .font(CadenciaFont.mono(11))
                            .foregroundStyle(CadenciaPalette.mutedSilver)
                    }
                    .padding(.top, 8)

                    // Interactive tap-to-gloss & tap-to-replay sentences
                    VStack(alignment: .leading, spacing: 18) {
                        ForEach(Array(passage.sentences.enumerated()), id: \.element.id) { idx, sentence in
                            sentenceCard(
                                sentence: sentence,
                                startingWordIndex: sentenceWordOffsets[idx],
                                newLemmas: newLemmasSet,
                                isBootstrap: passage.isBootstrap
                            )
                        }
                    }

                    // Probes section (Rule 5: fewer eligible items means fewer probes, and none means none!)
                    if !passage.probes.isEmpty {
                        if controller.showingProbes || passage.isAudioFirst {
                            probesSection(passage)
                        } else {
                            Button {
                                HapticDirector.shared.playGlossTap()
                                withAnimation(.spring(response: 0.45)) {
                                    controller.proceedToProbes()
                                }
                            } label: {
                                HStack {
                                    Text(passage.probes.first?.isPictureMatch == true
                                         ? "Continue to Picture-Match Probes (\(passage.probes.count))"
                                         : "Continue to Comprehension Probes (\(passage.probes.count))")
                                        .font(.headline)
                                    Spacer()
                                    Image(systemName: "arrow.right")
                                }
                                .padding(.horizontal, 22)
                                .padding(.vertical, 16)
                                .foregroundStyle(CadenciaPalette.obsidian)
                                .background(CadenciaPalette.saffronGold, in: Capsule())
                            }
                            .buttonStyle(.plain)
                            .padding(.top, 8)
                        }
                    }

                    if passage.probes.isEmpty || controller.probeAnswers.count >= passage.probes.count {
                        Button {
                            HapticDirector.shared.playStrongSuccess()
                            Task { await controller.completeCurrentPassageAndAdvance() }
                        } label: {
                            HStack {
                                Text("Log Learning Events & Next Passage")
                                    .font(.headline)
                                Spacer()
                                Image(systemName: "sparkles")
                            }
                            .padding(.horizontal, 22)
                            .padding(.vertical, 17)
                            .foregroundStyle(CadenciaPalette.obsidian)
                            .background(
                                LinearGradient(
                                    colors: [CadenciaPalette.mediterraneanJade, CadenciaPalette.saffronGold],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                ),
                                in: Capsule()
                            )
                        }
                        .buttonStyle(.plain)
                        .padding(.top, 6)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 100)
            }

            // Bottom floating Liquid Glass transport bar
            audioTransportBar(passage)
                .padding(.horizontal, 18)
                .padding(.bottom, 12)
        }
    }

    private func sentenceCard(
        sentence: PassageSentence,
        startingWordIndex: Int,
        newLemmas: Set<String>,
        isBootstrap: Bool
    ) -> some View {
        let rawWords = sentence.spanish.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        let isReplayed = controller.replayedSentenceIndices.contains(sentence.id)

        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 10) {
                // Bootstrap Sessions 1-4: every sentence carries a visual picture cue (Revised Protocol p. 8)
                if isBootstrap {
                    let symbolName = (sentence.visualSymbol?.isEmpty == false ? sentence.visualSymbol! : "sparkles")
                    Image(systemName: symbolName)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(CadenciaPalette.saffronGold)
                        .frame(width: 34, height: 34)
                        .background(CadenciaPalette.saffronGold.opacity(0.16), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                }

                WrappingWordsView(
                    words: rawWords,
                    startingIndex: startingWordIndex,
                    highlightedIndex: controller.highlightedWordIndex,
                    newLemmas: newLemmas,
                    lemmatizer: controller.appState.lemmatizer
                ) { tappedSurface in
                    Task { await controller.handleWordTapped(surface: tappedSurface) }
                }

                // Sentence replay button (logs Doubt!)
                Button {
                    Task { await controller.replaySentence(sentence) }
                } label: {
                    Image(systemName: isReplayed ? "arrow.counterclockwise.circle.fill" : "speaker.wave.2")
                        .font(.subheadline)
                        .foregroundStyle(isReplayed ? CadenciaPalette.saffronGold : CadenciaPalette.mutedSilver)
                        .frame(width: 32, height: 32)
                        .background(Color.white.opacity(0.06), in: Circle())
                }
                .buttonStyle(.plain)
            }

            // Bootstrap Sessions 1-4: tap to toggle English sentence line (PDF p. 7)
            if isBootstrap, let eng = sentence.englishLine, !eng.isEmpty {
                if revealedBootstrapLineIds.contains(sentence.id) {
                    Text(eng)
                        .font(.caption)
                        .foregroundStyle(CadenciaPalette.saffronGold)
                        .padding(.top, 2)
                } else {
                    Button("Show English line") {
                        HapticDirector.shared.playGlossTap()
                        revealedBootstrapLineIds.insert(sentence.id)
                    }
                    .font(CadenciaFont.mono(10, weight: .semibold))
                    .foregroundStyle(CadenciaPalette.mutedSilver)
                }
            }
        }
        .padding(16)
        .cadenciaGlassCard(cornerRadius: 20, tint: isReplayed ? CadenciaPalette.saffronGold : nil)
    }

    // MARK: - Load-Bearing & Bootstrap Picture-Matching Probes (Rule 5)

    private func probesSection(_ passage: GeneratedPassage) -> some View {
        let hasPictureProbes = passage.probes.contains(where: \.isPictureMatch)
        return VStack(alignment: .leading, spacing: 16) {
            Label(hasPictureProbes ? "PICTURE-MATCHING PROBES" : "LOAD-BEARING PROBES", systemImage: hasPictureProbes ? "photo.on.rectangle.angled" : "scope")
                .font(CadenciaFont.mono(11, weight: .bold))
                .tracking(2)
                .foregroundStyle(CadenciaPalette.saffronGold)

            ForEach(passage.probes) { probe in
                let chosen = controller.probeAnswers[probe.id]
                VStack(alignment: .leading, spacing: 12) {
                    if probe.isPictureMatch, let sentEs = probe.targetSentenceSpanish, !sentEs.isEmpty {
                        HStack(alignment: .center, spacing: 10) {
                            Text(sentEs)
                                .font(CadenciaFont.proseSerif(20))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Button {
                                Task {
                                    await controller.speakSpanishLine(
                                        sentEs,
                                        voice: passage.voice,
                                        accent: passage.accent
                                    )
                                }
                            } label: {
                                Image(systemName: "speaker.wave.2.fill")
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(CadenciaPalette.saffronGold)
                                    .frame(width: 34, height: 34)
                                    .background(CadenciaPalette.saffronGold.opacity(0.16), in: Circle())
                            }
                            .buttonStyle(.plain)
                        }
                        .padding(12)
                        .background(CadenciaPalette.saffronGold.opacity(0.10), in: RoundedRectangle(cornerRadius: 14, style: .continuous))

                        Text(probe.questionEn)
                            .font(CadenciaFont.uiCaption(13, weight: .semibold))
                            .foregroundStyle(CadenciaPalette.mutedSilver)
                    } else {
                        HStack(spacing: 8) {
                            Text(probe.questionEn)
                                .font(CadenciaFont.uiTitle(16))
                                .foregroundStyle(CadenciaPalette.ivoryText)
                            if probe.isPictureMatch {
                                Spacer()
                                Button {
                                    Task {
                                        await controller.speakSpanishLine(
                                            probe.targetDueLemma,
                                            voice: passage.voice,
                                            accent: passage.accent
                                        )
                                    }
                                } label: {
                                    Image(systemName: "speaker.wave.2.fill")
                                        .font(.caption.weight(.bold))
                                        .foregroundStyle(CadenciaPalette.saffronGold)
                                        .frame(width: 32, height: 32)
                                        .background(CadenciaPalette.saffronGold.opacity(0.16), in: Circle())
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }

                    if probe.isPictureMatch {
                        let symbols = probe.optionSymbols ?? []
                        LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 10) {
                            ForEach(Array(probe.options.enumerated()), id: \.offset) { idx, opt in
                                let isChosen = chosen == idx
                                let isCorrect = idx == probe.correctOptionIndex
                                let bg: Color = {
                                    guard chosen != nil else { return Color.white.opacity(0.07) }
                                    if isCorrect { return CadenciaPalette.mediterraneanJade.opacity(0.28) }
                                    if isChosen { return CadenciaPalette.crimsonAlert.opacity(0.28) }
                                    return Color.white.opacity(0.04)
                                }()
                                let icon = idx < symbols.count && !symbols[idx].isEmpty ? symbols[idx] : "sparkles"

                                Button {
                                    Task { await controller.selectProbeOption(probe: probe, optionIndex: idx) }
                                } label: {
                                    VStack(spacing: 8) {
                                        Image(systemName: icon)
                                            .font(.system(size: 26, weight: .semibold))
                                            .foregroundStyle(
                                                chosen != nil && isCorrect ? CadenciaPalette.mediterraneanJade :
                                                (isChosen && !isCorrect ? CadenciaPalette.crimsonAlert : CadenciaPalette.saffronGold)
                                            )
                                        Text(opt)
                                            .font(.subheadline.weight(.semibold))
                                            .foregroundStyle(CadenciaPalette.ivoryText)
                                            .multilineTextAlignment(.center)
                                            .lineLimit(2)
                                    }
                                    .frame(maxWidth: .infinity, minHeight: 84)
                                    .padding(10)
                                    .background(bg, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                                }
                                .buttonStyle(.plain)
                                .disabled(chosen != nil)
                            }
                        }
                    } else {
                        ForEach(Array(probe.options.enumerated()), id: \.offset) { idx, opt in
                            let isChosen = chosen == idx
                            let isCorrect = idx == probe.correctOptionIndex
                            let bg: Color = {
                                guard chosen != nil else { return Color.white.opacity(0.07) }
                                if isCorrect { return CadenciaPalette.mediterraneanJade.opacity(0.28) }
                                if isChosen { return CadenciaPalette.crimsonAlert.opacity(0.28) }
                                return Color.white.opacity(0.04)
                            }()

                            Button {
                                Task { await controller.selectProbeOption(probe: probe, optionIndex: idx) }
                            } label: {
                                HStack {
                                    Text(opt)
                                        .font(.subheadline)
                                        .foregroundStyle(CadenciaPalette.ivoryText)
                                        .multilineTextAlignment(.leading)
                                    Spacer()
                                    if chosen != nil && isCorrect {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundStyle(CadenciaPalette.mediterraneanJade)
                                    } else if isChosen && !isCorrect {
                                        Image(systemName: "xmark.circle.fill")
                                            .foregroundStyle(CadenciaPalette.crimsonAlert)
                                    }
                                }
                                .padding(13)
                                .background(bg, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                            }
                            .buttonStyle(.plain)
                            .disabled(chosen != nil)
                        }
                    }

                    if chosen != nil {
                        Text("Hinge item: «\(probe.targetDueLemma)» — \(probe.hingeExplanation)")
                            .font(.caption)
                            .foregroundStyle(CadenciaPalette.saffronGold)
                    }
                }
                .padding(18)
                .cadenciaGlassCard(cornerRadius: 22)
            }
        }
    }

    // MARK: - Morphing Liquid Glass Transport Bar & Tap-to-Gloss Popover

    private func audioTransportBar(_ passage: GeneratedPassage) -> some View {
        CadenciaGlassContainer(spacing: 12) {
            HStack(spacing: 12) {
                Button {
                    HapticDirector.shared.playGlossTap()
                    Task { await controller.togglePlayPause() }
                } label: {
                    Image(systemName: controller.isPlayingAudio ? "pause.fill" : "play.fill")
                        .font(.title3)
                        .foregroundStyle(CadenciaPalette.obsidian)
                        .frame(width: 48, height: 48)
                        .background(CadenciaPalette.saffronGold, in: Circle())
                }
                .buttonStyle(.plain)

                VStack(alignment: .leading, spacing: 2) {
                    Text("Phonon-40m-es · Native")
                        .font(CadenciaFont.mono(10, weight: .bold))
                        .foregroundStyle(CadenciaPalette.saffronGold)
                    Text("\(passage.voice.rawValue) (\(passage.voice.localeLabel)) · \(controller.currentWPMOverride) WPM")
                        .font(.caption)
                        .foregroundStyle(CadenciaPalette.ivoryText)
                }

                Spacer()

                // Pitch-preserving WPM speed selector (70 -> 160 WPM)
                HStack(spacing: 6) {
                    Button {
                        controller.currentWPMOverride = max(70, controller.currentWPMOverride - 10)
                        Task { await controller.appState.audioPlayer.updatePlaybackRate(targetWPM: controller.currentWPMOverride) }
                    } label: {
                        Image(systemName: "minus")
                            .frame(width: 30, height: 30)
                            .background(Color.white.opacity(0.10), in: Circle())
                    }
                    Button {
                        controller.currentWPMOverride = min(160, controller.currentWPMOverride + 10)
                        Task { await controller.appState.audioPlayer.updatePlaybackRate(targetWPM: controller.currentWPMOverride) }
                    } label: {
                        Image(systemName: "plus")
                            .frame(width: 30, height: 30)
                            .background(Color.white.opacity(0.10), in: Circle())
                    }
                }
                .foregroundStyle(CadenciaPalette.ivoryText)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .cadenciaGlassPill()
        }
    }

    private func glossPopoverCard(item: SyllabusItem, clue: String?, surface: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Image(systemName: item.visualSymbol)
                    .font(.title2)
                    .foregroundStyle(CadenciaPalette.saffronGold)
                    .frame(width: 42, height: 42)
                    .background(CadenciaPalette.saffronGold.opacity(0.18), in: Circle())
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(surface)  (\(item.spanish))")
                        .font(CadenciaFont.displaySerif(20, weight: .bold))
                        .foregroundStyle(CadenciaPalette.ivoryText)
                    Text("/\(item.ipaEs)/ · \(item.pos) · \(item.cefr.rawValue)")
                        .font(CadenciaFont.mono(11))
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
                Spacer()
                Button {
                    withAnimation(.spring(response: 0.3)) { controller.selectedGlossItem = nil }
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundStyle(CadenciaPalette.mutedSilver)
                }
            }

            Text(item.englishGloss)
                .font(.headline)
                .foregroundStyle(CadenciaPalette.saffronGold)

            if let clue, !clue.isEmpty {
                Text(clue)
                    .font(.caption)
                    .foregroundStyle(CadenciaPalette.mutedSilver)
            }
        }
        .padding(18)
        .cadenciaGlassCard(cornerRadius: 22, tint: CadenciaPalette.saffronGold)
    }
}

// MARK: - Flowing Tap-to-Gloss Spanish Word Layout (iOS 16+ SwiftUI Layout Protocol)

private struct WrappingWordsView: View {
    let words: [String]
    let startingIndex: Int
    let highlightedIndex: Int
    let newLemmas: Set<String>
    let lemmatizer: SpanishLemmatizer
    let onTapWord: (String) -> Void

    var body: some View {
        FlowTextLayout(horizontalSpacing: 5, verticalSpacing: 6) {
            ForEach(Array(words.enumerated()), id: \.offset) { offset, word in
                let globalIdx = startingIndex + offset
                let isHighlighted = globalIdx == highlightedIndex
                let clean = word.lowercased().trimmingCharacters(in: CharacterSet.letters.inverted)
                let lem = lemmatizer.lookupLemma(lower: clean).lemma
                let isNewWord = newLemmas.contains(lem) || newLemmas.contains(clean)

                Button {
                    onTapWord(word)
                } label: {
                    Text(word)
                        .font(CadenciaFont.proseSerif(21))
                        .foregroundStyle(
                            isHighlighted ? CadenciaPalette.obsidian :
                            (isNewWord ? CadenciaPalette.saffronGold : CadenciaPalette.ivoryText)
                        )
                        .padding(.horizontal, isHighlighted ? 6 : 2)
                        .padding(.vertical, 1)
                        .background(
                            isHighlighted
                                ? AnyShapeStyle(CadenciaPalette.saffronGold)
                                : AnyShapeStyle(Color.clear),
                            in: RoundedRectangle(cornerRadius: 6, style: .continuous)
                        )
                        .overlay(alignment: .bottom) {
                            if isNewWord && !isHighlighted {
                                Rectangle()
                                    .fill(CadenciaPalette.saffronGold.opacity(0.65))
                                    .frame(height: 1.5)
                            }
                        }
                }
                .buttonStyle(.plain)
            }
        }
    }
}

private struct FlowTextLayout: Layout {
    var horizontalSpacing: CGFloat = 5
    var verticalSpacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxW = proposal.width ?? 320
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowH: CGFloat = 0
        for sub in subviews {
            let sz = sub.sizeThatFits(.unspecified)
            if x + sz.width > maxW && x > 0 {
                x = 0
                y += rowH + verticalSpacing
                rowH = 0
            }
            x += sz.width + horizontalSpacing
            rowH = max(rowH, sz.height)
        }
        return CGSize(width: maxW, height: y + rowH)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX
        var y = bounds.minY
        var rowH: CGFloat = 0
        for sub in subviews {
            let sz = sub.sizeThatFits(.unspecified)
            if x + sz.width > bounds.maxX && x > bounds.minX {
                x = bounds.minX
                y += rowH + verticalSpacing
                rowH = 0
            }
            sub.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(sz))
            x += sz.width + horizontalSpacing
            rowH = max(rowH, sz.height)
        }
    }
}
