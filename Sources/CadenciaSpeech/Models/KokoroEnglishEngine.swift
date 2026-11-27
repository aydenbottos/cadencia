import CadenciaCore
import CryptoKit
import FluidAudio
import Foundation

/// On-device **Kokoro-82M** English conversation voice engine backed by `FluidAudio`'s `KokoroAneManager`.
///
/// Key engineering choices for iPhone 18 Pro Max (Apple SME chips / iOS 26–27):
///  - Uses `version: .legacy` + `vocoderPrecision: .float32` (`KokoroVocoder_v2`, FluidAudio #1135) to
///    completely eliminate whispery/robotic fp16 Vocoder onsets on SME silicon!
///  - Uses `computeUnits: .default` (automatically routes Noise+Tail to GPU on iOS 26 and CPU on iOS 27).
///  - Outputs native **24,000 Hz mono Float32** PCM — exact sample-rate match to `Phonon-40m-es`, allowing
///    bilingual Spanish + English tutor turns to stitch seamlessly with zero resampling!
public actor KokoroEnglishEngine {
    public static let sampleRate: Double = TutorAudioCodec.sampleRate
    public static let defaultFemaleVoice = "af_heart"
    public static let defaultMaleVoice = "am_michael"
    public static let preloadedVoices: Set<String> = [defaultFemaleVoice, defaultMaleVoice]

    private var manager: KokoroAneManager?
    private let audioCacheDir: URL

    public private(set) var isLoaded: Bool = false
    public private(set) var loadedDirectory: URL?

    public init(cacheDirectory: URL? = nil) {
        if let cacheDirectory {
            self.audioCacheDir = cacheDirectory
        } else {
            let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            self.audioCacheDir = caches.appendingPathComponent("CadenciaKokoroEnglishCache", isDirectory: true)
        }
        try? FileManager.default.createDirectory(at: self.audioCacheDir, withIntermediateDirectories: true)
    }

    /// Initializes and warms Kokoro-82M English models + English G2P frontend + default voices in `directory`.
    public func loadModels(from directory: URL, forceReload: Bool = false) async throws {
        if isLoaded && !forceReload && loadedDirectory == directory {
            return
        }
        if forceReload, let existing = manager {
            await existing.cleanup()
            manager = nil
            isLoaded = false
        }

        let kokoro = KokoroAneManager(
            variant: .english,
            defaultVoice: Self.defaultFemaleVoice,
            directory: directory,
            computeUnits: .default,
            version: .legacy,
            vocoderPrecision: .float32
        )
        try await kokoro.initialize(preloadVoices: Self.preloadedVoices)

        self.manager = kokoro
        self.loadedDirectory = directory
        self.isLoaded = true
    }

    public func unload() async {
        if let existing = manager {
            await existing.cleanup()
        }
        manager = nil
        isLoaded = false
        loadedDirectory = nil
    }

    /// Synthesizes an English tutor utterance at 24 kHz mono Float32 using Kokoro-82M + disk WAV cache.
    public func synthesizeEnglish(
        text: String,
        voice: String = KokoroEnglishEngine.defaultFemaleVoice,
        speed: Float = 1.0,
        startingWordIndex: Int = 0,
        timeOffset: TimeInterval = 0
    ) async throws -> SynthesizedAudioClip {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return SynthesizedAudioClip(
                samples: [],
                sampleRate: Self.sampleRate,
                durationSeconds: 0,
                wordTimings: [],
                synthesisLatencyMs: 0,
                realTimeFactor: 0,
                cachedFileURL: nil
            )
        }

        let cacheFile = cacheURL(for: trimmed, voice: voice, speed: speed)
        if FileManager.default.fileExists(atPath: cacheFile.path),
           let cached = try? TutorAudioCodec.loadWAV(from: cacheFile),
           !cached.isEmpty {
            let dur = Double(cached.count) / Self.sampleRate
            let timings = TutorAudioCodec.estimateWordTimings(
                text: trimmed,
                duration: dur,
                startWordIndex: startingWordIndex,
                offset: timeOffset
            )
            return SynthesizedAudioClip(
                samples: cached,
                sampleRate: Self.sampleRate,
                durationSeconds: dur,
                wordTimings: timings,
                synthesisLatencyMs: 0.4,
                realTimeFactor: 0.001,
                cachedFileURL: cacheFile
            )
        }

        guard let kokoro = manager, isLoaded else {
            throw NSError(domain: "KokoroEnglishEngine", code: 503, userInfo: [
                NSLocalizedDescriptionKey: "Kokoro-82M English tutor voice is not loaded yet. Install it in Settings -> On-Device Models."
            ])
        }

        let t0 = Date()
        let result = try await kokoro.synthesizeDetailed(text: trimmed, voice: voice, speed: speed)
        let cleanSamples = TutorAudioCodec.applyBoundaryFade(
            result.samples,
            sampleRate: Double(result.sampleRate),
            fadeMs: 4.0
        )
        try? TutorAudioCodec.saveWAV(samples: cleanSamples, sampleRate: Double(result.sampleRate), to: cacheFile)

        let elapsedMs = Date().timeIntervalSince(t0) * 1000.0
        let duration = Double(cleanSamples.count) / Double(result.sampleRate)
        let rtf = duration > 0 ? (elapsedMs / 1000.0) / duration : 0
        let timings = TutorAudioCodec.estimateWordTimings(
            text: trimmed,
            duration: duration,
            startWordIndex: startingWordIndex,
            offset: timeOffset
        )

        return SynthesizedAudioClip(
            samples: cleanSamples,
            sampleRate: Double(result.sampleRate),
            durationSeconds: duration,
            wordTimings: timings,
            synthesisLatencyMs: elapsedMs,
            realTimeFactor: rtf,
            cachedFileURL: cacheFile
        )
    }

    // MARK: - Unified Bilingual Conversation Turn Stitcher (Phonon-40m-es + Kokoro-82M)

    /// Synthesizes an ordered bilingual conversation turn by routing:
    ///  - `.spanish` segments -> `PhononSpanishEngine` (`Phonon-40m-es`, 24 kHz)
    ///  - `.english` segments -> `KokoroEnglishEngine` (`Kokoro-82M`, 24 kHz)
    /// and stitching all chunks seamlessly at 24 kHz with a natural 130 ms breath pause between switches.
    public static func synthesizeBilingualTurn(
        segments: [TutorSpeechSegment],
        spanishVoice: TutorVoice,
        accent: SpanishRegionalAccent,
        spanishEngine: PhononSpanishEngine,
        englishEngine: KokoroEnglishEngine
    ) async throws -> SynthesizedAudioClip {
        let startWall = Date()
        let resolvedSpanishVoice = TutorVoice.voice(for: accent, preferred: spanishVoice)
        let pairedEnglishVoice = resolvedSpanishVoice.pairedKokoroEnglishVoice

        let spanishReady = await spanishEngine.isLoaded
        let englishReady = await englishEngine.isLoaded

        var rendered: [SynthesizedAudioClip] = []
        rendered.reserveCapacity(segments.count)

        for seg in segments {
            let text = seg.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            switch seg.language {
            case .spanish:
                guard spanishReady else { continue }
                let clip = try await spanishEngine.synthesizeSentence(
                    text: text,
                    voice: resolvedSpanishVoice,
                    accent: accent
                )
                if !clip.samples.isEmpty { rendered.append(clip) }
            case .english:
                guard englishReady else { continue }
                let clip = try await englishEngine.synthesizeEnglish(
                    text: text,
                    voice: pairedEnglishVoice,
                    speed: 1.0
                )
                if !clip.samples.isEmpty { rendered.append(clip) }
            }
        }

        let elapsedMs = Date().timeIntervalSince(startWall) * 1000.0
        return TutorAudioCodec.stitchClips(rendered, pauseSeconds: 0.13, totalSynthesisLatencyMs: elapsedMs)
    }

    private func cacheURL(for text: String, voice: String, speed: Float) -> URL {
        let key = String(format: "kokoro_en_v1_%@_%.2f_%@", voice, speed, text)
        let digest = SHA256.hash(data: Data(key.utf8)).prefix(16).map { String(format: "%02x", $0) }.joined()
        return audioCacheDir.appendingPathComponent("kokoro_\(digest).wav")
    }
}
