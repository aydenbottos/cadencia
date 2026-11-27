import Accelerate
import AVFoundation
import Foundation

// MARK: - Word Timing Alignment & Synthesized Clip

public struct WordTimingSpan: Sendable, Codable, Hashable {
    public let wordIndex: Int
    public let word: String
    public let startTime: TimeInterval
    public let endTime: TimeInterval

    public init(wordIndex: Int, word: String, startTime: TimeInterval, endTime: TimeInterval) {
        self.wordIndex = wordIndex
        self.word = word
        self.startTime = startTime
        self.endTime = endTime
    }
}

public struct SynthesizedAudioClip: Sendable {
    public let samples: [Float]
    public let sampleRate: Double
    public let durationSeconds: TimeInterval
    public let wordTimings: [WordTimingSpan]
    public let synthesisLatencyMs: Double
    public let realTimeFactor: Double
    public let cachedFileURL: URL?

    public init(
        samples: [Float],
        sampleRate: Double,
        durationSeconds: TimeInterval,
        wordTimings: [WordTimingSpan],
        synthesisLatencyMs: Double,
        realTimeFactor: Double,
        cachedFileURL: URL?
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.durationSeconds = durationSeconds
        self.wordTimings = wordTimings
        self.synthesisLatencyMs = synthesisLatencyMs
        self.realTimeFactor = realTimeFactor
        self.cachedFileURL = cachedFileURL
    }
}

// MARK: - Shared 24 kHz PCM / WAV & Clip Stitching Utilities

public enum TutorAudioCodec {
    public static let sampleRate: Double = 24_000.0

    /// Applies a subtle cosine fade-in / fade-out at utterance boundaries to prevent clicks.
    public static func applyBoundaryFade(_ samples: [Float], sampleRate: Double = TutorAudioCodec.sampleRate, fadeMs: Double = 5.0) -> [Float] {
        guard samples.count > 16 else { return samples }
        var out = samples
        // Soft safety peak clamp without slamming quiet syllables to 0 dBFS
        var maxAbs: Float = 0
        vDSP_maxmgv(out, 1, &maxAbs, vDSP_Length(out.count))
        if maxAbs > 0.96 {
            var scale: Float = 0.95 / maxAbs
            vDSP_vsmul(out, 1, &scale, &out, 1, vDSP_Length(out.count))
        }
        let fadeLen = min(Int(sampleRate * (fadeMs / 1000.0)), out.count / 4)
        if fadeLen > 1 {
            for i in 0..<fadeLen {
                let g = 0.5 * (1.0 - cosf(.pi * Float(i) / Float(fadeLen)))
                out[i] *= g
                out[out.count - 1 - i] *= g
            }
        }
        return out
    }

    /// Estimates proportional word-level karaoke timing spans across an audio utterance.
    public static func estimateWordTimings(
        text: String,
        duration: TimeInterval,
        startWordIndex: Int = 0,
        offset: TimeInterval = 0
    ) -> [WordTimingSpan] {
        let words = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard !words.isEmpty, duration > 0 else { return [] }
        let weights = words.map { w -> Double in
            let core = w.filter(\.isLetter)
            let punctPause = (w.hasSuffix(",") || w.hasSuffix(";"))
                ? 2.5
                : ((w.hasSuffix(".") || w.hasSuffix("!") || w.hasSuffix("?")) ? 4.0 : 0.0)
            return Double(max(2, core.count)) + punctPause
        }
        let totalW = weights.reduce(0, +)
        var spans: [WordTimingSpan] = []
        spans.reserveCapacity(words.count)
        var cur = offset
        for (i, w) in words.enumerated() {
            let dt = duration * (weights[i] / totalW)
            spans.append(WordTimingSpan(wordIndex: startWordIndex + i, word: w, startTime: cur, endTime: cur + dt))
            cur += dt
        }
        return spans
    }

    /// Seamlessly stitches multiple 24 kHz mono clips (e.g., Spanish Phonon + English Kokoro turns)
    /// with a clean breath pause between clips and zero resampling.
    public static func stitchClips(
        _ clips: [SynthesizedAudioClip],
        pauseSeconds: TimeInterval = 0.13,
        totalSynthesisLatencyMs: Double? = nil
    ) -> SynthesizedAudioClip {
        let sr = clips.first?.sampleRate ?? sampleRate
        let pauseSamples = max(0, Int(sr * pauseSeconds))
        let silence = [Float](repeating: 0, count: pauseSamples)

        var combined: [Float] = []
        var timings: [WordTimingSpan] = []
        var cursor: TimeInterval = 0
        var wordCursor = 0
        var sumLatencyMs: Double = 0

        for (idx, clip) in clips.enumerated() {
            guard !clip.samples.isEmpty else { continue }
            combined.append(contentsOf: clip.samples)
            for span in clip.wordTimings {
                timings.append(WordTimingSpan(
                    wordIndex: wordCursor,
                    word: span.word,
                    startTime: cursor + span.startTime,
                    endTime: cursor + span.endTime
                ))
                wordCursor += 1
            }
            cursor += clip.durationSeconds
            sumLatencyMs += clip.synthesisLatencyMs
            if idx + 1 < clips.count && pauseSamples > 0 {
                combined.append(contentsOf: silence)
                cursor += pauseSeconds
            }
        }

        let totalDuration = Double(combined.count) / sr
        let latencyMs = totalSynthesisLatencyMs ?? sumLatencyMs
        let rtf = totalDuration > 0 ? (latencyMs / 1000.0) / totalDuration : 0
        return SynthesizedAudioClip(
            samples: combined,
            sampleRate: sr,
            durationSeconds: totalDuration,
            wordTimings: timings,
            synthesisLatencyMs: latencyMs,
            realTimeFactor: rtf,
            cachedFileURL: nil
        )
    }

    public static func saveWAV(samples: [Float], sampleRate: Double, to url: URL) throws {
        var data = Data()
        let numSamples = samples.count
        let dataBytes = UInt32(numSamples * 2)
        let chunkSize = 36 + dataBytes
        let sr = UInt32(sampleRate)
        let byteRate = sr * 2

        data.append(contentsOf: "RIFF".utf8)
        withUnsafeBytes(of: chunkSize.littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: "WAVEfmt ".utf8)
        withUnsafeBytes(of: UInt32(16).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt16(1).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt16(1).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: sr.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: byteRate.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt16(2).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt16(16).littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: "data".utf8)
        withUnsafeBytes(of: dataBytes.littleEndian) { data.append(contentsOf: $0) }

        var pcm16 = [Int16](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            let clamped = max(-1.0, min(1.0, samples[i]))
            pcm16[i] = Int16(clamped * 32767.0)
        }
        pcm16.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url, options: [.atomic])
    }

    public static func loadWAV(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count > 44 else { return [] }
        let pcmData = data.subdata(in: 44..<data.count)
        let count = pcmData.count / 2
        var samples = [Float](repeating: 0, count: count)
        pcmData.withUnsafeBytes { buf in
            let src = buf.bindMemory(to: Int16.self)
            for i in 0..<count { samples[i] = Float(src[i]) / 32768.0 }
        }
        return samples
    }
}

// MARK: - Unified Low-Latency AVAudioEngine Playback (Direct 1.0× PCM or Pitch-Preserving Pace Scaling)

public actor TutorAudioPlayer {
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var timePitchNode: AVAudioUnitTimePitch?
    private var activePlaybackFinish: (@Sendable () -> Void)?
    private var onAudioLevelUpdate: (@Sendable (Float) -> Void)?

    public init() {}

    public func setAudioLevelHandler(_ handler: (@Sendable (Float) -> Void)?) {
        self.onAudioLevelUpdate = handler
    }

    /// Plays any 24 kHz clip cleanly through AVAudioEngine:
    /// - Direct `player -> mainMixerNode` near 1.0× (zero time-stretch DSP artifacts)
    /// - Attaches `AVAudioUnitTimePitch` only when pace scaling deviates by >= 4%
    public func play(
        clip: SynthesizedAudioClip,
        targetWPM: Int = 140,
        baselineNativeWPM: Double = 140.0
    ) async throws {
        stopPlayback()
        guard !clip.samples.isEmpty else {
            throw NSError(domain: "TutorAudioPlayer", code: 504, userInfo: [
                NSLocalizedDescriptionKey: "Synthesized clip contains 0 audio samples."
            ])
        }

        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio, options: [])
        try session.setActive(true)
        #endif

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        engine.attach(player)

        guard let format = AVAudioFormat(standardFormatWithSampleRate: clip.sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(clip.samples.count))
        else { return }

        buffer.frameLength = AVAudioFrameCount(clip.samples.count)
        if let channel = buffer.floatChannelData?[0] {
            clip.samples.withUnsafeBufferPointer { ptr in
                channel.update(from: ptr.baseAddress!, count: clip.samples.count)
            }
        }

        let rate = Float(max(0.68, min(1.30, Double(targetWPM) / baselineNativeWPM)))
        let needsTimePitch = abs(rate - 1.0) >= 0.04
        var attachedTimePitch: AVAudioUnitTimePitch?

        if needsTimePitch {
            let tp = AVAudioUnitTimePitch()
            tp.rate = rate
            tp.pitch = 0.0
            engine.attach(tp)
            engine.connect(player, to: tp, format: format)
            engine.connect(tp, to: engine.mainMixerNode, format: format)
            attachedTimePitch = tp
        } else {
            engine.connect(player, to: engine.mainMixerNode, format: format)
        }

        let levelCallback = self.onAudioLevelUpdate
        engine.mainMixerNode.installTap(onBus: 0, bufferSize: 1024, format: nil) { pcm, _ in
            guard let data = pcm.floatChannelData?[0], pcm.frameLength > 0 else { return }
            var rms: Float = 0
            vDSP_rmsqv(data, 1, &rms, vDSP_Length(pcm.frameLength))
            levelCallback?(min(1.0, rms * 4.5))
        }

        engine.prepare()
        try engine.start()
        self.audioEngine = engine
        self.playerNode = player
        self.timePitchNode = attachedTimePitch

        final class OnceFlag: @unchecked Sendable {
            private let lock = NSLock()
            private var fired = false
            func fire(_ body: () -> Void) {
                lock.lock()
                defer { lock.unlock() }
                guard !fired else { return }
                fired = true
                body()
            }
        }

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            let once = OnceFlag()
            let finish: @Sendable () -> Void = {
                once.fire { cont.resume() }
            }
            self.activePlaybackFinish = finish
            player.scheduleBuffer(buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) { _ in
                finish()
            }
            player.play()
        }
        self.activePlaybackFinish = nil
        onAudioLevelUpdate?(0.0)
    }

    public func updatePlaybackRate(targetWPM: Int, baselineNativeWPM: Double = 140.0) {
        let rate = Float(max(0.68, min(1.30, Double(targetWPM) / baselineNativeWPM)))
        timePitchNode?.rate = rate
    }

    public func stopPlayback() {
        let finish = activePlaybackFinish
        activePlaybackFinish = nil
        playerNode?.stop()
        audioEngine?.mainMixerNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        playerNode = nil
        timePitchNode = nil
        finish?()
        onAudioLevelUpdate?(0.0)
    }
}
