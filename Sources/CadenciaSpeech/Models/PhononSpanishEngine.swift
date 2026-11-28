import CadenciaCore
import CryptoKit
import Foundation
import PhononCore

private struct SendableOpaquePointer: @unchecked Sendable {
    let raw: OpaquePointer
}

/// Owns a `PttsHandle*` safely across Swift 6 concurrency and guarantees cleanup on deinit.
private final class PhononHandleBox: @unchecked Sendable {
    let raw: OpaquePointer
    private let queue: DispatchQueue

    init(raw: OpaquePointer, queue: DispatchQueue) {
        self.raw = raw
        self.queue = queue
    }

    deinit {
        let ptr = SendableOpaquePointer(raw: raw)
        queue.async {
            ptts_free(ptr.raw)
        }
    }
}

/// Compute placement for Gradium **Phonon-40m-es** Core ML inference.
///
/// - `.fastCPU` (`PTTS_UNIT_CPU`, default): runs in FP32 on Apple Silicon CPU/AMX (~10–14× real-time
///   on iPhone 18 Pro Max, ~55 ms TTFA, highest numerical precision, and **loads in < 0.5 seconds**
///   because it completely bypasses iOS 26/27 `anecompilerservice` cold compilation!).
/// - `.neuralEngine` (`PTTS_UNIT_ANE`): runs in FP16 on Apple Neural Engine (~12–14.5× real-time,
///   lower battery draw, but requires an initial iOS `ANECCompile` compilation step on first load).
public enum PhononComputeMode: String, Sendable, CaseIterable, Identifiable {
    case fastCPU = "cpu"
    case neuralEngine = "ane"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .fastCPU: return "Fast CPU (FP32)"
        case .neuralEngine: return "ANE (FP16)"
        }
    }

    public var badgeLabel: String {
        switch self {
        case .fastCPU: return "CPU FP32"
        case .neuralEngine: return "ANE FP16"
        }
    }

    public var shortHardwareLabel: String {
        switch self {
        case .fastCPU: return "CPU/AMX FP32"
        case .neuralEngine: return "ANE FP16"
        }
    }

    var pttsUnit: UInt32 {
        switch self {
        case .fastCPU: return UInt32(PTTS_UNIT_CPU)
        case .neuralEngine: return UInt32(PTTS_UNIT_ANE)
        }
    }
}

/// Native on-device Gradium **Phonon-40m-es** Spanish TTS engine via `PhononCore.xcframework`
/// (`gradium-ai/xn-ptts` v0.7.1).
///
/// Characteristics:
///  - 24 kHz mono Float32 native Spanish speech (~10–14.5× real-time on Apple Silicon)
///  - Defaults to `.fastCPU` (Apple Silicon AMX FP32) for <0.5s instant load without iOS ANE compile stalls
///  - 8 authentic native Spanish voices (`es-ES` Castilian & `es-MX` Latin American)
///  - Pure-Rust on-device Core ML bundle exporter (`ptts_export_coreml`) — zero Python, zero computer
///  - SHA-256 disk WAV cache for instant repeat word/sentence playback
public actor PhononSpanishEngine {
    public static let sampleRate: Double = TutorAudioCodec.sampleRate

    private let ffiQueue = DispatchQueue(label: "com.aydenbottos.cadencia.phonon.ffi", qos: .userInitiated)
    private var handleBox: PhononHandleBox?
    private let audioCacheDir: URL

    public private(set) var isLoaded: Bool = false
    public private(set) var availableVoices: [String] = []
    public private(set) var activeVoiceName: String = ""
    public private(set) var loadedDirectory: URL?
    public private(set) var activeComputeMode: PhononComputeMode = .fastCPU

    public init(cacheDirectory: URL? = nil) {
        if let cacheDirectory {
            self.audioCacheDir = cacheDirectory
        } else {
            let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            self.audioCacheDir = caches.appendingPathComponent("CadenciaPhononAudioCache", isDirectory: true)
        }
        try? FileManager.default.createDirectory(at: self.audioCacheDir, withIntermediateDirectories: true)
    }

    // MARK: - On-Device Pure-Rust Core ML Exporter

    /// Exports a locally downloaded `gradium/phonon-40m-es` checkpoint + native Spanish voices into an
    /// Apple Core ML bundle (`flow-*.mlpackage`, `mimi-*.mlpackage`, `host.safetensors`, `tokenizer.json`,
    /// `voices/*.safetensors`, `bundle.json`) completely on-device in pure Rust.
    public static func exportCoreMLBundle(
        checkpointDir: URL,
        voicesDir: URL?,
        outputDir: URL
    ) async throws {
        let ckPath = checkpointDir.path
        let vPath = voicesDir?.path
        let outPath = outputDir.path
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                let ok: Bool = ckPath.withCString { cCk in
                    outPath.withCString { cOut in
                        if let vPath {
                            return vPath.withCString { cVoices in
                                ptts_export_coreml(cCk, cVoices, cOut)
                            }
                        } else {
                            return ptts_export_coreml(cCk, nil, cOut)
                        }
                    }
                }
                if ok {
                    cont.resume(returning: ())
                } else {
                    let msg = ptts_last_error(nil).map { String(cString: $0) } ?? "Unknown Phonon Core ML export failure"
                    cont.resume(throwing: NSError(
                        domain: "PhononSpanishEngine",
                        code: 500,
                        userInfo: [NSLocalizedDescriptionKey: "Phonon Core ML export failed: \(msg)"]
                    ))
                }
            }
        }
    }

    // MARK: - Load & Lifecycle

    public func loadModels(
        from bundleDir: URL,
        computeMode: PhononComputeMode = .fastCPU,
        forceReload: Bool = false
    ) async throws {
        if isLoaded && !forceReload && loadedDirectory == bundleDir && activeComputeMode == computeMode {
            return
        }
        if forceReload || isLoaded {
            handleBox = nil
            isLoaded = false
            availableVoices = []
            activeVoiceName = ""
        }

        let dirPath = bundleDir.path
        let queue = self.ffiQueue
        let unit = computeMode.pttsUnit
        let loaded: (PhononHandleBox, [String]) = try await runOnQueue(queue) {
            guard let ptr = dirPath.withCString({ cDir in
                "es".withCString { cLang in
                    ptts_new(cDir, unit, cLang)
                }
            }) else {
                let msg = ptts_last_error(nil).map { String(cString: $0) } ?? "Failed to initialize Phonon-40m-es"
                throw NSError(domain: "PhononSpanishEngine", code: 501, userInfo: [
                    NSLocalizedDescriptionKey: "Phonon-40m-es load failed: \(msg)"
                ])
            }
            _ = ptts_set_temperature(ptr, 0.3)
            let voices = Self.parseNulBlock(ptts_voices(ptr))
            return (PhononHandleBox(raw: ptr, queue: queue), voices)
        }

        self.handleBox = loaded.0
        self.availableVoices = loaded.1
        self.activeVoiceName = loaded.1.first ?? ""
        self.loadedDirectory = bundleDir
        self.activeComputeMode = computeMode
        self.isLoaded = true
    }

    public func unload() {
        handleBox = nil
        isLoaded = false
        availableVoices = []
        activeVoiceName = ""
        loadedDirectory = nil
    }

    // MARK: - Synthesis (Spanish Passage & Sentence)

    public func synthesizePassage(
        sentences: [String],
        voice: TutorVoice,
        accent: SpanishRegionalAccent,
        onSentenceReady: (@Sendable (Int, SynthesizedAudioClip) -> Void)? = nil
    ) async throws -> SynthesizedAudioClip {
        let startWall = Date()
        var clips: [SynthesizedAudioClip] = []
        clips.reserveCapacity(sentences.count)

        for (sIdx, sentence) in sentences.enumerated() {
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let clip = try await synthesizeSentence(
                text: trimmed,
                voice: voice,
                accent: accent
            )
            clips.append(clip)
            onSentenceReady?(sIdx, clip)
        }

        let elapsedMs = Date().timeIntervalSince(startWall) * 1000.0
        return TutorAudioCodec.stitchClips(clips, pauseSeconds: 0.14, totalSynthesisLatencyMs: elapsedMs)
    }

    public func synthesizeSentence(
        text: String,
        voice: TutorVoice,
        accent: SpanishRegionalAccent,
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

        // In Phonon, the voice embedding carries the native regional accent (es-ES vs es-MX)!
        let resolvedVoice = TutorVoice.voice(for: accent, preferred: voice)
        let cacheFile = cacheURL(for: trimmed, voice: resolvedVoice)

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

        guard let box = handleBox, isLoaded else {
            throw NSError(domain: "PhononSpanishEngine", code: 503, userInfo: [
                NSLocalizedDescriptionKey: "Phonon-40m-es is not loaded yet. Install it in Settings -> On-Device Models."
            ])
        }

        let targetVoiceName: String
        if availableVoices.contains(resolvedVoice.phononBundleName) {
            targetVoiceName = resolvedVoice.phononBundleName
        } else if let match = availableVoices.first(where: { $0.caseInsensitiveCompare(resolvedVoice.phononBundleName) == .orderedSame }) {
            targetVoiceName = match
        } else {
            targetVoiceName = availableVoices.first ?? resolvedVoice.phononBundleName
        }

        let needsVoiceSwitch = (targetVoiceName != activeVoiceName && !targetVoiceName.isEmpty)
        let t0 = Date()

        final class PCMCollector {
            var samples: [Float] = []
            init() { samples.reserveCapacity(48_000) }
        }

        let rawSamples: [Float] = try await runOnQueue(ffiQueue) {
            if needsVoiceSwitch {
                let switched = targetVoiceName.withCString { ptts_set_voice(box.raw, $0) }
                if !switched {
                    let msg = ptts_last_error(box.raw).map { String(cString: $0) } ?? "Voice switch failed"
                    throw NSError(domain: "PhononSpanishEngine", code: 504, userInfo: [
                        NSLocalizedDescriptionKey: "Phonon voice switch failed (\(targetVoiceName)): \(msg)"
                    ])
                }
            }

            let collector = PCMCollector()
            let unmanaged = Unmanaged.passRetained(collector)
            defer { unmanaged.release() }

            var result = PttsResult()
            let ok = trimmed.withCString { cText in
                ptts_speak(
                    box.raw,
                    cText,
                    { pcm, count, user in
                        guard let pcm, let user, count > 0 else { return true }
                        let col = Unmanaged<PCMCollector>.fromOpaque(user).takeUnretainedValue()
                        col.samples.append(contentsOf: UnsafeBufferPointer(start: pcm, count: count))
                        return true
                    },
                    unmanaged.toOpaque(),
                    &result
                )
            }
            guard ok else {
                let msg = ptts_last_error(box.raw).map { String(cString: $0) } ?? "Synthesis failed"
                throw NSError(domain: "PhononSpanishEngine", code: 505, userInfo: [
                    NSLocalizedDescriptionKey: "Phonon-40m-es synthesis error: \(msg)"
                ])
            }
            return collector.samples
        }

        if needsVoiceSwitch {
            self.activeVoiceName = targetVoiceName
        }

        let cleanSamples = TutorAudioCodec.applyBoundaryFade(rawSamples, sampleRate: Self.sampleRate, fadeMs: 4.0)
        try? TutorAudioCodec.saveWAV(samples: cleanSamples, sampleRate: Self.sampleRate, to: cacheFile)

        let elapsedMs = Date().timeIntervalSince(t0) * 1000.0
        let duration = Double(cleanSamples.count) / Self.sampleRate
        let rtf = duration > 0 ? (elapsedMs / 1000.0) / duration : 0
        let timings = TutorAudioCodec.estimateWordTimings(
            text: trimmed,
            duration: duration,
            startWordIndex: startingWordIndex,
            offset: timeOffset
        )

        return SynthesizedAudioClip(
            samples: cleanSamples,
            sampleRate: Self.sampleRate,
            durationSeconds: duration,
            wordTimings: timings,
            synthesisLatencyMs: elapsedMs,
            realTimeFactor: rtf,
            cachedFileURL: cacheFile
        )
    }

    // MARK: - Helpers

    private func runOnQueue<T: Sendable>(
        _ queue: DispatchQueue,
        _ body: @escaping @Sendable () throws -> T
    ) async throws -> T {
        try await withCheckedThrowingContinuation { cont in
            queue.async {
                cont.resume(with: Result { try body() })
            }
        }
    }

    private nonisolated static func parseNulBlock(_ block: UnsafePointer<CChar>?) -> [String] {
        var names: [String] = []
        guard var ptr = block else { return names }
        while true {
            let s = String(cString: ptr)
            if s.isEmpty { break }
            names.append(s)
            ptr = ptr.advanced(by: s.utf8.count + 1)
        }
        return names
    }

    private func cacheURL(for text: String, voice: TutorVoice) -> URL {
        let key = "phonon_v1_\(voice.phononBundleName)_\(text)"
        let digest = SHA256.hash(data: Data(key.utf8)).prefix(16).map { String(format: "%02x", $0) }.joined()
        return audioCacheDir.appendingPathComponent("phonon_\(digest).wav")
    }
}
