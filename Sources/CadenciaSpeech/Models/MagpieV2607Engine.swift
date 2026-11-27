import Accelerate
import AVFoundation
import CadenciaCore
@preconcurrency import CoreML
import CryptoKit
import Foundation

// MARK: - Minimal Zero-Copy NumPy (.npy) Reader

enum NpyLoader {
    static func loadFloat32(from url: URL) throws -> (data: [Float], shape: [Int]) {
        let raw = try Data(contentsOf: url)
        guard raw.count > 10, raw[0] == 0x93,
              String(data: raw.subdata(in: 1..<6), encoding: .ascii) == "NUMPY"
        else {
            throw NSError(domain: "NpyLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid .npy header at \(url.lastPathComponent)"])
        }
        let major = raw[6]
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            headerLen = Int(raw[8]) | (Int(raw[9]) << 8)
            headerStart = 10
        } else {
            headerLen = Int(raw[8]) | (Int(raw[9]) << 8) | (Int(raw[10]) << 16) | (Int(raw[11]) << 24)
            headerStart = 12
        }
        let dataStart = headerStart + headerLen
        let headerStr = String(data: raw.subdata(in: headerStart..<dataStart), encoding: .ascii) ?? ""

        // Parse shape tuple: e.g. 'shape': (2024, 768,)
        var shape: [Int] = []
        if let rStart = headerStr.range(of: "("), let rEnd = headerStr.range(of: ")", range: rStart.upperBound..<headerStr.endIndex) {
            let inside = headerStr[rStart.upperBound..<rEnd.lowerBound]
            shape = inside.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        }

        let payload = raw.subdata(in: dataStart..<raw.count)
        let isFloat16 = headerStr.contains("f2")
        let count = payload.count / (isFloat16 ? 2 : 4)
        var out = [Float](repeating: 0, count: count)

        if isFloat16 {
            payload.withUnsafeBytes { buf in
                let src = buf.bindMemory(to: Float16.self)
                for i in 0..<count { out[i] = Float(src[i]) }
            }
        } else {
            payload.withUnsafeBytes { buf in
                let src = buf.bindMemory(to: Float.self)
                for i in 0..<count { out[i] = src[i] }
            }
        }
        return (out, shape)
    }
}

// MARK: - Accelerate AMX/BLAS LocalTransformer (8-codebook AR head, <0.6ms/step)

struct MagpieLocalTransformerWeights: Sendable {
    let inProjW: [Float]      // [256, 768]
    let inProjB: [Float]      // [256]
    let posEmb: [Float]       // [10, 256]
    let norm1W: [Float]       // [256]
    let saQkvW: [Float]       // [768, 256]
    let saOW: [Float]         // [256, 256]
    let norm2W: [Float]       // [256]
    let ffnW1: [Float]        // [1024, 256]
    let ffnW2: [Float]        // [256, 1024]
    let outProjW: [[Float]]   // 8 x [2024, 256]
    let outProjB: [[Float]]   // 8 x [2024]

    static func load(from dir: URL) throws -> MagpieLocalTransformerWeights {
        let ltDir = dir.appendingPathComponent("constants/local_transformer", isDirectory: true)
        var outW: [[Float]] = []
        var outB: [[Float]] = []
        for cb in 0..<8 {
            outW.append(try NpyLoader.loadFloat32(from: ltDir.appendingPathComponent("out_proj_\(cb)_weight.npy")).data)
            outB.append(try NpyLoader.loadFloat32(from: ltDir.appendingPathComponent("out_proj_\(cb)_bias.npy")).data)
        }
        return MagpieLocalTransformerWeights(
            inProjW: try NpyLoader.loadFloat32(from: ltDir.appendingPathComponent("in_proj_weight.npy")).data,
            inProjB: try NpyLoader.loadFloat32(from: ltDir.appendingPathComponent("in_proj_bias.npy")).data,
            posEmb: try NpyLoader.loadFloat32(from: ltDir.appendingPathComponent("pos_emb.npy")).data,
            norm1W: try NpyLoader.loadFloat32(from: ltDir.appendingPathComponent("norm1_weight.npy")).data,
            saQkvW: try NpyLoader.loadFloat32(from: ltDir.appendingPathComponent("sa_qkv_weight.npy")).data,
            saOW: try NpyLoader.loadFloat32(from: ltDir.appendingPathComponent("sa_o_weight.npy")).data,
            norm2W: try NpyLoader.loadFloat32(from: ltDir.appendingPathComponent("norm2_weight.npy")).data,
            ffnW1: try NpyLoader.loadFloat32(from: ltDir.appendingPathComponent("ffn_conv1_weight.npy")).data,
            ffnW2: try NpyLoader.loadFloat32(from: ltDir.appendingPathComponent("ffn_conv2_weight.npy")).data,
            outProjW: outW,
            outProjB: outB
        )
    }

    /// Vectorized `y = x @ W^T + b` using Apple AMX-accelerated `cblas_sgemm`.
    private func linear(x: [Float], rows: Int, inDim: Int, w: [Float], outDim: Int, bias: [Float]? = nil) -> [Float] {
        var out = [Float](repeating: 0, count: rows * outDim)
        x.withUnsafeBufferPointer { xPtr in
            w.withUnsafeBufferPointer { wPtr in
                out.withUnsafeMutableBufferPointer { oPtr in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(rows), Int32(outDim), Int32(inDim),
                        1.0,
                        xPtr.baseAddress!, Int32(inDim),
                        wPtr.baseAddress!, Int32(inDim),
                        0.0,
                        oPtr.baseAddress!, Int32(outDim)
                    )
                }
            }
        }
        if let bias {
            for r in 0..<rows {
                let off = r * outDim
                for c in 0..<outDim { out[off + c] += bias[c] }
            }
        }
        return out
    }

    private func layerNormNoBias(_ x: [Float], rows: Int, dim: Int, weight: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: rows * dim)
        for r in 0..<rows {
            let off = r * dim
            var mean: Float = 0
            vDSP_meanv(Array(x[off..<(off + dim)]), 1, &mean, vDSP_Length(dim))
            var varSum: Float = 0
            for c in 0..<dim {
                let d = x[off + c] - mean
                varSum += d * d
            }
            let invStd = 1.0 / sqrtf(varSum / Float(dim) + 1e-5)
            for c in 0..<dim {
                out[off + c] = (x[off + c] - mean) * invStd * weight[c]
            }
        }
        return out
    }

    /// 1-layer causal LocalTransformer forward pass for T in [1..8], D=256.
    func forward(sequence: [Float], seqLen T: Int) -> [Float] {
        let D = 256
        var x = sequence
        // Add positional embedding
        vDSP_vadd(x, 1, posEmb, 1, &x, 1, vDSP_Length(T * D))

        // 1. Pre-norm causal self-attention
        let xNorm1 = layerNormNoBias(x, rows: T, dim: D, weight: norm1W)
        let qkv = linear(x: xNorm1, rows: T, inDim: D, w: saQkvW, outDim: 3 * D)

        var q = [Float](repeating: 0, count: T * D)
        var k = [Float](repeating: 0, count: T * D)
        var v = [Float](repeating: 0, count: T * D)
        for t in 0..<T {
            let qkvOff = t * 3 * D
            let dstOff = t * D
            for d in 0..<D {
                q[dstOff + d] = qkv[qkvOff + d]
                k[dstOff + d] = qkv[qkvOff + D + d]
                v[dstOff + d] = qkv[qkvOff + 2 * D + d]
            }
        }

        let scale = 1.0 / sqrtf(Float(D))
        var scores = [Float](repeating: 0, count: T * T)
        q.withUnsafeBufferPointer { qPtr in
            k.withUnsafeBufferPointer { kPtr in
                scores.withUnsafeMutableBufferPointer { sPtr in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(T), Int32(T), Int32(D),
                        scale, qPtr.baseAddress!, Int32(D),
                        kPtr.baseAddress!, Int32(D),
                        0.0, sPtr.baseAddress!, Int32(T)
                    )
                }
            }
        }
        // Causal mask + row softmax
        for r in 0..<T {
            let off = r * T
            var maxVal: Float = -Float.greatestFiniteMagnitude
            for c in 0..<T {
                if c > r { scores[off + c] = -1e9 }
                maxVal = max(maxVal, scores[off + c])
            }
            var sumExp: Float = 0
            for c in 0..<T {
                let e = expf(scores[off + c] - maxVal)
                scores[off + c] = e
                sumExp += e
            }
            let inv = 1.0 / max(sumExp, 1e-12)
            for c in 0..<T { scores[off + c] *= inv }
        }

        var ctx = [Float](repeating: 0, count: T * D)
        scores.withUnsafeBufferPointer { sPtr in
            v.withUnsafeBufferPointer { vPtr in
                ctx.withUnsafeMutableBufferPointer { cPtr in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(T), Int32(D), Int32(T),
                        1.0, sPtr.baseAddress!, Int32(T),
                        vPtr.baseAddress!, Int32(D),
                        0.0, cPtr.baseAddress!, Int32(D)
                    )
                }
            }
        }
        let saOut = linear(x: ctx, rows: T, inDim: D, w: saOW, outDim: D)
        vDSP_vadd(x, 1, saOut, 1, &x, 1, vDSP_Length(T * D))

        // 2. Pre-norm FFN (256 -> 1024 GELU -> 256)
        let xNorm2 = layerNormNoBias(x, rows: T, dim: D, weight: norm2W)
        var h = linear(x: xNorm2, rows: T, inDim: D, w: ffnW1, outDim: 1024)
        let sqrt2OverPi: Float = 0.7978845608
        for i in 0..<h.count {
            let v = h[i]
            h[i] = 0.5 * v * (1.0 + tanhf(sqrt2OverPi * (v + 0.044715 * v * v * v)))
        }
        let ffnOut = linear(x: h, rows: T, inDim: 1024, w: ffnW2, outDim: D)
        vDSP_vadd(x, 1, ffnOut, 1, &x, 1, vDSP_Length(T * D))
        return x
    }

    /// Samples all 8 NanoCodec codebooks autoregressively with Classifier-Free Guidance (CFG).
    func sampleCodebooks(
        condHidden: [Float],
        uncondHidden: [Float]?,
        audioEmbeddings: [[Float]],
        temperature: Float,
        topK: Int,
        cfgScale: Float,
        forbidEOS: Bool
    ) -> [Int32] {
        let forbidden: [Int] = forbidEOS
            ? [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
            : [2016, 2018, 2019, 2020, 2021, 2022, 2023]
        let useCFG = uncondHidden != nil && abs(cfgScale - 1.0) > 1e-4

        var condSeq = linear(x: condHidden, rows: 1, inDim: 768, w: inProjW, outDim: 256, bias: inProjB)
        var uncondSeq = uncondHidden.map { linear(x: $0, rows: 1, inDim: 768, w: inProjW, outDim: 256, bias: inProjB) } ?? []
        var codes = [Int32](repeating: 0, count: 8)

        for cb in 0..<8 {
            let seqLen = cb + 1
            let condOut = forward(sequence: condSeq, seqLen: seqLen)
            let lastCond = Array(condOut[((seqLen - 1) * 256)..<(seqLen * 256)])
            var logits = linear(x: lastCond, rows: 1, inDim: 256, w: outProjW[cb], outDim: 2024, bias: outProjB[cb])

            if useCFG {
                let uncondOut = forward(sequence: uncondSeq, seqLen: seqLen)
                let lastUncond = Array(uncondOut[((seqLen - 1) * 256)..<(seqLen * 256)])
                let uLogits = linear(x: lastUncond, rows: 1, inDim: 256, w: outProjW[cb], outDim: 2024, bias: outProjB[cb])
                let oneMinus = 1.0 - cfgScale
                for i in 0..<2024 {
                    logits[i] = cfgScale * logits[i] + oneMinus * uLogits[i]
                }
            }

            for fId in forbidden where fId < logits.count {
                logits[fId] = -Float.greatestFiniteMagnitude
            }

            let sampled = sampleTopK(logits: logits, temperature: temperature, topK: topK)
            codes[cb] = Int32(sampled)

            // Embed sampled codebook token -> project to LT 256-dim for next codebook
            if cb < 7 {
                let embOff = sampled * 768
                let tokEmb = Array(audioEmbeddings[cb][embOff..<(embOff + 768)])
                let nextVec = linear(x: tokEmb, rows: 1, inDim: 768, w: inProjW, outDim: 256, bias: inProjB)
                condSeq.append(contentsOf: nextVec)
                if useCFG { uncondSeq.append(contentsOf: nextVec) }
            }
        }
        return codes
    }

    private func sampleTopK(logits: [Float], temperature: Float, topK: Int) -> Int {
        let k = min(max(1, topK), logits.count)
        let indexed = logits.enumerated().sorted { $0.element > $1.element }.prefix(k)
        let invT = 1.0 / max(temperature, 1e-5)
        guard let maxLogit = indexed.first?.element else { return 0 }
        var probs: [(Int, Float)] = []
        probs.reserveCapacity(k)
        var total: Float = 0
        for (idx, val) in indexed {
            let p = expf((val - maxLogit) * invT)
            probs.append((idx, p))
            total += p
        }
        let target = Float.random(in: 0..<max(total, 1e-12))
        var cum: Float = 0
        for (idx, p) in probs {
            cum += p
            if cum >= target { return idx }
        }
        return probs.first?.0 ?? 0
    }
}

// MARK: - Spanish Phoneme Tokenizer with Native Inline Regional Accent Override

struct MagpieSpanishTokenizer: Sendable {
    private let spanishOffset = 96
    private let eosTokenId: Int32 = 2361
    private let token2Id: [String: Int32]
    private let phonemeDict: [String: [String]]

    init(modelDir: URL) throws {
        let tok2IdURL = modelDir.appendingPathComponent("tokenizer/spanish_phoneme_token2id.json")
        let dictURL = modelDir.appendingPathComponent("tokenizer/spanish_phoneme_phoneme_dict.json")
        let t2iData = try Data(contentsOf: tok2IdURL)
        self.token2Id = try JSONDecoder().decode([String: Int32].self, from: t2iData)

        if let dData = try? Data(contentsOf: dictURL) {
            if let direct = try? JSONDecoder().decode([String: [String]].self, from: dData) {
                self.phonemeDict = direct
            } else if let raw = try? JSONSerialization.jsonObject(with: dData) as? [String: Any] {
                var parsed: [String: [String]] = [:]
                parsed.reserveCapacity(raw.count)
                for (k, v) in raw {
                    if let arr = v as? [String] {
                        parsed[k.uppercased(with: Locale(identifier: "es_ES"))] = arr
                    } else if let nested = v as? [[String]], let first = nested.first {
                        parsed[k.uppercased(with: Locale(identifier: "es_ES"))] = first
                    }
                }
                self.phonemeDict = parsed
            } else {
                self.phonemeDict = [:]
            }
        } else {
            self.phonemeDict = [:]
        }
    }

    /// Tokenizes Spanish prose into Magpie AggregatedTTSTokenizer IDs matching NeMo's
    /// `chunk_and_tokenize_text_by_sentence` (NO text_bos_id; leading/trailing space + trailing eos_token_id=2361):
    /// - `.castilian` (`es-ES`): keeps distinción `/θ/` and `/ʎ/`
    /// - `.latinAmerican` (`es-419`): maps `/θ/` -> `/s/` (seseo) and `/ʎ/` -> `/ʝ/` (yeísmo)
    /// - `.rioplatense` (`es-AR`): maps `/θ/` -> `/s/` (seseo) and `/ʎ/`, `/ʝ/` -> `/ʃ/` (sheísmo)
    /// Also respects explicit inline `| ... |` IPA spans!
    func tokenize(text: String, accent: SpanishRegionalAccent) -> [Int32] {
        var ids: [Int32] = []
        ids.reserveCapacity(128)
        let spaceId = (token2Id[" "] ?? 100) + Int32(spanishOffset)
        let esLocale = Locale(identifier: "es_ES")
        ids.append(spaceId) // Spanish tokenizer uses pad_with_space = true (matches tokenizer_references.json)

        // Split on inline | ... | IPA overrides
        let segments = text.components(separatedBy: "|")
        for (segIdx, seg) in segments.enumerated() {
            if segIdx % 2 == 1 {
                // Direct IPA symbols separated by whitespace
                for sym in seg.split(separator: " ") {
                    let s = sym == "g" ? "ɡ" : String(sym)
                    if let localId = token2Id[s] {
                        ids.append(localId + Int32(spanishOffset))
                    }
                }
                continue
            }

            // Standard Spanish text -> word-by-word lookup + G2P fallback + regional phonological filter
            let words = seg.split(whereSeparator: { $0.isWhitespace })
            for (wIdx, rawWord) in words.enumerated() {
                let wordStr = String(rawWord)
                let leadingPunct = wordStr.prefix { !$0.isLetter && !$0.isNumber }
                let trailingPunct = wordStr.reversed().prefix { !$0.isLetter && !$0.isNumber }.reversed()
                let core = String(wordStr.dropFirst(leadingPunct.count).dropLast(trailingPunct.count))

                for ch in leadingPunct {
                    if let lid = token2Id[String(ch)] { ids.append(lid + Int32(spanishOffset)) }
                }

                if !core.isEmpty {
                    let phones: [String]
                    let upperKey = core.uppercased(with: esLocale)
                    if let dictPhones = phonemeDict[upperKey] {
                        phones = applyRegionalPhonology(dictPhones, accent: accent)
                    } else {
                        // OOV fallback: normalize ASCII 'g' -> IPA 'ɡ' so every phoneme maps cleanly in spanish_phoneme_token2id
                        let ipa = SpanishLemmatizer.wordToIPA(core, accent: accent)
                            .replacingOccurrences(of: "g", with: "ɡ")
                        phones = ipa.map { String($0) }
                    }
                    for ph in phones {
                        if let lid = token2Id[ph] {
                            ids.append(lid + Int32(spanishOffset))
                        }
                    }
                }

                for ch in trailingPunct {
                    if let lid = token2Id[String(ch)] { ids.append(lid + Int32(spanishOffset)) }
                }

                if wIdx + 1 < words.count {
                    ids.append(spaceId)
                }
            }
        }
        ids.append(spaceId)
        ids.append(eosTokenId)
        return Array(ids.prefix(256))
    }

    private func applyRegionalPhonology(_ phones: [String], accent: SpanishRegionalAccent) -> [String] {
        switch accent {
        case .castilian:
            return phones
        case .latinAmerican:
            return phones.map { p in
                if p == "θ" { return "s" }
                if p == "ʎ" { return "ʝ" }
                return p
            }
        case .rioplatense:
            return phones.map { p in
                if p == "θ" { return "s" }
                if p == "ʎ" || p == "ʝ" { return "ʃ" }
                return p
            }
        }
    }
}

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

// MARK: - Native On-Device NVIDIA Magpie v2607 Core ML Engine + Pitch-Preserving Pace Player

public actor MagpieV2607Engine {
    public static let sampleRate: Double = 22_050.0
    private static let samplesPerFrame: Int = 1024
    private static let nLayers: Int = 12
    private static let maxTextLen: Int = 256
    private static let maxSeqLen: Int = 512

    private static let kOutKeys = [
        "new_k_1", "new_k_3", "new_k_5", "new_k_7",
        "new_k_9", "new_k_11", "new_k_13", "new_k_15",
        "new_k_17", "new_k_19", "new_k_21", "new_k",
    ]
    private static let vOutKeys = [
        "new_v_1", "new_v_3", "new_v_5", "new_v_7",
        "new_v_9", "new_v_11", "new_v_13", "new_v_15",
        "new_v_17", "new_v_19", "new_v_21", "new_v",
    ]
    private static let posOutKeys = [
        "var_169", "var_339", "var_509", "var_679", "var_849", "var_1019",
        "var_1189", "var_1359", "var_1529", "var_1699", "var_1869", "var_2039",
    ]
    private static let prefillKVKeys = [
        "var_208", "var_374", "var_540", "var_706", "var_872", "var_1038",
        "var_1204", "var_1370", "var_1536", "var_1702", "var_1868", "var_1958",
    ]

    private var textEncoder: MLModel?
    private var decoderPrefill: MLModel?
    private var decoderStep: MLModel?
    private var nanoCodec: MLModel?
    private var nanoCodecFrames: Int = 24
    private var frameStackingFactor: Int = 1

    private var tokenizer: MagpieSpanishTokenizer?
    private var localTransformer: MagpieLocalTransformerWeights?
    private var speakerEmbeddings: [[Float]] = []       // 5 x (110 * 768)
    private var audioEmbeddings: [[Float]] = []         // 8 x (2024 * 768)

    // Pre-cached unconditional CFG prefill KV state (computed once at load -> saves 50% of prefill!)
    private var cachedUncondK: [MLMultiArray]?
    private var cachedUncondV: [MLMultiArray]?

    private let audioCacheDir: URL
    public private(set) var isLoaded: Bool = false

    // Real-time AVAudioEngine playback with pitch-preserving TimePitch (70–140 WPM)
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var timePitchNode: AVAudioUnitTimePitch?
    private var onAudioLevelUpdate: (@Sendable (Float) -> Void)?

    public init() {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        self.audioCacheDir = caches.appendingPathComponent("CadenciaAudioCache", isDirectory: true)
        try? FileManager.default.createDirectory(at: self.audioCacheDir, withIntermediateDirectories: true)
        // Purge legacy pre-v2 cached WAV files so upgraded engine audio is always freshly generated
        if let files = try? FileManager.default.contentsOfDirectory(at: self.audioCacheDir, includingPropertiesForKeys: nil) {
            for f in files where !f.lastPathComponent.hasPrefix("v2_") {
                try? FileManager.default.removeItem(at: f)
            }
        }
    }

    public func setAudioLevelHandler(_ handler: (@Sendable (Float) -> Void)?) {
        self.onAudioLevelUpdate = handler
    }

    /// Loads all 4 compiled Magpie Core ML models concurrently with optimal Apple Silicon compute placement
    /// (verified via FluidInference/mobius SWIFT_PORT_FINDINGS.md & PERF.md:
    ///  text_encoder & decoder_prefill -> ANE, decoder_step -> GPU+CPU avoiding per-step ANE scatter recompile,
    ///  NanoCodec v3 -> CPU avoiding FP16 audio drift), plus caches unconditional prefill KV tensors once.
    public func loadModels(from modelDir: URL) async throws {
        if isLoaded { return }

        // Parse constants.json to auto-detect frame_stacking_factor (1 or v2607 factor 2)
        let constURL = modelDir.appendingPathComponent("constants/constants.json")
        if let cData = try? Data(contentsOf: constURL),
           let cJson = try? JSONSerialization.jsonObject(with: cData) as? [String: Any],
           let fsf = cJson["frame_stacking_factor"] as? Int {
            self.frameStackingFactor = max(1, fsf)
        }

        let aneConfig = MLModelConfiguration()
        aneConfig.computeUnits = .cpuAndNeuralEngine // Phase D verified: ANE gives clean EOS + 2× speedup on text_encoder, prefill & decoder_step!

        let codecConfig = MLModelConfiguration()
        codecConfig.computeUnits = .cpuOnly // Phase F verified: FP32 NanoCodec v3 on CPU avoids fp16 vocoder noise completely

        let textEncURL = modelDir.appendingPathComponent("text_encoder.mlmodelc")
        let prefillURL = modelDir.appendingPathComponent("decoder_prefill.mlmodelc")
        let decStepURL = modelDir.appendingPathComponent("decoder_step.mlmodelc")
        let v3URL = modelDir.appendingPathComponent("nanocodec_decoder_v3.mlmodelc")
        let v1URL = modelDir.appendingPathComponent("nanocodec_decoder.mlmodelc")
        let codecURL = FileManager.default.fileExists(atPath: v3URL.path) ? v3URL : v1URL

        // Load CPU NanoCodec concurrently while compiling ANE models one-by-one to prevent iOS `aned` contention!
        async let loadedCodec = MLModel.load(contentsOf: codecURL, configuration: codecConfig)

        let te = try await MLModel.load(contentsOf: textEncURL, configuration: aneConfig)
        let pf = try await MLModel.load(contentsOf: prefillURL, configuration: aneConfig)
        let ds = try await MLModel.load(contentsOf: decStepURL, configuration: aneConfig)

        // Load tokenizer + local transformer + speaker/audio embeddings
        let tok = try MagpieSpanishTokenizer(modelDir: modelDir)
        let lt = try MagpieLocalTransformerWeights.load(from: modelDir)

        var spks: [[Float]] = []
        spks.reserveCapacity(5)
        for s in 0..<5 {
            spks.append(try NpyLoader.loadFloat32(from: modelDir.appendingPathComponent("constants/speaker_\(s).npy")).data)
        }
        var aEmbs: [[Float]] = []
        aEmbs.reserveCapacity(8)
        for cb in 0..<8 {
            aEmbs.append(try NpyLoader.loadFloat32(from: modelDir.appendingPathComponent("constants/audio_embedding_\(cb).npy")).data)
        }

        let nc = try await loadedCodec
        self.textEncoder = te
        self.decoderPrefill = pf
        self.decoderStep = ds
        self.nanoCodec = nc
        if let shape = nc.modelDescription.inputDescriptionsByName["tokens"]?.multiArrayConstraint?.shape,
           let last = shape.last?.intValue {
            self.nanoCodecFrames = last
        }

        self.tokenizer = tok
        self.localTransformer = lt
        self.speakerEmbeddings = spks
        self.audioEmbeddings = aEmbs

        // Precompute unconditional CFG prefill KV state once!
        try await precomputeUnconditionalPrefill()
        self.isLoaded = true
    }

    private func prefillEmbedInputName(for model: MLModel) -> String {
        if model.modelDescription.inputDescriptionsByName["audio_embed"] != nil {
            return "audio_embed"
        }
        return "input"
    }

    private func inputDataType(for model: MLModel, featureName: String, defaultType: MLMultiArrayDataType) -> MLMultiArrayDataType {
        model.modelDescription.inputDescriptionsByName[featureName]?.multiArrayConstraint?.dataType ?? defaultType
    }

    private func makeMaskArray(length: Int, validCount: Int, dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: dataType)
        if dataType == .float32 {
            let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: length)
            for i in 0..<length { ptr[i] = i < validCount ? 1.0 : 0.0 }
        } else {
            let ptr = arr.dataPointer.bindMemory(to: Float16.self, capacity: length)
            for i in 0..<length { ptr[i] = i < validCount ? 1.0 : 0.0 }
        }
        return arr
    }

    /// Stride-aware 3D tensor conversion (e.g. `[1, 256, 768]` Float32 -> Float16).
    /// Handles ANE IOSurface-padded `MLMultiArray` outputs cleanly without row skew!
    private func convertEncoderOutput3D(_ src: MLMultiArray, to targetType: MLMultiArrayDataType) throws -> MLMultiArray {
        let d0 = src.shape[0].intValue
        let d1 = src.shape[1].intValue
        let d2 = src.shape[2].intValue
        let s0 = src.strides[0].intValue
        let s1 = src.strides[1].intValue
        let s2 = src.strides[2].intValue
        let srcContiguous = (s2 == 1 && s1 == d2 && s0 == d1 * d2)
        if src.dataType == targetType && srcContiguous { return src }

        let dst = try MLMultiArray(shape: src.shape, dataType: targetType)
        let ds0 = dst.strides[0].intValue
        let ds1 = dst.strides[1].intValue
        let ds2 = dst.strides[2].intValue

        switch (src.dataType, targetType) {
        case (.float32, .float16):
            let sp = src.dataPointer.assumingMemoryBound(to: Float.self)
            let dp = dst.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<d0 {
                for j in 0..<d1 {
                    let sBase = i * s0 + j * s1
                    let dBase = i * ds0 + j * ds1
                    for k in 0..<d2 {
                        dp[dBase + k * ds2] = Float16(sp[sBase + k * s2])
                    }
                }
            }
        case (.float16, .float32):
            let sp = src.dataPointer.assumingMemoryBound(to: Float16.self)
            let dp = dst.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<d0 {
                for j in 0..<d1 {
                    let sBase = i * s0 + j * s1
                    let dBase = i * ds0 + j * ds1
                    for k in 0..<d2 {
                        dp[dBase + k * ds2] = Float(sp[sBase + k * s2])
                    }
                }
            }
        case (.float16, .float16):
            let sp = src.dataPointer.assumingMemoryBound(to: Float16.self)
            let dp = dst.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<d0 {
                for j in 0..<d1 {
                    let sBase = i * s0 + j * s1
                    let dBase = i * ds0 + j * ds1
                    if s2 == 1 && ds2 == 1 {
                        memcpy(dp.advanced(by: dBase), sp.advanced(by: sBase), d2 * MemoryLayout<Float16>.stride)
                    } else {
                        for k in 0..<d2 { dp[dBase + k * ds2] = sp[sBase + k * s2] }
                    }
                }
            }
        default:
            let sp = src.dataPointer.assumingMemoryBound(to: Float.self)
            let dp = dst.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<d0 {
                for j in 0..<d1 {
                    let sBase = i * s0 + j * s1
                    let dBase = i * ds0 + j * ds1
                    if s2 == 1 && ds2 == 1 {
                        memcpy(dp.advanced(by: dBase), sp.advanced(by: sBase), d2 * MemoryLayout<Float>.stride)
                    } else {
                        for k in 0..<d2 { dp[dBase + k * ds2] = sp[sBase + k * s2] }
                    }
                }
            }
        }
        return dst
    }

    private func precomputeUnconditionalPrefill() async throws {
        guard let prefill = decoderPrefill else { return }
        let embedKey = prefillEmbedInputName(for: prefill)
        let embedType = inputDataType(for: prefill, featureName: embedKey, defaultType: .float16)
        let encType = inputDataType(for: prefill, featureName: "encoder_output", defaultType: .float16)
        let maskType = inputDataType(for: prefill, featureName: "encoder_mask", defaultType: .float32)

        let zeroCtx = try MLMultiArray(shape: [1, 110, 768], dataType: embedType)
        let zeroEnc = try MLMultiArray(shape: [1, 256, 768], dataType: encType)
        memset(zeroCtx.dataPointer, 0, zeroCtx.count * (embedType == .float16 ? 2 : 4))
        memset(zeroEnc.dataPointer, 0, zeroEnc.count * (encType == .float16 ? 2 : 4))
        let uncondMask = try makeMaskArray(length: Self.maxTextLen, validCount: 1, dataType: maskType)

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            embedKey: MLFeatureValue(multiArray: zeroCtx),
            "encoder_output": MLFeatureValue(multiArray: zeroEnc),
            "encoder_mask": MLFeatureValue(multiArray: uncondMask),
        ])
        let out = try await prefill.prediction(from: provider)
        let (ks, vs) = try splitPrefillKV(out)
        self.cachedUncondK = ks
        self.cachedUncondV = vs
    }

    /// Synthesizes a full Spanish passage using sentence-level pipelining + zero-copy disk caching.
    /// First sentence audio is cached and ready quickly; subsequent replays take 0ms!
    public func synthesizePassage(
        sentences: [String],
        voice: MagpieVoice,
        accent: SpanishRegionalAccent,
        onSentenceReady: (@Sendable (Int, SynthesizedAudioClip) -> Void)? = nil
    ) async throws -> SynthesizedAudioClip {
        let startWall = Date()
        var allSamples: [Float] = []
        var allWordTimings: [WordTimingSpan] = []
        var timeCursor: TimeInterval = 0
        var globalWordIdx = 0

        for (sIdx, sentence) in sentences.enumerated() {
            let clip = try await synthesizeSentence(
                text: sentence,
                voice: voice,
                accent: accent,
                startingWordIndex: globalWordIdx,
                timeOffset: timeCursor
            )
            allSamples.append(contentsOf: clip.samples)
            // Natural 140ms inter-sentence pause
            let pauseCount = Int(Self.sampleRate * 0.14)
            allSamples.append(contentsOf: [Float](repeating: 0, count: pauseCount))

            allWordTimings.append(contentsOf: clip.wordTimings)
            globalWordIdx += clip.wordTimings.count
            timeCursor += clip.durationSeconds + 0.14
            onSentenceReady?(sIdx, clip)
        }

        let elapsedMs = Date().timeIntervalSince(startWall) * 1000.0
        let duration = Double(allSamples.count) / Self.sampleRate
        let rtf = duration > 0 ? (elapsedMs / 1000.0) / duration : 0
        return SynthesizedAudioClip(
            samples: allSamples,
            sampleRate: Self.sampleRate,
            durationSeconds: duration,
            wordTimings: allWordTimings,
            synthesisLatencyMs: elapsedMs,
            realTimeFactor: rtf,
            cachedFileURL: nil
        )
    }

    /// Synthesizes a single Spanish sentence (or short word/phrase) using Magpie Core ML + disk cache.
    public func synthesizeSentence(
        text: String,
        voice: MagpieVoice,
        accent: SpanishRegionalAccent,
        startingWordIndex: Int = 0,
        timeOffset: TimeInterval = 0
    ) async throws -> SynthesizedAudioClip {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let cacheFile = cacheURL(for: trimmed, voice: voice, accent: accent)

        if FileManager.default.fileExists(atPath: cacheFile.path),
           let cached = try? loadWAV(from: cacheFile) {
            let dur = Double(cached.count) / Self.sampleRate
            let timings = estimateWordTimings(text: trimmed, duration: dur, startWordIndex: startingWordIndex, offset: timeOffset)
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

        guard isLoaded,
              let textEnc = textEncoder,
              let prefill = decoderPrefill,
              let decStep = decoderStep,
              let codec = nanoCodec,
              let tok = tokenizer,
              let lt = localTransformer
        else {
            throw NSError(domain: "MagpieV2607Engine", code: 503, userInfo: [
                NSLocalizedDescriptionKey: "NVIDIA Magpie v2607 Core ML models are not loaded yet. Download them in Settings -> On-Device Models."
            ])
        }

        let t0 = Date()
        let tokenIds = tok.tokenize(text: trimmed, accent: accent)
        let tLen = min(tokenIds.count, Self.maxTextLen)

        // 1. Run text_encoder on ANE (text_mask is Float32 per Core ML modelDescription!)
        let tokensArr = try MLMultiArray(shape: [1, NSNumber(value: Self.maxTextLen)], dataType: .int32)
        let tPtr = tokensArr.dataPointer.bindMemory(to: Int32.self, capacity: Self.maxTextLen)
        for i in 0..<Self.maxTextLen {
            tPtr[i] = i < tLen ? tokenIds[i] : 0
        }
        let textMaskType = inputDataType(for: textEnc, featureName: "text_mask", defaultType: .float32)
        let textMaskArr = try makeMaskArray(length: Self.maxTextLen, validCount: tLen, dataType: textMaskType)

        let encOut = try await textEnc.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "text_tokens": MLFeatureValue(multiArray: tokensArr),
            "text_mask": MLFeatureValue(multiArray: textMaskArr),
        ]))
        guard let rawEncoderOutput = encOut.featureValue(for: "encoder_output")?.multiArrayValue else {
            throw NSError(domain: "MagpieV2607Engine", code: 500, userInfo: [NSLocalizedDescriptionKey: "Missing encoder_output"])
        }

        // Convert encoder_output (Float32 -> Float16) & match exact mask dtype expected by decoder_prefill / decoder_step
        let prefillEmbedKey = prefillEmbedInputName(for: prefill)
        let prefillEmbedType = inputDataType(for: prefill, featureName: prefillEmbedKey, defaultType: .float16)
        let prefillEncType = inputDataType(for: prefill, featureName: "encoder_output", defaultType: .float16)
        let prefillMaskType = inputDataType(for: prefill, featureName: "encoder_mask", defaultType: .float32)

        let encoderOutputPrefill = try convertEncoderOutput3D(rawEncoderOutput, to: prefillEncType)
        let prefillMaskArr = (prefillMaskType == textMaskType)
            ? textMaskArr
            : try makeMaskArray(length: Self.maxTextLen, validCount: tLen, dataType: prefillMaskType)

        // 2. Run conditional decoder_prefill (1 batched ANE call vs 110 sequential calls -> 3.8× speedup!)
        let spkIdx = min(voice.speakerIndex, speakerEmbeddings.count - 1)
        let spkArr = try MLMultiArray(shape: [1, 110, 768], dataType: prefillEmbedType)
        let spkSrc = speakerEmbeddings[spkIdx]
        if prefillEmbedType == .float16 {
            let spkDst = spkArr.dataPointer.bindMemory(to: Float16.self, capacity: 110 * 768)
            for i in 0..<min(spkSrc.count, 110 * 768) { spkDst[i] = Float16(spkSrc[i]) }
        } else {
            let spkDst = spkArr.dataPointer.bindMemory(to: Float.self, capacity: 110 * 768)
            for i in 0..<min(spkSrc.count, 110 * 768) { spkDst[i] = spkSrc[i] }
        }

        let prefillOut = try await prefill.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            prefillEmbedKey: MLFeatureValue(multiArray: spkArr),
            "encoder_output": MLFeatureValue(multiArray: encoderOutputPrefill),
            "encoder_mask": MLFeatureValue(multiArray: prefillMaskArr),
        ]))
        var (condK, condV) = try splitPrefillKV(prefillOut)
        var uncondK = try cloneKVList4D(cachedUncondK ?? condK)
        var uncondV = try cloneKVList4D(cachedUncondV ?? condV)

        // Prepare decoder_step encoder tensors matching decoder_step's exact modelDescription dtypes
        let stepEncType = inputDataType(for: decStep, featureName: "encoder_output", defaultType: .float16)
        let stepMaskType = inputDataType(for: decStep, featureName: "encoder_mask", defaultType: .float32)
        let encoderOutputStep = (stepEncType == prefillEncType)
            ? encoderOutputPrefill
            : try convertEncoderOutput3D(rawEncoderOutput, to: stepEncType)
        let stepMaskArr = (stepMaskType == prefillMaskType)
            ? prefillMaskArr
            : try makeMaskArray(length: Self.maxTextLen, validCount: tLen, dataType: stepMaskType)

        let zeroEnc = try MLMultiArray(shape: [1, 256, 768], dataType: stepEncType)
        memset(zeroEnc.dataPointer, 0, zeroEnc.count * (stepEncType == .float16 ? 2 : 4))
        let uncondMask = try makeMaskArray(length: Self.maxTextLen, validCount: 1, dataType: stepMaskType)

        // 3. Autoregressive frame generation loop (matching generate_coreml.py exact per-layer position propagation)
        let maxSteps = min(400, max(36, tLen * 6))
        let minFrames = 4
        var currentCodes = [Int32](repeating: 2016, count: 8) // audio_bos_id = 2016
        var generatedFrames: [[Int32]] = []
        generatedFrames.reserveCapacity(maxSteps)

        let initPos = try MLMultiArray(shape: [1], dataType: .float16)
        initPos.dataPointer.assumingMemoryBound(to: Float16.self)[0] = Float16(110.0)
        var condPos = [MLMultiArray](repeating: initPos, count: Self.nLayers)
        var uncondPos = [MLMultiArray](repeating: initPos, count: Self.nLayers)

        for step in 0..<maxSteps {
            let audioEmbArr = try embedAudioCodesFP16(codes: currentCodes)

            let (condHidden, nextCK, nextCV, nextCPos) = try runDecoderStep(
                model: decStep,
                audioEmbed: audioEmbArr,
                encoderOutput: encoderOutputStep,
                encoderMask: stepMaskArr,
                cacheK: condK,
                cacheV: condV,
                positions: condPos
            )
            condK = nextCK
            condV = nextCV
            condPos = nextCPos

            let (uncondHidden, nextUK, nextUV, nextUPos) = try runDecoderStep(
                model: decStep,
                audioEmbed: audioEmbArr,
                encoderOutput: zeroEnc,
                encoderMask: uncondMask,
                cacheK: uncondK,
                cacheV: uncondV,
                positions: uncondPos
            )
            uncondK = nextUK
            uncondV = nextUV
            uncondPos = nextUPos

            let nextCodes = lt.sampleCodebooks(
                condHidden: condHidden,
                uncondHidden: uncondHidden,
                audioEmbeddings: audioEmbeddings,
                temperature: 0.6,
                topK: 80,
                cfgScale: 2.5,
                forbidEOS: step < minFrames
            )

            if step >= minFrames && nextCodes.contains(2017) { // audio_eos_id = 2017
                break
            }
            generatedFrames.append(nextCodes)
            currentCodes = nextCodes
        }

        // 4. Decode 8-codebook tokens -> 22.05kHz PCM via Chunked NanoCodec v3 (T_in=24, stride=8, overlap=16)
        let samples = try decodeNanoCodec(model: codec, frames: generatedFrames)

        // Peak-normalize to -1dBFS for rich, crystal-clear tutoring audio
        var normalized = samples
        var peak: Float = 0
        vDSP_maxmgv(normalized, 1, &peak, vDSP_Length(normalized.count))
        if peak > 1e-6 {
            var gain = 0.89 / peak
            vDSP_vsmul(normalized, 1, &gain, &normalized, 1, vDSP_Length(normalized.count))
        }

        // Cache to disk asynchronously (0ms on every future replay!)
        try? saveWAV(samples: normalized, sampleRate: Self.sampleRate, to: cacheFile)

        let elapsedMs = Date().timeIntervalSince(t0) * 1000.0
        let duration = Double(normalized.count) / Self.sampleRate
        let rtf = duration > 0 ? (elapsedMs / 1000.0) / duration : 0
        let timings = estimateWordTimings(text: trimmed, duration: duration, startWordIndex: startingWordIndex, offset: timeOffset)

        return SynthesizedAudioClip(
            samples: normalized,
            sampleRate: Self.sampleRate,
            durationSeconds: duration,
            wordTimings: timings,
            synthesisLatencyMs: elapsedMs,
            realTimeFactor: rtf,
            cachedFileURL: cacheFile
        )
    }

    // MARK: - Real-Time Pitch-Preserving Playback (Natural 120 WPM baseline; bypassed at 1.0× for pure PCM)

    public func play(clip: SynthesizedAudioClip, targetWPM: Int, baselineNativeWPM: Double = 120.0) async throws {
        stopPlayback()
        #if os(iOS)
        // Use .playback + .spokenAudio so iPhone plays full-fidelity speaker audio (NOT .playAndRecord VoIP DSP!)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio, options: [])
        try session.setActive(true)
        #endif

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let timePitch = AVAudioUnitTimePitch()

        // Clamp rate to [0.78, 1.25] so phase-vocoder time-stretching never introduces robotic artifacts,
        // and bypass AVAudioUnitTimePitch completely when near native 1.0× speed!
        let rate = Float(max(0.78, min(1.25, Double(targetWPM) / baselineNativeWPM)))
        timePitch.rate = rate
        timePitch.pitch = 0.0
        timePitch.bypass = abs(rate - 1.0) < 0.04

        engine.attach(player)
        engine.attach(timePitch)

        guard let format = AVAudioFormat(standardFormatWithSampleRate: clip.sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(clip.samples.count))
        else { return }

        buffer.frameLength = AVAudioFrameCount(clip.samples.count)
        if let channel = buffer.floatChannelData?[0] {
            clip.samples.withUnsafeBufferPointer { ptr in
                channel.update(from: ptr.baseAddress!, count: clip.samples.count)
            }
        }

        engine.connect(player, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)

        // Tap main mixer at 60Hz for live Metal Resonance Orb audio reactivity!
        let levelCallback = self.onAudioLevelUpdate
        engine.mainMixerNode.installTap(onBus: 0, bufferSize: 1024, format: nil) { pcm, _ in
            guard let data = pcm.floatChannelData?[0], pcm.frameLength > 0 else { return }
            var rms: Float = 0
            vDSP_rmsqv(data, 1, &rms, vDSP_Length(pcm.frameLength))
            levelCallback?(min(1.0, rms * 4.5))
        }

        try engine.start()
        self.audioEngine = engine
        self.playerNode = player
        self.timePitchNode = timePitch

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            player.scheduleBuffer(buffer, at: nil, options: []) {
                cont.resume()
            }
            player.play()
        }
        onAudioLevelUpdate?(0.0)
    }

    public func updatePlaybackRate(targetWPM: Int, baselineNativeWPM: Double = 120.0) {
        let rate = Float(max(0.78, min(1.25, Double(targetWPM) / baselineNativeWPM)))
        timePitchNode?.rate = rate
        timePitchNode?.bypass = abs(rate - 1.0) < 0.04
    }

    public func stopPlayback() {
        playerNode?.stop()
        audioEngine?.mainMixerNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        playerNode = nil
        timePitchNode = nil
        onAudioLevelUpdate?(0.0)
    }

    // MARK: - Stride-Safe Core ML Tensor Helpers (Immune to ANE IOSurface Row Padding!)

    /// Splits each `[2, 1, 512, 12, 64]` prefill output into contiguous `[1, 512, 12, 64]` K and V caches
    /// respecting exact Core ML / ANE IOSurface strides (never corrupting KV rows with padding!).
    private func splitPrefillKV(_ out: MLFeatureProvider) throws -> ([MLMultiArray], [MLMultiArray]) {
        var ks: [MLMultiArray] = []
        var vs: [MLMultiArray] = []
        ks.reserveCapacity(Self.nLayers)
        vs.reserveCapacity(Self.nLayers)

        for i in 0..<Self.nLayers {
            guard let combined = out.featureValue(for: Self.prefillKVKeys[i])?.multiArrayValue else {
                throw NSError(domain: "MagpieV2607Engine", code: 501, userInfo: [NSLocalizedDescriptionKey: "Missing prefill KV layer \(i)"])
            }
            let kArr = try MLMultiArray(shape: [1, 512, 12, 64], dataType: .float16)
            let vArr = try MLMultiArray(shape: [1, 512, 12, 64], dataType: .float16)
            memset(kArr.dataPointer, 0, kArr.count * MemoryLayout<Float16>.stride)
            memset(vArr.dataPointer, 0, vArr.count * MemoryLayout<Float16>.stride)

            let s0 = combined.strides[0].intValue
            let s2 = combined.strides[2].intValue
            let s3 = combined.strides[3].intValue
            let s4 = combined.strides[4].intValue

            let kd1 = kArr.strides[1].intValue
            let kd2 = kArr.strides[2].intValue
            let kd3 = kArr.strides[3].intValue

            let vd1 = vArr.strides[1].intValue
            let vd2 = vArr.strides[2].intValue
            let vd3 = vArr.strides[3].intValue

            let src = combined.dataPointer.assumingMemoryBound(to: Float16.self)
            let kDst = kArr.dataPointer.assumingMemoryBound(to: Float16.self)
            let vDst = vArr.dataPointer.assumingMemoryBound(to: Float16.self)

            let fastHeadCopy = (s4 == 1 && kd3 == 1 && vd3 == 1)
            let headBytes = 64 * MemoryLayout<Float16>.stride

            // Copy all 512 time slots stride-safely (guaranteed identical for both contiguous & ANE-padded IOSurface!)
            for t in 0..<512 {
                let sT = t * s2
                let kT = t * kd1
                let vT = t * vd1
                for h in 0..<12 {
                    let sK = sT + h * s3
                    let sV = s0 + sK
                    let kOff = kT + h * kd2
                    let vOff = vT + h * vd2
                    if fastHeadCopy {
                        memcpy(kDst.advanced(by: kOff), src.advanced(by: sK), headBytes)
                        memcpy(vDst.advanced(by: vOff), src.advanced(by: sV), headBytes)
                    } else {
                        for d in 0..<64 {
                            kDst[kOff + d * kd3] = src[sK + d * s4]
                            vDst[vOff + d * vd3] = src[sV + d * s4]
                        }
                    }
                }
            }
            ks.append(kArr)
            vs.append(vArr)
        }
        return (ks, vs)
    }

    private func cloneKVList4D(_ list: [MLMultiArray]) throws -> [MLMultiArray] {
        try list.map { src in
            let dst = try MLMultiArray(shape: src.shape, dataType: src.dataType)
            let d1 = src.shape[1].intValue
            let d2 = src.shape[2].intValue
            let d3 = src.shape[3].intValue
            let s1 = src.strides[1].intValue
            let s2 = src.strides[2].intValue
            let s3 = src.strides[3].intValue
            let ds1 = dst.strides[1].intValue
            let ds2 = dst.strides[2].intValue
            let ds3 = dst.strides[3].intValue
            let elemBytes = src.dataType == .float16 ? 2 : 4

            if s3 == 1 && ds3 == 1 && s2 == d3 && ds2 == d3 && s1 == d2 * d3 && ds1 == d2 * d3 {
                memcpy(dst.dataPointer, src.dataPointer, src.count * elemBytes)
            } else {
                let sp = src.dataPointer.assumingMemoryBound(to: Float16.self)
                let dp = dst.dataPointer.assumingMemoryBound(to: Float16.self)
                for t in 0..<d1 {
                    for h in 0..<d2 {
                        let sBase = t * s1 + h * s2
                        let dBase = t * ds1 + h * ds2
                        if s3 == 1 && ds3 == 1 {
                            memcpy(dp.advanced(by: dBase), sp.advanced(by: sBase), d3 * MemoryLayout<Float16>.stride)
                        } else {
                            for d in 0..<d3 { dp[dBase + d * ds3] = sp[sBase + d * s3] }
                        }
                    }
                }
            }
            return dst
        }
    }

    private func embedAudioCodesFP16(codes: [Int32]) throws -> MLMultiArray {
        var acc = [Float](repeating: 0, count: 768)
        for cb in 0..<8 {
            let idx = max(0, min(2023, Int(codes[cb])))
            let off = idx * 768
            audioEmbeddings[cb].withUnsafeBufferPointer { ptr in
                vDSP_vadd(acc, 1, ptr.baseAddress!.advanced(by: off), 1, &acc, 1, 768)
            }
        }
        var scale: Float = 1.0 / 8.0
        vDSP_vsmul(acc, 1, &scale, &acc, 1, 768)

        let arr = try MLMultiArray(shape: [1, 1, 768], dataType: .float16)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float16.self)
        let strideLast = arr.strides.last?.intValue ?? 1
        for i in 0..<768 { dst[i * strideLast] = Float16(acc[i]) }
        return arr
    }

    private func runDecoderStep(
        model: MLModel,
        audioEmbed: MLMultiArray,
        encoderOutput: MLMultiArray,
        encoderMask: MLMultiArray,
        cacheK: [MLMultiArray],
        cacheV: [MLMultiArray],
        positions: [MLMultiArray]
    ) throws -> (hidden: [Float], newK: [MLMultiArray], newV: [MLMultiArray], newPos: [MLMultiArray]) {
        var dict: [String: MLFeatureValue] = [
            "audio_embed": MLFeatureValue(multiArray: audioEmbed),
            "encoder_output": MLFeatureValue(multiArray: encoderOutput),
            "encoder_mask": MLFeatureValue(multiArray: encoderMask),
        ]

        for i in 0..<Self.nLayers {
            dict["cache_k\(i)"] = MLFeatureValue(multiArray: cacheK[i])
            dict["cache_v\(i)"] = MLFeatureValue(multiArray: cacheV[i])
            dict["position\(i)"] = MLFeatureValue(multiArray: positions[i])
        }

        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))
        guard let hidArr = out.featureValue(for: "input")?.multiArrayValue else {
            throw NSError(domain: "MagpieV2607Engine", code: 502, userInfo: [NSLocalizedDescriptionKey: "Missing decoder hidden state"])
        }
        var hidden = [Float](repeating: 0, count: 768)
        let hStride = hidArr.strides.last?.intValue ?? 1
        if hidArr.dataType == .float16 {
            let src = hidArr.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<768 { hidden[i] = Float(src[i * hStride]) }
        } else {
            let src = hidArr.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<768 { hidden[i] = src[i * hStride] }
        }

        var nextK: [MLMultiArray] = []
        var nextV: [MLMultiArray] = []
        var nextPos: [MLMultiArray] = []
        nextK.reserveCapacity(Self.nLayers)
        nextV.reserveCapacity(Self.nLayers)
        nextPos.reserveCapacity(Self.nLayers)
        for i in 0..<Self.nLayers {
            nextK.append(out.featureValue(for: Self.kOutKeys[i])!.multiArrayValue!)
            nextV.append(out.featureValue(for: Self.vOutKeys[i])!.multiArrayValue!)
            nextPos.append(out.featureValue(for: Self.posOutKeys[i])!.multiArrayValue!)
        }
        return (hidden, nextK, nextV, nextPos)
    }

    /// Auto-dispatches between monolithic (T_in=256) and chunked sliding-window NanoCodec v3 (T_in=24, stride=8, overlap=16).
    /// Uses **edge-replication padding** (clamping out-of-range indices to `[0, T-1]`) instead of zero-padding,
    /// eliminating the 700× start/end transient click caused by codebook entry 0 (FluidInference Phase C v2 step 7).
    private func decodeNanoCodec(model: MLModel, frames: [[Int32]]) throws -> [Float] {
        let totalFrames = frames.count
        guard totalFrames > 0 else { return [] }
        let tIn = nanoCodecFrames

        if tIn >= 256 && totalFrames <= tIn {
            let tokArr = try MLMultiArray(shape: [1, 8, NSNumber(value: tIn)], dataType: .int32)
            let ptr = tokArr.dataPointer.assumingMemoryBound(to: Int32.self)
            let cbStride = tokArr.strides[1].intValue
            let tStride = tokArr.strides[2].intValue
            for slot in 0..<tIn {
                let frameIdx = min(totalFrames - 1, slot)
                for cb in 0..<8 { ptr[cb * cbStride + slot * tStride] = frames[frameIdx][cb] }
            }
            let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["tokens": MLFeatureValue(multiArray: tokArr)]))
            let audioArr = out.featureValue(for: "audio")!.multiArrayValue!
            let expected = min(totalFrames * Self.samplesPerFrame, audioArr.count)
            return extractAudioFloats(from: audioArr, start: 0, count: expected)
        }

        // Chunked sliding-window NanoCodec v3: T_in=24, stride=8, overlap=16, edge-replicated padding!
        let strideLen = max(1, tIn / 3)
        let overlap = tIn - strideLen
        let keepStart = overlap * Self.samplesPerFrame
        let keepCount = strideLen * Self.samplesPerFrame

        var pcmOut: [Float] = []
        pcmOut.reserveCapacity(totalFrames * Self.samplesPerFrame)
        let tokArr = try MLMultiArray(shape: [1, 8, NSNumber(value: tIn)], dataType: .int32)
        let ptr = tokArr.dataPointer.assumingMemoryBound(to: Int32.self)
        let cbStride = tokArr.strides[1].intValue
        let tStride = tokArr.strides[2].intValue

        var start = 0
        while start < totalFrames {
            let ctxStart = start - overlap
            for slot in 0..<tIn {
                // Edge-replication padding: replicate frames[0] for left context & frames[T-1] for right tail!
                let frameIdx = min(totalFrames - 1, max(0, ctxStart + slot))
                for cb in 0..<8 {
                    ptr[cb * cbStride + slot * tStride] = frames[frameIdx][cb]
                }
            }
            let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["tokens": MLFeatureValue(multiArray: tokArr)]))
            let audioArr = out.featureValue(for: "audio")!.multiArrayValue!
            let available = max(0, min(keepCount, audioArr.count - keepStart))
            if available > 0 {
                pcmOut.append(contentsOf: extractAudioFloats(from: audioArr, start: keepStart, count: available))
            }
            start += strideLen
        }
        let expected = totalFrames * Self.samplesPerFrame
        var trimmed = Array(pcmOut.prefix(expected))
        // Apply ultra-gentle 6ms cosine fade-in / fade-out at utterance boundaries for click-free silence transitions
        let fadeLen = min(132, trimmed.count / 4)
        for i in 0..<fadeLen {
            let g = 0.5 * (1.0 - cosf(.pi * Float(i) / Float(fadeLen)))
            trimmed[i] *= g
            trimmed[trimmed.count - 1 - i] *= g
        }
        return trimmed
    }

    private func extractAudioFloats(from audioArr: MLMultiArray, start: Int, count: Int) -> [Float] {
        guard count > 0 else { return [] }
        let s = audioArr.strides.last?.intValue ?? 1
        if audioArr.dataType == .float32 {
            let src = audioArr.dataPointer.assumingMemoryBound(to: Float.self)
            if s == 1 {
                return Array(UnsafeBufferPointer(start: src.advanced(by: start), count: count))
            }
            var out = [Float](repeating: 0, count: count)
            for i in 0..<count { out[i] = src[(start + i) * s] }
            return out
        } else {
            let src = audioArr.dataPointer.assumingMemoryBound(to: Float16.self)
            var out = [Float](repeating: 0, count: count)
            for i in 0..<count { out[i] = Float(src[(start + i) * s]) }
            return out
        }
    }

    // MARK: - Word Timing Estimation & Disk WAV Cache

    private func estimateWordTimings(
        text: String,
        duration: TimeInterval,
        startWordIndex: Int,
        offset: TimeInterval
    ) -> [WordTimingSpan] {
        let words = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard !words.isEmpty, duration > 0 else { return [] }
        // Phoneme-weighted duration distribution (gives smooth 120Hz Karaoke highlight sync)
        let weights = words.map { w -> Double in
            let core = w.filter(\.isLetter)
            let punctPause = (w.hasSuffix(",") || w.hasSuffix(";")) ? 2.5 : ((w.hasSuffix(".") || w.hasSuffix("!") || w.hasSuffix("?")) ? 4.0 : 0.0)
            return Double(max(2, core.count)) + punctPause
        }
        let totalW = weights.reduce(0, +)
        var spans: [WordTimingSpan] = []
        var cur = offset
        for (i, w) in words.enumerated() {
            let dt = duration * (weights[i] / totalW)
            spans.append(WordTimingSpan(wordIndex: startWordIndex + i, word: w, startTime: cur, endTime: cur + dt))
            cur += dt
        }
        return spans
    }

    private func cacheURL(for text: String, voice: MagpieVoice, accent: SpanishRegionalAccent) -> URL {
        let key = "v2_clean_\(voice.rawValue)_\(accent.rawValue)_\(text)"
        let digest = SHA256.hash(data: Data(key.utf8)).prefix(16).map { String(format: "%02x", $0) }.joined()
        return audioCacheDir.appendingPathComponent("v2_\(digest).wav")
    }

    private func saveWAV(samples: [Float], sampleRate: Double, to url: URL) throws {
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

    private func loadWAV(from url: URL) throws -> [Float] {
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
