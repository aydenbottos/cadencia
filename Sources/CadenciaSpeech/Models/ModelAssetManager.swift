import Compression
import CoreML
import FluidAudio
import Foundation

public enum ManagedModelKind: String, CaseIterable, Identifiable, Sendable {
    case magpieTTS = "magpie_v2607"
    case parakeetUltra = "parakeet_ultra_int8"
    case zipaPhoneme = "zipa_cr_large_300m_fp16"
    case sileroVAD = "silero_vad"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .magpieTTS: return "NVIDIA Magpie v2607 TTS"
        case .parakeetUltra: return "Parakeet Ultra 600M (INT8)"
        case .zipaPhoneme: return "ZIPA CR Large 300M (FP16)"
        case .sileroVAD: return "Silero VAD (Core ML)"
        }
    }

    public var roleDescription: String {
        switch self {
        case .magpieTTS: return "On-device neural Spanish tutor · 5 voices · 22.05kHz NanoCodec"
        case .parakeetUltra: return "On-device learner speech-to-text · INT8 ANE · >100× real-time"
        case .zipaPhoneme: return "On-device IPA phoneme recognizer · FP16 Multifunction ANE"
        case .sileroVAD: return "Real-time speech endpointing & silence trimming"
        }
    }

    public var estimatedSizeMB: Int {
        switch self {
        case .magpieTTS: return 680
        case .parakeetUltra: return 640
        case .zipaPhoneme: return 590
        case .sileroVAD: return 4
        }
    }
}

public struct ModelDownloadProgress: Sendable {
    public let kind: ManagedModelKind
    public let fractionCompleted: Double
    public let statusMessage: String
    public let isReady: Bool
    public let errorMessage: String?

    public init(
        kind: ManagedModelKind,
        fractionCompleted: Double,
        statusMessage: String,
        isReady: Bool,
        errorMessage: String?
    ) {
        self.kind = kind
        self.fractionCompleted = fractionCompleted
        self.statusMessage = statusMessage
        self.isReady = isReady
        self.errorMessage = errorMessage
    }
}

/// Manages on-device storage, built-in bundle discovery, and direct cloud downloads
/// (HuggingFace & GitHub Releases CDN) for all 4 Core ML models on iPhone 18 Pro Max.
/// Zero computer/LAN interaction required.
public actor ModelAssetManager {
    public static let magpieHuggingFaceRepo = "FluidInference/magpie-tts-multilingual-357m-coreml"
    public static let zipaInfoPlistKey = "CadenciaZipaReleaseURL"

    public let rootDirectory: URL
    private let session: URLSession

    public init(rootDirectory: URL? = nil, session: URLSession = .shared) {
        if let rootDirectory {
            self.rootDirectory = rootDirectory
        } else {
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            self.rootDirectory = appSupport.appendingPathComponent("CadenciaModels", isDirectory: true)
        }
        self.session = session
        try? FileManager.default.createDirectory(at: self.rootDirectory, withIntermediateDirectories: true)
    }

    public func directory(for kind: ManagedModelKind) -> URL {
        rootDirectory.appendingPathComponent(kind.rawValue, isDirectory: true)
    }

    /// Returns true when `ZipaCRLarge300M_FP16.mlmodelc` is baked straight into the iOS `.app` bundle
    /// (default CI build — zero download, zero compile delay, works offline on first launch).
    public func isZipaBundledInApp() -> Bool {
        Bundle.main.url(forResource: "ZipaCRLarge300M_FP16", withExtension: "mlmodelc") != nil
    }

    public func isModelInstalled(_ kind: ManagedModelKind) -> Bool {
        let fm = FileManager.default
        let dir = directory(for: kind)
        switch kind {
        case .magpieTTS:
            let req = [
                "text_encoder.mlmodelc",
                "decoder_prefill.mlmodelc",
                "decoder_step.mlmodelc",
                "constants/constants.json",
                "tokenizer/spanish_phoneme_token2id.json",
            ]
            let hasCodec = fm.fileExists(atPath: dir.appendingPathComponent("nanocodec_decoder_v3.mlmodelc").path)
                || fm.fileExists(atPath: dir.appendingPathComponent("nanocodec_decoder.mlmodelc").path)
            return hasCodec && req.allSatisfy { fm.fileExists(atPath: dir.appendingPathComponent($0).path) }

        case .parakeetUltra:
            return AsrModels.modelsExist(
                at: AsrModels.defaultCacheDirectory(for: .ultra),
                version: .ultra,
                encoderPrecision: .int8
            )

        case .zipaPhoneme:
            if isZipaBundledInApp() { return true }
            let compiled = dir.appendingPathComponent("ZipaCRLarge300M_FP16.mlmodelc")
            let pkg = dir.appendingPathComponent("ZipaCRLarge300M_FP16.mlpackage")
            return fm.fileExists(atPath: compiled.path) || fm.fileExists(atPath: pkg.path)

        case .sileroVAD:
            return fm.fileExists(atPath: dir.appendingPathComponent(".ready").path)
        }
    }

    // MARK: - Download NVIDIA Magpie v2607 Core ML + Spanish Phoneme Assets

    public func downloadMagpieAssets(
        repoId: String = ModelAssetManager.magpieHuggingFaceRepo,
        onProgress: @Sendable (ModelDownloadProgress) -> Void
    ) async throws {
        let destRoot = directory(for: .magpieTTS)
        try FileManager.default.createDirectory(at: destRoot, withIntermediateDirectories: true)

        onProgress(.init(kind: .magpieTTS, fractionCompleted: 0.02, statusMessage: "Fetching Magpie manifest…", isReady: false, errorMessage: nil))

        var filesToFetch: [String] = [
            "manifest.json",
            "constants/constants.json",
            "constants/speaker_info.json",
            "constants/tokenizer_info.json",
            "constants/tokenizer_metadata.json",
            "constants/speaker_0.npy",
            "constants/speaker_1.npy",
            "constants/speaker_2.npy",
            "constants/speaker_3.npy",
            "constants/speaker_4.npy",
            "constants/text_embedding.npy",
            "tokenizer/spanish_phoneme_token2id.json",
            "tokenizer/spanish_phoneme_phoneme_dict.json",
            "constants/local_transformer/in_proj_weight.npy",
            "constants/local_transformer/in_proj_bias.npy",
            "constants/local_transformer/pos_emb.npy",
            "constants/local_transformer/norm1_weight.npy",
            "constants/local_transformer/norm2_weight.npy",
            "constants/local_transformer/sa_qkv_weight.npy",
            "constants/local_transformer/sa_o_weight.npy",
            "constants/local_transformer/ffn_conv1_weight.npy",
            "constants/local_transformer/ffn_conv2_weight.npy",
        ]
        for cb in 0..<8 {
            filesToFetch.append("constants/audio_embedding_\(cb).npy")
            filesToFetch.append("constants/local_transformer/out_proj_\(cb)_weight.npy")
            filesToFetch.append("constants/local_transformer/out_proj_\(cb)_bias.npy")
        }

        let bundles = [
            "text_encoder.mlmodelc",
            "decoder_prefill.mlmodelc",
            "decoder_step.mlmodelc",
            "nanocodec_decoder_v3.mlmodelc",
        ]
        let bundleSubfiles = [
            "coremldata.bin",
            "metadata.json",
            "model.mil",
            "weights/weight.bin",
            "analytics/coremldata.bin",
        ]
        for b in bundles {
            for sub in bundleSubfiles {
                filesToFetch.append("\(b)/\(sub)")
            }
        }

        let total = Double(filesToFetch.count)
        for (idx, relPath) in filesToFetch.enumerated() {
            let localURL = destRoot.appendingPathComponent(relPath)
            if FileManager.default.fileExists(atPath: localURL.path) { continue }
            let remoteURL = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(relPath)")!
            onProgress(.init(
                kind: .magpieTTS,
                fractionCompleted: Double(idx) / total,
                statusMessage: "Downloading \(relPath)…",
                isReady: false,
                errorMessage: nil
            ))
            try await downloadFile(from: remoteURL, to: localURL)
        }

        onProgress(.init(kind: .magpieTTS, fractionCompleted: 1.0, statusMessage: "Magpie v2607 Core ML ready", isReady: true, errorMessage: nil))
    }

    // MARK: - Download Parakeet Ultra 600M INT8 via FluidAudio

    public func downloadParakeetUltra(
        onProgress: @Sendable (ModelDownloadProgress) -> Void
    ) async throws -> AsrModels {
        onProgress(.init(kind: .parakeetUltra, fractionCompleted: 0.05, statusMessage: "Downloading Parakeet Ultra 600M INT8 Core ML…", isReady: false, errorMessage: nil))
        let models = try await AsrModels.downloadAndLoad(
            version: .ultra,
            encoderPrecision: .int8,
            encoderComputeUnits: .cpuAndNeuralEngine,
            progressHandler: { prog in
                onProgress(.init(
                    kind: .parakeetUltra,
                    fractionCompleted: prog.fractionCompleted,
                    statusMessage: "Parakeet Ultra INT8: \(Int(prog.fractionCompleted * 100))%",
                    isReady: false,
                    errorMessage: nil
                ))
            }
        )
        onProgress(.init(kind: .parakeetUltra, fractionCompleted: 1.0, statusMessage: "Parakeet Ultra 600M INT8 ready on ANE", isReady: true, errorMessage: nil))
        return models
    }

    // MARK: - Zero-Computer ZIPA CR Large 300M FP16 (Built-In Bundle OR Direct Cloud CDN)

    /// Returns the precompiled `ZipaCRLarge300M_FP16.mlmodelc` baked into `Cadencia.app` immediately,
    /// or downloads & extracts `ZipaCRLarge300M_FP16.mlmodelc.zip` directly on-device from the
    /// CI-configured GitHub Release CDN (`Info.plist -> CadenciaZipaReleaseURL`).
    public func installZipaFromCloud(
        onProgress: @Sendable (ModelDownloadProgress) -> Void
    ) async throws -> URL {
        // 1. Instant path: already built into Cadencia.app by GitHub Actions (zero download!)
        if let bundled = Bundle.main.url(forResource: "ZipaCRLarge300M_FP16", withExtension: "mlmodelc") {
            onProgress(.init(
                kind: .zipaPhoneme,
                fractionCompleted: 1.0,
                statusMessage: "Built-in · ANE Ready",
                isReady: true,
                errorMessage: nil
            ))
            return bundled
        }

        let destRoot = directory(for: .zipaPhoneme)
        try FileManager.default.createDirectory(at: destRoot, withIntermediateDirectories: true)
        let compiledFinal = destRoot.appendingPathComponent("ZipaCRLarge300M_FP16.mlmodelc")
        if FileManager.default.fileExists(atPath: compiledFinal.path) {
            onProgress(.init(kind: .zipaPhoneme, fractionCompleted: 1.0, statusMessage: "ZIPA CR Large 300M FP16 ready on ANE", isReady: true, errorMessage: nil))
            return compiledFinal
        }

        // 2. Cloud download path (for lightweight IPA builds): uses auto-injected GitHub Release CDN URL
        guard let urlStr = (Bundle.main.object(forInfoDictionaryKey: Self.zipaInfoPlistKey) as? String)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              !urlStr.isEmpty,
              let remoteURL = URL(string: urlStr)
        else {
            throw NSError(domain: "ModelAssetManager", code: 404, userInfo: [
                NSLocalizedDescriptionKey: "ZIPA FP16 cloud release URL is not configured in this build. Rebuild with GitHub Actions (default builds bundle ZIPA directly inside the app)."
            ])
        }

        let tmpZip = destRoot.appendingPathComponent("zipa_download.zip")
        try? FileManager.default.removeItem(at: tmpZip)

        onProgress(.init(
            kind: .zipaPhoneme,
            fractionCompleted: 0.05,
            statusMessage: "Downloading ZIPA CR Large 300M FP16 from cloud release…",
            isReady: false,
            errorMessage: nil
        ))

        try await downloadFileWithProgress(from: remoteURL, to: tmpZip) { frac in
            onProgress(.init(
                kind: .zipaPhoneme,
                fractionCompleted: 0.05 + 0.80 * frac,
                statusMessage: "Downloading ZIPA FP16: \(Int(frac * 100))%",
                isReady: false,
                errorMessage: nil
            ))
        }

        onProgress(.init(
            kind: .zipaPhoneme,
            fractionCompleted: 0.88,
            statusMessage: "Extracting precompiled ANE model bundle…",
            isReady: false,
            errorMessage: nil
        ))

        try MiniZipExtractor.extract(zipURL: tmpZip, to: destRoot)
        try? FileManager.default.removeItem(at: tmpZip)

        // If the archive contained an uncompiled .mlpackage instead of .mlmodelc, compile natively on-device
        let pkgDest = destRoot.appendingPathComponent("ZipaCRLarge300M_FP16.mlpackage")
        if !FileManager.default.fileExists(atPath: compiledFinal.path),
           FileManager.default.fileExists(atPath: pkgDest.path) {
            onProgress(.init(
                kind: .zipaPhoneme,
                fractionCompleted: 0.93,
                statusMessage: "Compiling ZIPA FP16 for Neural Engine…",
                isReady: false,
                errorMessage: nil
            ))
            let compiledTmp = try await MLModel.compileModel(at: pkgDest)
            try FileManager.default.moveItem(at: compiledTmp, to: compiledFinal)
            try? FileManager.default.removeItem(at: pkgDest)
        }

        guard FileManager.default.fileExists(atPath: compiledFinal.path) else {
            throw NSError(domain: "ModelAssetManager", code: 500, userInfo: [
                NSLocalizedDescriptionKey: "Downloaded archive did not contain ZipaCRLarge300M_FP16.mlmodelc."
            ])
        }

        onProgress(.init(
            kind: .zipaPhoneme,
            fractionCompleted: 1.0,
            statusMessage: "ZIPA CR Large 300M FP16 ready on ANE",
            isReady: true,
            errorMessage: nil
        ))
        return compiledFinal
    }

    public func markVADReady() {
        let dir = directory(for: .sileroVAD)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: dir.appendingPathComponent(".ready").path, contents: Data())
    }

    private func downloadFile(from remote: URL, to local: URL) async throws {
        try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
        let (tmpURL, response) = try await session.download(from: remote)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw NSError(domain: "ModelAssetManager", code: code, userInfo: [
                NSLocalizedDescriptionKey: "Failed to download \(remote.lastPathComponent) (HTTP \(code))"
            ])
        }
        if FileManager.default.fileExists(atPath: local.path) {
            try FileManager.default.removeItem(at: local)
        }
        try FileManager.default.moveItem(at: tmpURL, to: local)
    }

    private func downloadFileWithProgress(
        from remote: URL,
        to local: URL,
        onFraction: @Sendable (Double) -> Void
    ) async throws {
        try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
        let (bytes, response) = try await session.bytes(from: remote)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw NSError(domain: "ModelAssetManager", code: code, userInfo: [
                NSLocalizedDescriptionKey: "Cloud download failed for \(remote.lastPathComponent) (HTTP \(code))"
            ])
        }
        let expectedLen = max(1, response.expectedContentLength)
        FileManager.default.createFile(atPath: local.path, contents: nil)
        let handle = try FileHandle(forWritingTo: local)
        defer { try? handle.close() }

        var buffer = Data()
        buffer.reserveCapacity(1_048_576)
        var written: Int64 = 0
        var lastReportedBucket = -1

        for try await byte in bytes {
            buffer.append(byte)
            if buffer.count >= 1_048_576 {
                try handle.write(contentsOf: buffer)
                written += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                let frac = min(1.0, Double(written) / Double(expectedLen))
                let bucket = Int(frac * 100)
                if bucket != lastReportedBucket {
                    lastReportedBucket = bucket
                    onFraction(frac)
                }
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
        }
        onFraction(1.0)
    }
}

// MARK: - Zero-Dependency Native ZIP Extractor (Stored + Deflate via Apple Compression.framework)

public enum MiniZipExtractor {
    public static func extract(zipURL: URL, to destinationDir: URL) throws {
        let data = try Data(contentsOf: zipURL, options: .mappedIfSafe)
        let totalLen = data.count
        guard totalLen >= 22 else {
            throw NSError(domain: "MiniZipExtractor", code: 1, userInfo: [NSLocalizedDescriptionKey: "Truncated ZIP archive."])
        }

        // Locate End of Central Directory record (0x06054b50) scanning backwards from EOF
        let minSearch = max(0, totalLen - 65_557)
        var eocdOffset = -1
        var cursor = totalLen - 22
        while cursor >= minSearch {
            if readUInt32LE(data, at: cursor) == 0x06054b50 {
                eocdOffset = cursor
                break
            }
            cursor -= 1
        }
        guard eocdOffset >= 0 else {
            throw NSError(domain: "MiniZipExtractor", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid ZIP archive (missing EOCD)."])
        }

        let totalEntries = Int(readUInt16LE(data, at: eocdOffset + 10))
        let cdOffset = Int(readUInt32LE(data, at: eocdOffset + 16))
        var cdPos = cdOffset
        let fm = FileManager.default
        let canonicalDest = destinationDir.standardizedFileURL.path

        for _ in 0..<totalEntries {
            guard cdPos + 46 <= totalLen, readUInt32LE(data, at: cdPos) == 0x02014b50 else { break }
            let method = readUInt16LE(data, at: cdPos + 10)
            let compSize = Int(readUInt32LE(data, at: cdPos + 20))
            let uncompSize = Int(readUInt32LE(data, at: cdPos + 24))
            let nameLen = Int(readUInt16LE(data, at: cdPos + 28))
            let extraLen = Int(readUInt16LE(data, at: cdPos + 30))
            let commentLen = Int(readUInt16LE(data, at: cdPos + 32))
            let localHeaderOff = Int(readUInt32LE(data, at: cdPos + 42))

            let nameStart = cdPos + 46
            guard nameStart + nameLen <= totalLen else { break }
            let name = String(data: data.subdata(in: nameStart..<(nameStart + nameLen)), encoding: .utf8) ?? ""
            cdPos = nameStart + nameLen + extraLen + commentLen

            // Guard against zip-slip path traversal
            let outURL = destinationDir.appendingPathComponent(name).standardizedFileURL
            guard outURL.path.hasPrefix(canonicalDest) else { continue }

            if name.hasSuffix("/") {
                try fm.createDirectory(at: outURL, withIntermediateDirectories: true)
                continue
            }

            guard localHeaderOff + 30 <= totalLen, readUInt32LE(data, at: localHeaderOff) == 0x04034b50 else {
                continue
            }
            let localNameLen = Int(readUInt16LE(data, at: localHeaderOff + 26))
            let localExtraLen = Int(readUInt16LE(data, at: localHeaderOff + 28))
            let payloadStart = localHeaderOff + 30 + localNameLen + localExtraLen
            let payloadEnd = payloadStart + compSize
            guard payloadEnd <= totalLen else { continue }

            try fm.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let compressedSlice = data.subdata(in: payloadStart..<payloadEnd)

            if method == 0 {
                // Stored (uncompressed) — zero-overhead direct write!
                try compressedSlice.write(to: outURL, options: .atomic)
            } else if method == 8 {
                // Raw DEFLATE (RFC 1951) decoded via Apple's hardware-accelerated Compression.framework
                var output = Data(count: max(1, uncompSize))
                let decodedCount: Int = output.withUnsafeMutableBytes { dstBuf in
                    compressedSlice.withUnsafeBytes { srcBuf in
                        compression_decode_buffer(
                            dstBuf.bindMemory(to: UInt8.self).baseAddress!,
                            max(1, uncompSize),
                            srcBuf.bindMemory(to: UInt8.self).baseAddress!,
                            compSize,
                            nil,
                            COMPRESSION_ZLIB
                        )
                    }
                }
                guard decodedCount == uncompSize else {
                    throw NSError(domain: "MiniZipExtractor", code: 3, userInfo: [
                        NSLocalizedDescriptionKey: "Deflate extraction size mismatch for \(name) (\(decodedCount) != \(uncompSize))."
                    ])
                }
                try output.prefix(decodedCount).write(to: outURL, options: .atomic)
            } else {
                throw NSError(domain: "MiniZipExtractor", code: 4, userInfo: [
                    NSLocalizedDescriptionKey: "Unsupported ZIP compression method \(method) for \(name)."
                ])
            }
        }
    }

    private static func readUInt16LE(_ data: Data, at offset: Int) -> UInt16 {
        UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
    }

    private static func readUInt32LE(_ data: Data, at offset: Int) -> UInt32 {
        UInt32(data[offset])
            | (UInt32(data[offset + 1]) << 8)
            | (UInt32(data[offset + 2]) << 16)
            | (UInt32(data[offset + 3]) << 24)
    }
}
