import CadenciaCore
import Compression
import CoreML
import FluidAudio
import Foundation

public enum ManagedModelKind: String, CaseIterable, Identifiable, Sendable {
    case phononSpanish = "phonon_40m_es"
    case kokoroEnglish = "kokoro_82m_en"
    case parakeetUltra = "parakeet_ultra_int8"
    case zipaPhoneme = "zipa_cr_large_300m_fp16"
    case sileroVAD = "silero_vad"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .phononSpanish: return "Gradium Phonon-40m-es"
        case .kokoroEnglish: return "Kokoro-82M (English Tutor)"
        case .parakeetUltra: return "Parakeet Ultra 600M (INT8)"
        case .zipaPhoneme: return "ZIPA CR Large 300M (FP16)"
        case .sileroVAD: return "Silero VAD (Core ML)"
        }
    }

    public var roleDescription: String {
        switch self {
        case .phononSpanish: return "Native Spanish voice for all passages & ledger · 24 kHz ANE · 8 es-ES & es-MX voices"
        case .kokoroEnglish: return "On-device English conversation tutor · 24 kHz · FP32 Vocoder"
        case .parakeetUltra: return "On-device learner speech-to-text · INT8 ANE · >100× real-time"
        case .zipaPhoneme: return "On-device IPA phoneme recognizer · FP16 Multifunction Metal GPU"
        case .sileroVAD: return "Real-time speech endpointing & silence trimming"
        }
    }

    public var estimatedSizeMB: Int {
        switch self {
        case .phononSpanish: return 125
        case .kokoroEnglish: return 210
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

/// Manages on-device storage, built-in bundle discovery, and direct on-device cloud downloads
/// (HuggingFace & GitHub Releases CDN) for all 5 Core ML models on iPhone 18 Pro Max.
/// Zero computer/LAN interaction required.
public actor ModelAssetManager {
    public static let phononModelRepo = "gradium/phonon-40m-es"
    public static let phononVoicesRepo = "gradium/phonon-voices"
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
        case .phononSpanish:
            let hasMeta = fm.fileExists(atPath: dir.appendingPathComponent("bundle.json").path)
            let hasHost = fm.fileExists(atPath: dir.appendingPathComponent("host.safetensors").path)
            let hasTok = fm.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path)
            let items = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
            let hasFlow = items.contains { $0.hasPrefix("flow-") && ($0.hasSuffix(".mlpackage") || $0.hasSuffix(".mlmodelc")) }
            let hasMimi = items.contains { $0.hasPrefix("mimi-") && ($0.hasSuffix(".mlpackage") || $0.hasSuffix(".mlmodelc")) }
            let voiceFiles = (try? fm.contentsOfDirectory(atPath: dir.appendingPathComponent("voices").path)) ?? []
            let hasVoices = voiceFiles.contains { $0.hasSuffix(".safetensors") }
            return hasMeta && hasHost && hasTok && hasFlow && hasMimi && hasVoices

        case .kokoroEnglish:
            guard fm.fileExists(atPath: dir.appendingPathComponent(".ready").path) else { return false }
            let contents = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
            return contents.contains { $0 != ".ready" && !$0.hasPrefix(".") }

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

    /// Returns the number of installed native Spanish voice embeddings in the local Phonon bundle.
    public func installedPhononVoiceCount() -> Int {
        let voicesDir = directory(for: .phononSpanish).appendingPathComponent("voices")
        let files = (try? FileManager.default.contentsOfDirectory(atPath: voicesDir.path)) ?? []
        return files.filter { $0.hasSuffix(".safetensors") }.count
    }

    // MARK: - 1. Download Gradium Phonon-40m-es (User HF Auth + Public Native Spanish Voices + Pure-Rust Core ML Export)

    /// Downloads `gradium/phonon-40m-es` directly from Hugging Face using the user's own Keychain HF Read token,
    /// downloads all 8 native `es-ES` & `es-MX` voices from `gradium/phonon-voices`, converts them on-device in
    /// pure Rust (`ptts_export_coreml`) into Apple Neural Engine Core ML packages, and removes the raw download.
    public func downloadPhononSpanish(
        hfToken: String,
        onProgress: @escaping @Sendable (ModelDownloadProgress) -> Void
    ) async throws -> URL {
        let token = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            throw NSError(domain: "ModelAssetManager", code: 401, userInfo: [
                NSLocalizedDescriptionKey: "Hugging Face token required. Accept gradium/phonon-40m-es on huggingface.co and save your free Read token in Settings -> Hugging Face Vault."
            ])
        }

        let fm = FileManager.default
        let destRoot = directory(for: .phononSpanish)
        let rawDir = rootDirectory.appendingPathComponent("phonon_40m_es_raw", isDirectory: true)
        let rawVoicesDir = rawDir.appendingPathComponent("voices", isDirectory: true)
        let stagingDir = rootDirectory.appendingPathComponent("phonon_40m_es_staging", isDirectory: true)

        try? fm.removeItem(at: rawDir)
        try? fm.removeItem(at: stagingDir)
        try fm.createDirectory(at: rawVoicesDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: stagingDir, withIntermediateDirectories: true)

        defer {
            try? fm.removeItem(at: rawDir)
        }

        onProgress(.init(
            kind: .phononSpanish,
            fractionCompleted: 0.03,
            statusMessage: "Authenticating with Hugging Face for gradium/phonon-40m-es…",
            isReady: false,
            errorMessage: nil
        ))

        // 1. Download checkpoint metadata + tokenizer + full-fidelity model.safetensors (116 MB)
        let smallFiles = ["config.json", "tokenizer.json"]
        for (idx, file) in smallFiles.enumerated() {
            let url = URL(string: "https://huggingface.co/\(Self.phononModelRepo)/resolve/main/\(file)")!
            try await downloadFile(
                from: url,
                to: rawDir.appendingPathComponent(file),
                bearerToken: token
            )
            onProgress(.init(
                kind: .phononSpanish,
                fractionCompleted: 0.04 + 0.02 * Double(idx + 1),
                statusMessage: "Downloaded \(file)…",
                isReady: false,
                errorMessage: nil
            ))
        }

        let weightsURL = URL(string: "https://huggingface.co/\(Self.phononModelRepo)/resolve/main/model.safetensors")!
        try await downloadFileWithProgress(
            from: weightsURL,
            to: rawDir.appendingPathComponent("model.safetensors"),
            bearerToken: token
        ) { frac in
            onProgress(.init(
                kind: .phononSpanish,
                fractionCompleted: 0.08 + 0.60 * frac,
                statusMessage: "Downloading Phonon-40m-es weights: \(Int(frac * 100))%",
                isReady: false,
                errorMessage: nil
            ))
        }

        // 2. Download all 8 curated native Spanish voices (4 Castilian es-ES + 4 Mexican es-MX, public repo, ~2 MB total)
        let voices = TutorVoice.allCases
        for (idx, voice) in voices.enumerated() {
            let voiceURL = URL(string: "https://huggingface.co/\(Self.phononVoicesRepo)/resolve/main/voices/\(voice.phononVoiceId).safetensors")!
            let destVoice = rawVoicesDir.appendingPathComponent("\(voice.phononBundleName).safetensors")
            onProgress(.init(
                kind: .phononSpanish,
                fractionCompleted: 0.68 + 0.12 * (Double(idx) / Double(voices.count)),
                statusMessage: "Downloading native \(voice.localeLabel) voice: \(voice.rawValue)…",
                isReady: false,
                errorMessage: nil
            ))
            try await downloadFile(from: voiceURL, to: destVoice, bearerToken: nil)
        }

        // 3. Export pure-Rust Core ML packages on-device (`ptts_export_coreml`)
        onProgress(.init(
            kind: .phononSpanish,
            fractionCompleted: 0.82,
            statusMessage: "Exporting Phonon-40m-es Core ML graphs on device…",
            isReady: false,
            errorMessage: nil
        ))
        try await PhononSpanishEngine.exportCoreMLBundle(
            checkpointDir: rawDir,
            voicesDir: rawVoicesDir,
            outputDir: stagingDir
        )

        // 4. Atomically install into CadenciaModels/phonon_40m_es
        if fm.fileExists(atPath: destRoot.path) {
            try fm.removeItem(at: destRoot)
        }
        try fm.moveItem(at: stagingDir, to: destRoot)

        onProgress(.init(
            kind: .phononSpanish,
            fractionCompleted: 0.90,
            statusMessage: "Compiling Phonon-40m-es for Apple Neural Engine…",
            isReady: false,
            errorMessage: nil
        ))
        return destRoot
    }

    // MARK: - 2. Download Kokoro-82M English Tutor via FluidAudio

    public func downloadKokoroEnglish(
        onProgress: @escaping @Sendable (ModelDownloadProgress) -> Void
    ) async throws -> URL {
        let destRoot = directory(for: .kokoroEnglish)
        try FileManager.default.createDirectory(at: destRoot, withIntermediateDirectories: true)

        onProgress(.init(
            kind: .kokoroEnglish,
            fractionCompleted: 0.04,
            statusMessage: "Connecting to Hugging Face for Kokoro-82M English Core ML…",
            isReady: false,
            errorMessage: nil
        ))

        _ = try await KokoroAneResourceDownloader.ensureModels(
            variant: .english,
            directory: destRoot,
            additionalModelNames: ["KokoroVocoder_v2.mlmodelc"],
            progressHandler: { prog in
                let dlFrac = min(1.0, max(0.0, prog.fractionCompleted <= 0.51 ? prog.fractionCompleted * 2.0 : prog.fractionCompleted))
                let uiFrac = 0.05 + 0.72 * dlFrac
                let message: String
                switch prog.phase {
                case .listing:
                    message = "Listing Kokoro-82M English Core ML files…"
                case .downloading(let done, let total):
                    if total > 0 {
                        message = "Downloading Kokoro-82M English (\(done)/\(total) files · \(Int(dlFrac * 100))%)…"
                    } else {
                        message = "Downloading Kokoro-82M English (\(Int(dlFrac * 100))%)…"
                    }
                case .compiling(let name):
                    message = "Compiling \(name)…"
                }
                onProgress(.init(
                    kind: .kokoroEnglish,
                    fractionCompleted: uiFrac,
                    statusMessage: message,
                    isReady: false,
                    errorMessage: nil
                ))
            }
        )

        onProgress(.init(
            kind: .kokoroEnglish,
            fractionCompleted: 0.84,
            statusMessage: "Downloading English G2P lexicon & voices (af_heart, am_michael)…",
            isReady: false,
            errorMessage: nil
        ))
        return destRoot
    }

    public func markKokoroReady() {
        let dir = directory(for: .kokoroEnglish)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: dir.appendingPathComponent(".ready").path, contents: Data())
    }

    // MARK: - 3. Download Parakeet Ultra 600M INT8 via FluidAudio

    public func downloadParakeetUltra(
        onProgress: @escaping @Sendable (ModelDownloadProgress) -> Void
    ) async throws -> AsrModels {
        onProgress(.init(
            kind: .parakeetUltra,
            fractionCompleted: 0.04,
            statusMessage: "Connecting to HuggingFace for Parakeet Ultra INT8…",
            isReady: false,
            errorMessage: nil
        ))

        let targetDir = try await AsrModels.download(
            version: .ultra,
            encoderPrecision: .int8,
            encoderComputeUnits: .cpuAndNeuralEngine,
            progressHandler: { prog in
                let dlFrac = min(1.0, max(0.0, prog.fractionCompleted <= 0.51 ? prog.fractionCompleted * 2.0 : prog.fractionCompleted))
                let uiFrac = 0.05 + 0.70 * dlFrac
                let message: String
                switch prog.phase {
                case .listing:
                    message = "Listing Parakeet Ultra INT8 files…"
                case .downloading(let done, let total):
                    if total > 0 {
                        message = "Downloading Parakeet Ultra INT8 (\(done)/\(total) files · \(Int(dlFrac * 100))%)…"
                    } else {
                        message = "Downloading Parakeet Ultra INT8 (\(Int(dlFrac * 100))%)…"
                    }
                case .compiling(let name):
                    message = "Compiling \(name) for Apple Neural Engine…"
                }
                onProgress(.init(
                    kind: .parakeetUltra,
                    fractionCompleted: uiFrac,
                    statusMessage: message,
                    isReady: false,
                    errorMessage: nil
                ))
            }
        )

        onProgress(.init(
            kind: .parakeetUltra,
            fractionCompleted: 0.78,
            statusMessage: "Download complete · Compiling Parakeet INT8 for Neural Engine…",
            isReady: false,
            errorMessage: nil
        ))

        let models = try await AsrModels.load(
            from: targetDir,
            version: .ultra,
            encoderPrecision: .int8,
            encoderComputeUnits: .cpuAndNeuralEngine,
            progressHandler: { prog in
                let compileMsg: String
                switch prog.phase {
                case .compiling(let name):
                    compileMsg = "Compiling \(name) for Neural Engine…"
                default:
                    compileMsg = "Optimizing Parakeet Ultra INT8 on Neural Engine…"
                }
                let uiFrac = min(0.96, max(0.80, 0.78 + 0.18 * prog.fractionCompleted))
                onProgress(.init(
                    kind: .parakeetUltra,
                    fractionCompleted: uiFrac,
                    statusMessage: compileMsg,
                    isReady: false,
                    errorMessage: nil
                ))
            }
        )

        onProgress(.init(
            kind: .parakeetUltra,
            fractionCompleted: 1.0,
            statusMessage: "Installed & ready on device",
            isReady: true,
            errorMessage: nil
        ))
        return models
    }

    // MARK: - 4. Zero-Computer ZIPA CR Large 300M FP16 (Built-In Bundle OR Direct Cloud CDN)

    public func installZipaFromCloud(
        onProgress: @escaping @Sendable (ModelDownloadProgress) -> Void
    ) async throws -> URL {
        if let bundled = Bundle.main.url(forResource: "ZipaCRLarge300M_FP16", withExtension: "mlmodelc") {
            onProgress(.init(
                kind: .zipaPhoneme,
                fractionCompleted: 1.0,
                statusMessage: "Built-in · Metal GPU Ready",
                isReady: true,
                errorMessage: nil
            ))
            return bundled
        }

        let destRoot = directory(for: .zipaPhoneme)
        try FileManager.default.createDirectory(at: destRoot, withIntermediateDirectories: true)
        let compiledFinal = destRoot.appendingPathComponent("ZipaCRLarge300M_FP16.mlmodelc")
        if FileManager.default.fileExists(atPath: compiledFinal.path) {
            onProgress(.init(kind: .zipaPhoneme, fractionCompleted: 1.0, statusMessage: "ZIPA CR Large 300M FP16 ready", isReady: true, errorMessage: nil))
            return compiledFinal
        }

        guard let urlStr = (Bundle.main.object(forInfoDictionaryKey: Self.zipaInfoPlistKey) as? String)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              !urlStr.isEmpty,
              let remoteURL = URL(string: urlStr)
        else {
            throw NSError(domain: "ModelAssetManager", code: 404, userInfo: [
                NSLocalizedDescriptionKey: "ZIPA FP16 cloud release URL is not configured in this build."
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

        try await downloadFileWithProgress(from: remoteURL, to: tmpZip, bearerToken: nil) { frac in
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
            statusMessage: "Extracting precompiled ZIPA model bundle…",
            isReady: false,
            errorMessage: nil
        ))

        try MiniZipExtractor.extract(zipURL: tmpZip, to: destRoot)
        try? FileManager.default.removeItem(at: tmpZip)

        let pkgDest = destRoot.appendingPathComponent("ZipaCRLarge300M_FP16.mlpackage")
        if !FileManager.default.fileExists(atPath: compiledFinal.path),
           FileManager.default.fileExists(atPath: pkgDest.path) {
            onProgress(.init(
                kind: .zipaPhoneme,
                fractionCompleted: 0.93,
                statusMessage: "Compiling ZIPA FP16…",
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
            statusMessage: "ZIPA CR Large 300M FP16 ready",
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

    // MARK: - Authenticated & Streaming HTTP Helpers

    private func makeRequest(url: URL, bearerToken: String?) -> URLRequest {
        var req = URLRequest(url: url)
        req.timeoutInterval = 300
        if let token = bearerToken, !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return req
    }

    private func formatHTTPError(code: Int, fileName: String, hasBearer: Bool) -> NSError {
        let msg: String
        if (code == 401 || code == 403) && hasBearer {
            msg = "HTTP \(code) downloading \(fileName): accept the license at huggingface.co/gradium/phonon-40m-es and verify your Hugging Face Read token."
        } else {
            msg = "Download failed for \(fileName) (HTTP \(code))"
        }
        return NSError(domain: "ModelAssetManager", code: code, userInfo: [NSLocalizedDescriptionKey: msg])
    }

    private func downloadFile(from remote: URL, to local: URL, bearerToken: String?) async throws {
        try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
        let req = makeRequest(url: remote, bearerToken: bearerToken)
        let (tmpURL, response) = try await session.download(for: req)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw formatHTTPError(code: code, fileName: remote.lastPathComponent, hasBearer: bearerToken != nil)
        }
        if FileManager.default.fileExists(atPath: local.path) {
            try FileManager.default.removeItem(at: local)
        }
        try FileManager.default.moveItem(at: tmpURL, to: local)
    }

    private func downloadFileWithProgress(
        from remote: URL,
        to local: URL,
        bearerToken: String?,
        onFraction: @Sendable (Double) -> Void
    ) async throws {
        try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
        let req = makeRequest(url: remote, bearerToken: bearerToken)
        let (bytes, response) = try await session.bytes(for: req)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw formatHTTPError(code: code, fileName: remote.lastPathComponent, hasBearer: bearerToken != nil)
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

            if method == 0 {
                fm.createFile(atPath: outURL.path, contents: nil)
                let outHandle = try FileHandle(forWritingTo: outURL)
                defer { try? outHandle.close() }
                let chunkSize = 4 * 1024 * 1024
                var offset = payloadStart
                while offset < payloadEnd {
                    let end = min(payloadEnd, offset + chunkSize)
                    let slice = data[offset..<end]
                    try outHandle.write(contentsOf: slice)
                    offset = end
                }
            } else if method == 8 {
                let compressedSlice = data.subdata(in: payloadStart..<payloadEnd)
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
