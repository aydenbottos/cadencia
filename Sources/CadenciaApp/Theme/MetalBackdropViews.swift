import CadenciaCore
import MetalKit
import QuartzCore
import SwiftUI

// MARK: - Runtime-Compiled Metal Library (Guaranteed on BOTH Xcode & Linux xtool Builds!)

enum CadenciaMetalLibrary {
    static func loadLibrary(device: MTLDevice) -> MTLLibrary? {
        // 1. Precompiled default.metallib (fastest on Xcode iOS builds!)
        if let mainDefault = device.makeDefaultLibrary() {
            return mainDefault
        }
        #if SWIFT_PACKAGE
        if let modDefault = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return modDefault
        }
        if let url = Bundle.module.url(forResource: "Shaders", withExtension: "metal")
            ?? Bundle.main.url(forResource: "Shaders", withExtension: "metal"),
           let source = try? String(contentsOf: url, encoding: .utf8),
           let compiled = try? device.makeLibrary(source: source, options: nil) {
            return compiled
        }
        #else
        if let url = Bundle.main.url(forResource: "Shaders", withExtension: "metal"),
           let source = try? String(contentsOf: url, encoding: .utf8),
           let compiled = try? device.makeLibrary(source: source, options: nil) {
            return compiled
        }
        #endif
        return nil
    }
}

// MARK: - 1. Living 120Hz Mediterranean Aurora Backdrop

public enum AuroraMood: Float, Sendable {
    case studio = 0
    case warmup = 1
    case input1 = 2
    case input2AudioFirst = 3
    case outputSpeaking = 4
}

#if os(iOS)
public struct MediterraneanAuroraView: UIViewRepresentable {
    public var mood: AuroraMood
    public var audioEnergy: Float

    public init(mood: AuroraMood = .studio, audioEnergy: Float = 0.0) {
        self.mood = mood
        self.audioEnergy = audioEnergy
    }

    public func makeCoordinator() -> AuroraRenderer {
        AuroraRenderer(mood: mood, audioEnergy: audioEnergy)
    }

    public func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.framebufferOnly = true
        view.preferredFramesPerSecond = 120
        view.colorPixelFormat = .bgra8Unorm
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        context.coordinator.attach(to: view)
        view.delegate = context.coordinator
        return view
    }

    public func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.targetMood = mood.rawValue
        context.coordinator.targetEnergy = audioEnergy
    }
}
#else
public struct MediterraneanAuroraView: View {
    public var mood: AuroraMood
    public var audioEnergy: Float
    public init(mood: AuroraMood = .studio, audioEnergy: Float = 0.0) {
        self.mood = mood
        self.audioEnergy = audioEnergy
    }
    public var body: some View {
        LinearGradient(
            colors: [CadenciaPalette.obsidian, CadenciaPalette.midnightIndigo, CadenciaPalette.terracotta.opacity(0.35)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }
}
#endif

@MainActor
public final class AuroraRenderer: NSObject, MTKViewDelegate {
    private struct Uniforms {
        var time: Float
        var audioEnergy: Float
        var moodIndex: Float
        var aspect: Float
    }

    private var commandQueue: MTLCommandQueue?
    private var pipeline: MTLRenderPipelineState?
    private let startTime = CACurrentMediaTime()
    public var targetMood: Float
    public var targetEnergy: Float
    private var smoothedEnergy: Float = 0

    public init(mood: AuroraMood, audioEnergy: Float) {
        self.targetMood = mood.rawValue
        self.targetEnergy = audioEnergy
    }

    public func attach(to view: MTKView) {
        guard let device = view.device, let lib = CadenciaMetalLibrary.loadLibrary(device: device) else { return }
        self.commandQueue = device.makeCommandQueue()
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = lib.makeFunction(name: "fullscreenVertex")
        desc.fragmentFunction = lib.makeFunction(name: "mediterraneanAuroraFragment")
        desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
        self.pipeline = try? device.makeRenderPipelineState(descriptor: desc)
    }

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    public func draw(in view: MTKView) {
        guard let pipeline,
              let queue = commandQueue,
              let passDesc = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: passDesc)
        else { return }

        smoothedEnergy += (targetEnergy - smoothedEnergy) * 0.14
        let aspect = Float(max(1.0, view.drawableSize.width) / max(1.0, view.drawableSize.height))
        var u = Uniforms(
            time: Float(CACurrentMediaTime() - startTime),
            audioEnergy: smoothedEnergy,
            moodIndex: targetMood,
            aspect: aspect
        )

        enc.setRenderPipelineState(pipeline)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }
}

// MARK: - 2. Raymarched Iridescent Acoustic Resonance Orb (120Hz Metal)

public enum ResonanceOrbMode: Float, Sendable {
    case idle = 0
    case listening = 1
    case thinking = 2
    case speaking = 3
}

#if os(iOS)
public struct ResonanceOrbView: UIViewRepresentable {
    public var mode: ResonanceOrbMode
    public var energy: Float

    public init(mode: ResonanceOrbMode, energy: Float) {
        self.mode = mode
        self.energy = energy
    }

    public func makeCoordinator() -> OrbRenderer {
        OrbRenderer(mode: mode, energy: energy)
    }

    public func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.isOpaque = false
        view.backgroundColor = .clear
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        view.preferredFramesPerSecond = 120
        view.colorPixelFormat = .bgra8Unorm
        context.coordinator.attach(to: view)
        view.delegate = context.coordinator
        return view
    }

    public func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.targetMode = mode.rawValue
        context.coordinator.targetEnergy = energy
    }
}
#else
public struct ResonanceOrbView: View {
    public var mode: ResonanceOrbMode
    public var energy: Float
    public init(mode: ResonanceOrbMode, energy: Float) {
        self.mode = mode
        self.energy = energy
    }
    public var body: some View {
        Circle()
            .fill(RadialGradient(colors: [CadenciaPalette.saffronGold, CadenciaPalette.terracotta], center: .center, startRadius: 10, endRadius: 110))
            .scaleEffect(1.0 + CGFloat(energy) * 0.12)
    }
}
#endif

@MainActor
public final class OrbRenderer: NSObject, MTKViewDelegate {
    private struct Uniforms {
        var time: Float
        var energy: Float
        var state: Float
        var aspect: Float
        var bands0: SIMD4<Float>
        var bands1: SIMD4<Float>
    }

    private var commandQueue: MTLCommandQueue?
    private var pipeline: MTLRenderPipelineState?
    private let startTime = CACurrentMediaTime()
    public var targetMode: Float
    public var targetEnergy: Float
    private var smoothedEnergy: Float = 0

    public init(mode: ResonanceOrbMode, energy: Float) {
        self.targetMode = mode.rawValue
        self.targetEnergy = energy
    }

    public func attach(to view: MTKView) {
        guard let device = view.device, let lib = CadenciaMetalLibrary.loadLibrary(device: device) else { return }
        self.commandQueue = device.makeCommandQueue()
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = lib.makeFunction(name: "fullscreenVertex")
        desc.fragmentFunction = lib.makeFunction(name: "resonanceOrbFragment")
        desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        self.pipeline = try? device.makeRenderPipelineState(descriptor: desc)
    }

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    public func draw(in view: MTKView) {
        guard let pipeline,
              let queue = commandQueue,
              let passDesc = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: passDesc)
        else { return }

        smoothedEnergy += (targetEnergy - smoothedEnergy) * 0.18
        let aspect = Float(max(1.0, view.drawableSize.width) / max(1.0, view.drawableSize.height))
        var u = Uniforms(
            time: Float(CACurrentMediaTime() - startTime),
            energy: smoothedEnergy,
            state: targetMode,
            aspect: aspect,
            bands0: .zero,
            bands1: .zero
        )

        enc.setRenderPipelineState(pipeline)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }
}

// MARK: - 3. GPU-Instanced 4,500-Item Memory Constellation

public struct StarPointGPU {
    public var pos: SIMD2<Float>
    public var state: Float
    public var recall: Float
    public var isDue: Float
    public var phase: Float

    public init(pos: SIMD2<Float>, state: Float, recall: Float, isDue: Float, phase: Float) {
        self.pos = pos
        self.state = state
        self.recall = recall
        self.isDue = isDue
        self.phase = phase
    }
}

#if os(iOS)
public struct MemoryConstellationMetalView: UIViewRepresentable {
    public var stars: [StarPointGPU]
    public var zoom: Float
    public var pan: SIMD2<Float>

    public init(stars: [StarPointGPU], zoom: Float = 1.0, pan: SIMD2<Float> = .zero) {
        self.stars = stars
        self.zoom = zoom
        self.pan = pan
    }

    public func makeCoordinator() -> ConstellationRenderer {
        ConstellationRenderer()
    }

    public func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.isOpaque = false
        view.backgroundColor = .clear
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        view.preferredFramesPerSecond = 60
        view.colorPixelFormat = .bgra8Unorm
        context.coordinator.attach(to: view)
        context.coordinator.updateStars(stars)
        view.delegate = context.coordinator
        return view
    }

    public func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.updateStars(stars)
        context.coordinator.zoom = zoom
        context.coordinator.pan = pan
    }
}
#else
public struct MemoryConstellationMetalView: View {
    public var stars: [StarPointGPU]
    public var zoom: Float = 1.0
    public var pan: SIMD2<Float> = .zero
    public init(stars: [StarPointGPU], zoom: Float = 1.0, pan: SIMD2<Float> = .zero) {
        self.stars = stars; self.zoom = zoom; self.pan = pan
    }
    public var body: some View {
        Canvas { ctx, size in
            for s in stars {
                let x = (CGFloat(s.pos.x) * 0.5 + 0.5) * size.width
                let y = (0.5 - CGFloat(s.pos.y) * 0.5) * size.height
                let col: Color = s.state >= 3 ? CadenciaPalette.saffronGold : (s.state >= 2 ? CadenciaPalette.bioluminescentCobalt : (s.state >= 1 ? CadenciaPalette.terracotta : .white.opacity(0.25)))
                ctx.fill(Path(ellipseIn: CGRect(x: x - 1.5, y: y - 1.5, width: 3, height: 3)), with: .color(col))
            }
        }
    }
}
#endif

@MainActor
public final class ConstellationRenderer: NSObject, MTKViewDelegate {
    private struct Uniforms {
        var time: Float
        var zoom: Float
        var pan: SIMD2<Float>
        var aspect: Float
        var pad: Float
    }

    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var pipeline: MTLRenderPipelineState?
    private var vertexBuffer: MTLBuffer?
    private var starCount: Int = 0
    private let startTime = CACurrentMediaTime()
    public var zoom: Float = 1.0
    public var pan: SIMD2<Float> = .zero

    public func attach(to view: MTKView) {
        guard let dev = view.device, let lib = CadenciaMetalLibrary.loadLibrary(device: dev) else { return }
        self.device = dev
        self.commandQueue = dev.makeCommandQueue()
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = lib.makeFunction(name: "memoryStarVertex")
        desc.fragmentFunction = lib.makeFunction(name: "memoryStarFragment")
        desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .one
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .one
        self.pipeline = try? dev.makeRenderPipelineState(descriptor: desc)
    }

    public func updateStars(_ stars: [StarPointGPU]) {
        guard let device, !stars.isEmpty else { starCount = 0; return }
        let byteLen = stars.count * MemoryLayout<StarPointGPU>.stride
        vertexBuffer = device.makeBuffer(bytes: stars, length: byteLen, options: .storageModeShared)
        starCount = stars.count
    }

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    public func draw(in view: MTKView) {
        guard let pipeline,
              let queue = commandQueue,
              let vbuf = vertexBuffer,
              starCount > 0,
              let passDesc = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: passDesc)
        else { return }

        let aspect = Float(max(1.0, view.drawableSize.width) / max(1.0, view.drawableSize.height))
        var u = Uniforms(
            time: Float(CACurrentMediaTime() - startTime),
            zoom: zoom,
            pan: pan,
            aspect: aspect,
            pad: 0
        )

        enc.setRenderPipelineState(pipeline)
        enc.setVertexBuffer(vbuf, offset: 0, index: 0)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: starCount)
        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }
}
