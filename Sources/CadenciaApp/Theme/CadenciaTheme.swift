import CadenciaCore
import SwiftUI

// MARK: - Editorial Luxury Color Palette ("Mediterranean Nocturne & Sunlit Terracotta")

public enum CadenciaPalette {
    // Base surfaces
    public static let obsidian = Color(red: 0.055, green: 0.062, blue: 0.088)
    public static let midnightIndigo = Color(red: 0.078, green: 0.098, blue: 0.172)
    public static let deepSlate = Color(red: 0.115, green: 0.128, blue: 0.175)
    public static let warmAlabaster = Color(red: 0.968, green: 0.953, blue: 0.925)
    public static let parchmentCream = Color(red: 0.935, green: 0.908, blue: 0.862)
    public static let inkCharcoal = Color(red: 0.122, green: 0.114, blue: 0.106)

    // Luminous accents
    public static let terracotta = Color(red: 0.845, green: 0.365, blue: 0.235)
    public static let saffronGold = Color(red: 0.945, green: 0.725, blue: 0.320)
    public static let bioluminescentCobalt = Color(red: 0.255, green: 0.620, blue: 0.985)
    public static let mediterraneanJade = Color(red: 0.215, green: 0.765, blue: 0.595)
    public static let crimsonAlert = Color(red: 0.910, green: 0.285, blue: 0.320)
    public static let ivoryText = Color(red: 0.965, green: 0.955, blue: 0.935)
    public static let mutedSilver = Color(red: 0.685, green: 0.705, blue: 0.755)

    public static func color(for state: MasteryState) -> Color {
        switch state {
        case .new: return mutedSilver.opacity(0.45)
        case .learning: return terracotta
        case .known: return bioluminescentCobalt
        case .productive: return saffronGold
        }
    }

    public static func color(for block: SessionBlockType) -> Color {
        switch block {
        case .warmup: return saffronGold
        case .input1: return terracotta
        case .output1: return mediterraneanJade
        case .input2: return bioluminescentCobalt
        case .output2: return saffronGold
        case .closing: return mediterraneanJade
        }
    }
}

// MARK: - Editorial Typography

public enum CadenciaFont {
    public static func displaySerif(_ size: CGFloat, weight: Font.Weight = .semibold) -> Font {
        .system(size: size, weight: weight, design: .serif)
    }

    public static func proseSerif(_ size: CGFloat = 21) -> Font {
        .system(size: size, weight: .regular, design: .serif)
    }

    public static func uiTitle(_ size: CGFloat = 18, weight: Font.Weight = .semibold) -> Font {
        .system(size: size, weight: weight, design: .default)
    }

    public static func uiCaption(_ size: CGFloat = 12, weight: Font.Weight = .medium) -> Font {
        .system(size: size, weight: weight, design: .default)
    }

    public static func mono(_ size: CGFloat = 13, weight: Font.Weight = .medium) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

// MARK: - iOS 26/27 Liquid Glass Compatibility Wrappers

public struct CadenciaGlassContainer<Content: View>: View {
    private let spacing: CGFloat
    private let content: Content

    public init(spacing: CGFloat = 16, @ViewBuilder content: () -> Content) {
        self.spacing = spacing
        self.content = content()
    }

    public var body: some View {
        #if os(iOS)
        if #available(iOS 26.0, *) {
            GlassEffectContainer(spacing: spacing) {
                content
            }
        } else {
            content
        }
        #else
        content
        #endif
    }
}

public extension View {
    @ViewBuilder
    func cadenciaGlassCard(cornerRadius: CGFloat = 24, tint: Color? = nil, interactive: Bool = false) -> some View {
        #if os(iOS)
        if #available(iOS 26.0, *) {
            let base: Glass = {
                var g = Glass.regular
                if let tint { g = g.tint(tint.opacity(0.22)) }
                if interactive { g = g.interactive() }
                return g
            }()
            self.glassEffect(base, in: .rect(cornerRadius: cornerRadius))
        } else {
            self
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .strokeBorder(
                            LinearGradient(
                                colors: [Color.white.opacity(0.28), (tint ?? Color.white).opacity(0.08)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                )
        }
        #else
        self
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.18), lineWidth: 1)
            )
        #endif
    }

    @ViewBuilder
    func cadenciaGlassPill(tint: Color? = nil, interactive: Bool = true) -> some View {
        #if os(iOS)
        if #available(iOS 26.0, *) {
            let base: Glass = {
                var g = Glass.regular
                if let tint { g = g.tint(tint.opacity(0.28)) }
                if interactive { g = g.interactive() }
                return g
            }()
            self.glassEffect(base, in: .capsule)
        } else {
            self
                .background(.ultraThinMaterial, in: Capsule())
                .overlay(
                    Capsule()
                        .strokeBorder(
                            LinearGradient(
                                colors: [Color.white.opacity(0.32), (tint ?? Color.white).opacity(0.12)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                )
        }
        #else
        self
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(Color.white.opacity(0.22), lineWidth: 1))
        #endif
    }
}
