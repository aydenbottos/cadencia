import Foundation
#if os(iOS)
import CoreHaptics
import UIKit
#endif

/// Bespoke CoreHaptics tactile choreographer providing subtle, memorable physical signatures
/// for every learning micro-event (tap-to-gloss, strong success, promotion to Known/Productive,
/// sentence replay doubt pulse, and live voice orb activation).
@MainActor
public final class HapticDirector {
    public static let shared = HapticDirector()

    #if os(iOS)
    private var engine: CHHapticEngine?
    #endif

    private init() {
        prepareEngine()
    }

    public func prepareEngine() {
        #if os(iOS)
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else { return }
        do {
            let eng = try CHHapticEngine()
            eng.playsHapticsOnly = true
            eng.resetHandler = { [weak self] in
                Task { @MainActor in self?.prepareEngine() }
            }
            try eng.start()
            self.engine = eng
        } catch {
            self.engine = nil
        }
        #endif
    }

    /// Crisp tactile tick when tapping a word for its gloss.
    public func playGlossTap() {
        #if os(iOS)
        playTransient(intensity: 0.55, sharpness: 0.78)
        #endif
    }

    /// Warm double-thump resonance on Strong Success (x2.2 half-life jump).
    public func playStrongSuccess() {
        #if os(iOS)
        playPattern([
            (0.00, 0.65, 0.45),
            (0.09, 0.92, 0.72),
        ])
        #endif
    }

    /// Gentle silk pulse when flagging Doubt (sentence replay / slow read).
    public func playDoubtPulse() {
        #if os(iOS)
        playTransient(intensity: 0.40, sharpness: 0.25)
        #endif
    }

    /// Soft warning double-tap when a probe answer is wrong (x0.5 half-life).
    public func playFailure() {
        #if os(iOS)
        let gen = UINotificationFeedbackGenerator()
        gen.notificationOccurred(.warning)
        #endif
    }

    /// Crystalline ascending 3-beat cascade when an item promotes to Known or Productive!
    public func playStatePromotion() {
        #if os(iOS)
        playPattern([
            (0.00, 0.50, 0.60),
            (0.07, 0.72, 0.78),
            (0.15, 1.00, 0.95),
        ])
        #endif
    }

    /// Subtle inhale click when activating the live voice orb.
    public func playMicStart() {
        #if os(iOS)
        playTransient(intensity: 0.75, sharpness: 0.65)
        #endif
    }

    public func playMicStop() {
        #if os(iOS)
        playTransient(intensity: 0.48, sharpness: 0.35)
        #endif
    }

    #if os(iOS)
    private func playTransient(intensity: Float, sharpness: Float) {
        guard let engine else {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            return
        }
        do {
            let ev = CHHapticEvent(
                eventType: .hapticTransient,
                parameters: [
                    CHHapticEventParameter(parameterID: .hapticIntensity, value: intensity),
                    CHHapticEventParameter(parameterID: .hapticSharpness, value: sharpness),
                ],
                relativeTime: 0
            )
            let pattern = try CHHapticPattern(events: [ev], parameters: [])
            let player = try engine.makePlayer(withPattern: pattern)
            try player.start(atTime: CHHapticTimeImmediate)
        } catch {
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        }
    }

    private func playPattern(_ beats: [(Double, Float, Float)]) {
        guard let engine else {
            UINotificationFeedbackGenerator().notificationOccurred(.success)
            return
        }
        do {
            let events = beats.map { t, intensity, sharpness in
                CHHapticEvent(
                    eventType: .hapticTransient,
                    parameters: [
                        CHHapticEventParameter(parameterID: .hapticIntensity, value: intensity),
                        CHHapticEventParameter(parameterID: .hapticSharpness, value: sharpness),
                    ],
                    relativeTime: t
                )
            }
            let pattern = try CHHapticPattern(events: events, parameters: [])
            let player = try engine.makePlayer(withPattern: pattern)
            try player.start(atTime: CHHapticTimeImmediate)
        } catch {
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        }
    }
    #endif
}
