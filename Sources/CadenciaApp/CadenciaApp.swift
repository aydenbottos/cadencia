import CadenciaCore
import CadenciaSpeech
import SwiftUI

public enum CadenciaTab: Int, CaseIterable, Identifiable {
    case studio = 0
    case memory = 1
    case lab = 2
    case system = 3

    public var id: Int { rawValue }

    public var title: String {
        switch self {
        case .studio: return "Studio"
        case .memory: return "Memory"
        case .lab: return "Lab"
        case .system: return "Settings"
        }
    }

    public var icon: String {
        switch self {
        case .studio: return "sparkles"
        case .memory: return "circle.hexagongrid.fill"
        case .lab: return "waveform.path.ecg"
        case .system: return "cpu.fill"
        }
    }
}

@main
public struct CadenciaApp: App {
    @State private var appState = AppState()
    @State private var selectedTab: CadenciaTab = .studio

    public init() {}

    public var body: some Scene {
        WindowGroup {
            rootContainer
                .preferredColorScheme(.dark)
                .task {
                    await appState.bootstrap()
                }
        }
    }

    @ViewBuilder
    private var rootContainer: some View {
        ZStack {
            // Living 120Hz Metal Mediterranean Aurora Backdrop
            MediterraneanAuroraView(
                mood: selectedTab == .memory ? .input2AudioFirst : .studio,
                audioEnergy: appState.liveAudioEnergy
            )
            .ignoresSafeArea()

            if appState.isBootstrapping {
                bootSplashView
            } else if appState.profile == nil {
                OnboardingVaultView(appState: appState)
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
            } else {
                mainTabbedShell
            }
        }
        #if os(iOS)
        .fullScreenCover(
            isPresented: Binding(
                get: { appState.activeSessionController != nil },
                set: { if !$0 { appState.dismissActiveSession() } }
            )
        ) {
            if let controller = appState.activeSessionController {
                SessionOrchestratorView(controller: controller) {
                    appState.dismissActiveSession()
                }
            }
        }
        #else
        .sheet(
            isPresented: Binding(
                get: { appState.activeSessionController != nil },
                set: { if !$0 { appState.dismissActiveSession() } }
            )
        ) {
            if let controller = appState.activeSessionController {
                SessionOrchestratorView(controller: controller) {
                    appState.dismissActiveSession()
                }
                .frame(minWidth: 720, minHeight: 820)
            }
        }
        #endif
    }

    private var bootSplashView: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [CadenciaPalette.saffronGold.opacity(0.85), CadenciaPalette.terracotta.opacity(0.45), .clear],
                            center: .center,
                            startRadius: 8,
                            endRadius: 56
                        )
                    )
                    .frame(width: 112, height: 112)
                Image(systemName: "waveform.and.magnifyingglass")
                    .font(.system(size: 32, weight: .semibold))
                    .foregroundStyle(CadenciaPalette.ivoryText)
            }
            Text("Cadencia")
                .font(CadenciaFont.displaySerif(38, weight: .bold))
                .foregroundStyle(CadenciaPalette.ivoryText)
            ProgressView()
                .tint(CadenciaPalette.saffronGold)
        }
    }

    private var mainTabbedShell: some View {
        Group {
            switch selectedTab {
            case .studio:
                HomeStudioView(
                    appState: appState,
                    selectedTab: Binding(
                        get: { selectedTab.rawValue },
                        set: { selectedTab = CadenciaTab(rawValue: $0) ?? .studio }
                    )
                )
            case .memory:
                MemoryAtlasView(appState: appState)
            case .lab:
                ResearchLabView(appState: appState)
            case .system:
                HardwareSettingsView(appState: appState)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // Reserves dedicated bottom safe area so scrollable lists (especially Memory Ledger)
        // never overlap or steal touches from the floating glass dock!
        .safeAreaInset(edge: .bottom, spacing: 0) {
            floatingLiquidGlassTabBar
                .padding(.horizontal, 16)
                .padding(.top, 8)
                .padding(.bottom, 6)
                .background(
                    LinearGradient(
                        colors: [Color.clear, CadenciaPalette.obsidian.opacity(0.72), CadenciaPalette.obsidian.opacity(0.92)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                    .ignoresSafeArea(edges: .bottom)
                    .allowsHitTesting(false)
                )
        }
    }

    private var floatingLiquidGlassTabBar: some View {
        HStack(spacing: 4) {
            ForEach(CadenciaTab.allCases) { tab in
                let active = selectedTab == tab
                Button {
                    withAnimation(.spring(response: 0.28, dampingFraction: 0.85)) {
                        selectedTab = tab
                    }
                    HapticDirector.shared.playGlossTap()
                } label: {
                    VStack(spacing: 3) {
                        Image(systemName: tab.icon)
                            .font(.system(size: 17, weight: active ? .bold : .semibold))
                        Text(tab.title)
                            .font(CadenciaFont.uiCaption(10, weight: active ? .bold : .medium))
                            .lineLimit(1)
                    }
                    .foregroundStyle(active ? CadenciaPalette.obsidian : CadenciaPalette.ivoryText.opacity(0.88))
                    .frame(maxWidth: .infinity)
                    .frame(minHeight: 50)
                    .background {
                        if active {
                            Capsule()
                                .fill(
                                    LinearGradient(
                                        colors: [CadenciaPalette.saffronGold, CadenciaPalette.terracotta],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                        }
                    }
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(5)
        // Outer dock uses non-interactive Liquid Glass so iOS 26 never intercepts taps meant for the 4 tab buttons!
        .cadenciaGlassPill(interactive: false)
        .contentShape(Capsule())
    }
}
