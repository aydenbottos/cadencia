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
                set: { if !$0 { Task { await appState.finishActiveSession() } } }
            )
        ) {
            if let controller = appState.activeSessionController {
                SessionOrchestratorView(controller: controller) {
                    Task { await appState.finishActiveSession() }
                }
            }
        }
        #else
        .sheet(
            isPresented: Binding(
                get: { appState.activeSessionController != nil },
                set: { if !$0 { Task { await appState.finishActiveSession() } } }
            )
        ) {
            if let controller = appState.activeSessionController {
                SessionOrchestratorView(controller: controller) {
                    Task { await appState.finishActiveSession() }
                }
                .frame(minWidth: 720, minHeight: 820)
            }
        }
        #endif
    }

    private var bootSplashView: some View {
        VStack(spacing: 18) {
            ResonanceOrbView(mode: .thinking, energy: 0.35)
                .frame(width: 150, height: 150)
            Text("Cadencia")
                .font(CadenciaFont.displaySerif(38, weight: .bold))
                .foregroundStyle(CadenciaPalette.ivoryText)
            Text("Loading 4,900-item linguistic ledger & Core ML engines…")
                .font(CadenciaFont.mono(12))
                .foregroundStyle(CadenciaPalette.mutedSilver)
        }
    }

    private var mainTabbedShell: some View {
        ZStack(alignment: .bottom) {
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

            floatingLiquidGlassTabBar
                .padding(.horizontal, 18)
                .padding(.bottom, 10)
        }
    }

    private var floatingLiquidGlassTabBar: some View {
        CadenciaGlassContainer(spacing: 8) {
            HStack(spacing: 6) {
                ForEach(CadenciaTab.allCases) { tab in
                    let active = selectedTab == tab
                    Button {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
                            selectedTab = tab
                        }
                        HapticDirector.shared.playGlossTap()
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: tab.icon)
                                .font(.system(size: 15, weight: .semibold))
                            if active {
                                Text(tab.title)
                                    .font(CadenciaFont.uiCaption(13, weight: .bold))
                                    .lineLimit(1)
                            }
                        }
                        .foregroundStyle(active ? CadenciaPalette.obsidian : CadenciaPalette.ivoryText.opacity(0.85))
                        .padding(.horizontal, active ? 16 : 14)
                        .padding(.vertical, 12)
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
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(6)
            .cadenciaGlassPill(interactive: true)
        }
    }
}
