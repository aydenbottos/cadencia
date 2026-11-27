// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Cadencia",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(
            name: "CadenciaCore",
            targets: ["CadenciaCore"]
        ),
        .library(
            name: "CadenciaSpeech",
            targets: ["CadenciaSpeech"]
        ),
        .library(
            name: "CadenciaApp",
            targets: ["CadenciaApp"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.18.4"),
    ],
    targets: [
        .target(
            name: "CadenciaCore",
            dependencies: [],
            path: "Sources/CadenciaCore",
            resources: [
                .process("Resources"),
            ],
            linkerSettings: [
                .linkedLibrary("sqlite3"),
            ]
        ),
        .target(
            name: "CadenciaSpeech",
            dependencies: [
                "CadenciaCore",
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources/CadenciaSpeech"
        ),
        .target(
            name: "CadenciaApp",
            dependencies: [
                "CadenciaCore",
                "CadenciaSpeech",
            ],
            path: "Sources/CadenciaApp",
            resources: [
                .process("Resources"),
            ]
        ),
        .testTarget(
            name: "CadenciaCoreTests",
            dependencies: [
                "CadenciaCore",
                "CadenciaSpeech",
            ],
            path: "Tests/CadenciaCoreTests"
        ),
    ]
)
