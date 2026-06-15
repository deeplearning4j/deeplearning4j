// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SdxRuntime",
    platforms: [
        .macOS(.v13),
        .iOS(.v15)
    ],
    products: [
        .library(name: "SdxRuntime", targets: ["SdxRuntime"])
    ],
    targets: [
        .systemLibrary(
            name: "CSdxRuntime",
            path: "Sources/CSdxRuntime"
        ),
        .target(
            name: "SdxRuntime",
            dependencies: ["CSdxRuntime"],
            path: "Sources/SdxRuntime"
        )
    ]
)
