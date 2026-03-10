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
        .target(
            name: "SdxRuntime",
            path: "Sources/SdxRuntime"
        )
    ]
)
