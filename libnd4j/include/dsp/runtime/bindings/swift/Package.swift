// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SdxRuntime",
    platforms: [
        .macOS(.v13),
        .iOS(.v15)
    ],
    products: [
        // DSP general inference runtime (dsp_runtime_c.h)
        .library(name: "SdxRuntime", targets: ["SdxRuntime"]),
        // LLM/VLM/STT AOT runtime (sdx_llm_c.h)
        .library(name: "SdxLlm",     targets: ["SdxLlm"]),
    ],
    targets: [
        // ── DSP general runtime ──────────────────────────────────────────────
        .systemLibrary(
            name: "CSdxRuntime",
            path: "Sources/CSdxRuntime"
        ),
        .target(
            name: "SdxRuntime",
            dependencies: ["CSdxRuntime"],
            path: "Sources/SdxRuntime"
        ),

        // ── LLM / VLM / STT AOT runtime (libsdx_llm.so) ─────────────────────
        // Requires -Xcc -I<sdk>/include and -Xlinker -L<sdk>/lib at build time.
        // See Sources/CSdxLlm/shim.h for header resolution details.
        .systemLibrary(
            name: "CSdxLlm",
            path: "Sources/CSdxLlm"
        ),
        .target(
            name: "SdxLlm",
            dependencies: ["CSdxLlm"],
            path: "Sources/SdxLlm"
        ),
    ]
)
