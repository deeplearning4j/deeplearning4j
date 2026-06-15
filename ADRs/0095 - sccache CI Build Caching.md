# ADR 0095 - sccache CI Build Caching

## Status
Implemented

Proposed by: Adam Gibson (May 2026)

## Context

All CI build workflows previously used ccache with `actions/cache` to save/restore a bulk directory snapshot. This approach had several significant problems:

1. **Bulk cache eviction**: The entire ccache directory (~2 GB) was uploaded/downloaded as a single tar archive per run, with 7-day eviction on the whole directory — a single stale entry could not be evicted independently.
2. **Cross-variant cache misses**: Matrix variants (e.g., `cudnn`/`compile`/base) each maintained separate caches, causing eviction pressure and preventing shared hits.
3. **CUDA wrapper complexity**: A custom `nvcc_filter.py` wrapper and `SmartCcache` CMake module were required to handle nvcc response-file expansion on Windows.
4. **PCH invalidation**: Precompiled headers caused high miss rates across platforms due to mtime mismatches, requiring extensive diagnostics and workarounds.

The project builds 10 separate CI workflows across Linux x86_64, Linux ARM64, macOS ARM64, Windows x86_64, and Android (ARM64 + x86_64), each with CPU and/or CUDA variants. Build times range from 20 minutes (CPU-only) to 90+ minutes (CUDA with Triton), making cache hit rates critical.

## Decision

Replace ccache with Mozilla sccache v0.10.0 across all 10 CI build workflows, using the **GitHub Actions Cache API** as the storage backend.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   GitHub Actions Runner                  │
│                                                         │
│  ┌───────────────────┐    ┌──────────────────────────┐ │
│  │  CMake Build       │    │  sccache daemon          │ │
│  │                    │    │                          │ │
│  │  CMAKE_C_COMPILER  │───>│  Per-object cache keys   │ │
│  │  _LAUNCHER=sccache │    │  Content-addressed hash  │ │
│  │                    │    │                          │ │
│  │  CMAKE_CUDA_       │───>│  Native nvcc support     │ │
│  │  COMPILER_LAUNCHER │    │  (no wrapper scripts)    │ │
│  │  =sccache          │    │                          │ │
│  └───────────────────┘    └──────────┬───────────────┘ │
│                                      │                  │
│                           ┌──────────▼───────────────┐ │
│                           │  GitHub Actions Cache API │ │
│                           │  (ACTIONS_RESULTS_URL)    │ │
│                           │                          │ │
│                           │  Per-object storage      │ │
│                           │  Independent eviction    │ │
│                           │  Cross-job sharing       │ │
│                           └──────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Per-Platform Composite Actions

Three composite actions were added under `.github/actions/`:

| Action | Platforms | Installation Method |
|---|---|---|
| `setup-sccache-linux` | Linux x86_64, Linux ARM64 | Downloads musl binary (v0.10.0) |
| `setup-sccache-macos` | macOS ARM64 | Homebrew (`brew install sccache`) |
| `setup-sccache-windows` | Windows x86_64 | Downloads MSVC binary |

### Configuration

- **Backend**: `SCCACHE_GHA_ENABLED=true` with `ACTIONS_RESULTS_URL` and `ACTIONS_RUNTIME_TOKEN` exposed via `actions/github-script`
- **Cache partitioning**: `SCCACHE_GHA_VERSION` is set from a `cache-suffix` input (e.g., `cuda-12.9`, `arm64`, `cpu`) to prevent cross-platform cache contamination
- **CMake integration**: `CMAKE_C_COMPILER_LAUNCHER`, `CMAKE_CXX_COMPILER_LAUNCHER`, and `CMAKE_CUDA_COMPILER_LAUNCHER` all set to `sccache`. `SD_USE_SCCACHE=1` tells `CMakeLists.txt` to skip ccache-specific setup (nvcc_filter.py, SmartCcache module)
- **PCH disabled**: `SD_DISABLE_PCH=1` is always set to avoid mtime-based invalidation

### Workflows Migrated

All 10 build workflows were migrated in two commits:

1. Linux x86_64 CUDA 12.6 and 12.9, Windows CUDA 12.6 and 12.9
2. Linux x86_64 CPU, Linux ARM64, macOS ARM64, Windows CPU, Android ARM64, Android x86_64

### CMake Dependency Cache (Orthogonal)

A separate CMake-level cache for ExternalProject dependencies (FlatBuffers, OneDNN) lives at `~/.libnd4j/dep-cache` (configurable via `-DSD_DEP_CACHE_DIR`). This is opt-in (`-DSD_DEP_CACHE=ON`) and orthogonal to the compiler caching — it caches downloaded/built dependency artifacts, not compiled object files.

## Consequences

### Positive

- **Independent eviction**: Each compiled object is cached individually, eliminating bulk invalidation
- **Cross-variant sharing**: All matrix variants (`cudnn`, `compile`, base) share cache hits automatically
- **Native CUDA support**: sccache handles nvcc natively — no `nvcc_filter.py` wrapper needed
- **Simpler configuration**: One tool replaces ccache + SmartCcache + nvcc_filter.py
- **Faster cache restore**: No 2 GB tar download at job start

### Negative

- **Vendor lock-in**: GitHub Actions Cache API is GitHub-specific (though sccache supports S3, GCS, Azure as alternative backends)
- **Local dev divergence**: Developer machines still use ccache locally (sccache is CI-only)

## Related ADRs

- None (first CI caching ADR)
