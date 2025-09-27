# ADR: CUDA Architecture Target Reduction

## Status

Proposed

Proposed by: Adam Gibson (September 2025)

Discussed with: Development Team

## Context

LibND4J's CUDA builds have grown increasingly complex as we've tried to support a wide range of GPU architectures. Currently, we compile for nine different compute capabilities (5.0, 5.2, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, and 9.0), which creates several challenges:

Build times have become a significant bottleneck in our development workflow. A full CUDA build now takes over 4 hours, with CI/CD pipelines frequently timing out. This impacts developer productivity and slows our release cycle considerably.

The binary size issue is equally concerning. Our CUDA libraries approach 800MB per platform, and Docker images exceed 4GB. We're hitting size limits for Maven artifacts, which complicates distribution and deployment. Mobile and edge deployments, an increasingly important use case, struggle with these large binaries.

Our analysis of actual GPU usage patterns reveals that the majority of our users are on modern hardware. The older Maxwell (GTX 900 series) and Pascal (GTX 1000 series) architectures show declining usage, while Ampere and newer architectures dominate. Supporting legacy hardware is creating an outsized maintenance burden for diminishing returns.

## Decision

We will reduce our CUDA architecture targets to focus exclusively on modern GPUs, specifically compute capabilities 8.6 and 9.0. This means dropping support for all architectures older than Ampere.

The technical implementation is straightforward - we'll modify our CMake configuration from:
```cmake
-DCOMPUTE_CAPABILITIES="5.0 5.2 6.0 6.1 7.0 7.5 8.0 8.6 9.0"
```

To:
```cmake
-DCOMPUTE_CAPABILITIES="8.6 9.0"
```

This change will support:
- Compute capability 8.6: RTX 3050-3090, RTX A2000-A6000, A40, A100
- Compute capability 9.0: RTX 4060-4090, L4, L40, H100

We will no longer support:
- Maxwell (5.0-5.2): GTX 900 series
- Pascal (6.0-6.1): GTX 1000 series  
- Volta/Turing (7.0-7.5): RTX 2000 series
- Older Ampere (8.0): Original A100 revision

## Implementation Strategy

We recognize this is a breaking change that requires careful communication and migration planning.

### Communication Timeline
We'll announce the change 3 months before the next major release, providing:
- Clear documentation of minimum hardware requirements
- A comprehensive compatibility matrix
- Migration guides for affected users

### Legacy Support
The 1.0.0-M3 release will be our last version with full architecture support. We'll maintain this release with security patches only, providing a stable option for users who cannot upgrade their hardware. We'll also document how users can build from source if they need support for specific legacy architectures.

### Alternative Options
For users with older GPUs, we'll recommend:
- Using CPU inference as a fallback
- Community-maintained builds for legacy architectures
- Docker images with specific architecture targets
- Cloud-based solutions for training and inference

## Consequences

### Advantages

The build time improvements are dramatic - we expect a 75% reduction:
- Full builds: 4+ hours → ~90 minutes
- Incremental builds: 45 minutes → 15 minutes
- CI/CD pipeline success rates will improve significantly

Binary size reductions are equally impressive:
- CUDA libraries: ~800MB → ~200MB per platform
- Docker images: 4GB → 1.5GB
- Maven artifacts will comfortably fit within size limits

From a development perspective, this simplifies our testing matrix considerably. We can focus our optimization efforts on modern hardware features, leading to better performance for the majority of users. The reduced CI/CD costs and faster developer iteration cycles will accelerate our development velocity.

### Disadvantages

The primary drawback is obvious - users with older GPUs cannot use newer versions of LibND4J. This potentially reduces our user base and may particularly impact:
- Academic institutions with older hardware
- Developing regions where newer GPUs are less accessible
- Legacy enterprise systems that can't be easily upgraded

There's no gradual migration path - it's a hard cutoff. Users must either upgrade their hardware or stay on older software versions.

## References

- NVIDIA CUDA Compatibility Guide
- LibND4J build performance metrics
- User hardware survey results (2025)
- CI/CD cost analysis reports