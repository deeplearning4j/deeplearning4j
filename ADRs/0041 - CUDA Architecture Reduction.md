# ADR-0041: CUDA Architecture Target Reduction

## Status

Proposed

Proposed by: Adam Gibson (27-09-2025)

## Context

LibND4J CUDA builds target multiple GPU architectures through compute capabilities.
Current builds compile for architectures: 5.0, 5.2, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 9.0.
Each architecture adds significant compilation time and binary size.
Build times exceed 4 hours for full CUDA builds.
Binary artifacts approach size limits for distribution.

Analysis shows most users run modern GPUs (Ampere/Ada Lovelace).
Older architectures (Maxwell/Pascal) have declining usage.
Supporting all architectures creates maintenance burden.
CI/CD pipeline timeouts occur frequently.

## Decision

Reduce CUDA architecture targets to modern GPUs only.
Target compute capabilities 8.6 and 9.0 exclusively.
Drop support for architectures older than Ampere.
Implement clear communication about hardware requirements.

### Technical Changes

**CMake Configuration**
```cmake
# Before
-DCOMPUTE_CAPABILITIES="5.0 5.2 6.0 6.1 7.0 7.5 8.0 8.6 9.0"

# After  
-DCOMPUTE_CAPABILITIES="8.6 9.0"
```

**Supported GPUs After Change**
- 8.6: RTX 3050-3090, RTX A2000-A6000, A40, A100
- 9.0: RTX 4060-4090, L4, L40, H100

**Dropped GPU Support**
- 5.0-5.2: Maxwell (GTX 900 series)
- 6.0-6.1: Pascal (GTX 1000 series)
- 7.0-7.5: Volta/Turing (RTX 2000 series)
- 8.0: Ampere A100 (older revision)

### Build Impact

Compilation time reduction:
- Full build: 4+ hours → ~90 minutes
- Incremental build: 45 minutes → 15 minutes
- CI pipeline success rate improvement

Binary size reduction:
- CUDA libraries: ~800MB → ~200MB per platform
- Docker images: 4GB → 1.5GB
- Maven artifacts within size limits

## Consequences

### Advantages
- 75% faster build times
- 75% smaller binary size
- Reduced CI/CD costs
- Faster developer iteration
- Simplified testing matrix
- Focus on modern hardware

### Disadvantages  
- Users with older GPUs cannot upgrade
- Potential user base reduction
- Legacy system incompatibility
- Enterprise users may have older hardware
- No gradual migration path

### Migration Strategy

1. **Communication Plan**
   - Announce change 3 months before release
   - Document minimum hardware requirements
   - Provide compatibility matrix

2. **Legacy Support**
   - Maintain 1.0.0-M3 with full architecture support
   - Security patches only for legacy version
   - Document self-build instructions

3. **Alternative Options**
   - CPU fallback for older systems
   - Community builds for legacy architectures
   - Docker images with specific targets

### Future Considerations

Architecture support policy:
- Support current and previous GPU generation
- Re-evaluate every 2 years
- Consider separate legacy builds
- Monitor usage statistics

Build optimization opportunities:
- Just-in-time compilation exploration
- Architecture-specific optimization
- Reduced kernel variants

## References
- NVIDIA CUDA Compatibility Guide
- LibND4J build statistics
- User hardware surveys
- CI/CD performance metrics