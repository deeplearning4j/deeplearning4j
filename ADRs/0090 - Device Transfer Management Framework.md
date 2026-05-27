# ADR: Device Transfer Management Framework

**Date:** 2026-04-03  
**Status:** Proposed  
**Author:** ND4J Team

## Context

CUDA graph replay bakes GPU memory addresses into captured graphs. Currently, to prevent memory leaks and pointer invalidation, buffers are pinned to specific devices via `_frozenRefCount` in `DataBuffer`. On non-P2P multi-GPU setups, this means host-staged copies (GPU→Host→GPU).

The current system has several gaps:
1. No per-variable control over device placement
2. No diagnostics for when/why transfers happen
3. No leak detection for replicated arrays
4. No mechanism to gracefully recreate a plan when GPU memory distribution becomes untenable
5. No pointer stability validation for graph replay

## Decision

Implement a comprehensive Device Transfer Management Framework with five priority levels:

### P1: Per-Variable Device Pinning
- New `DevicePinPolicy` enum: STICKY, FOLLOW_THREAD, EXPLICIT
- `DevicePinningManager` singleton for pin storage and resolution
- Integration with migration logic to respect pin policies
- Frozen buffer checks to block migration when `_frozenRefCount > 0`

### P2: Transfer Diagnostics
- New `TransferSubsystem` with ring buffer (4096 events)
- Zero-overhead when disabled (single boolean check)
- Per-variable aggregated statistics
- C++ DSP_DIAG integration for H2D, D2H, D2D transfers
- Access via `Nd4j.framework.device().transfers()`

### P3: Replica Leak Detection
- Track replicated arrays across devices
- Detect leaks when replicas aren't properly cleaned up
- Integration with existing `LeakDetector`
- Access via `Nd4j.framework.device().replicaLeaks()`

### P4: Pointer Stability Guarantees
- Track GPU buffer addresses for graph replay validation
- Validate addresses remain stable across capture/replay
- Frozen buffer checks before migration attempts
- Access via `Nd4j.framework.device().pointerStability()`

### P5: Plan Destruction and Recreation (Last Priority)
- Detect memory pressure on execution device
- Trigger plan recreation with new device distribution
- Clear replica caches and reset execution state

## Implementation

### New Java Files (11)
- `TransferDirection.java`, `TransferReason.java`, `TransferEvent.java`
- `TransferStats.java`, `TransferReport.java`
- `DevicePinPolicy.java`, `DevicePinning.java`
- `TransferSubsystem.java`, `DevicePinningManager.java`
- `ReplicaLeakDetector.java`, `PointerStabilityGuard.java`

### Modified Java Files (6)
- `ND4JSystemProperties.java` - 4 new system properties
- `DeviceState.java` - transfer/pinning/frozen count fields
- `DeviceSubsystem.java` - new accessors
- `Framework.java` - include transfer stats in state capture
- `LeakDetector.java` - integrate replica leak detection
- `PotentialLeak.java` - LEAKED_REPLICA reason

### Modified C++ Files (3)
- `DspDiagnostics.h` - DSP_DIAG_TRANSFER category (13 total)
- `DspDiagnostics.cpp` - "TRANSFER" category name
- `DataBuffer.cu` - DSP_DIAG calls in sync/migrate methods

## System Properties

```
-Dnd4j.device.transfer.tracking=true      # Enable transfer tracking
-Dnd4j.device.pinning.enabled=true        # Enable device pinning
-Dnd4j.device.replica.leak.detection=true # Enable replica leak detection
-Dnd4j.device.pointerStability.check=true # Enable pointer stability checks
```

## C++ Diagnostics

Enable with: `-Dnd4j.dsp.diagnostics=TRANSFER -Dnd4j.dsp.diagnostics.level=full`

## Consequences

### Positive
- Zero-overhead diagnostics when disabled
- Fine-grained control over device placement
- Early detection of replica memory leaks
- Pointer stability validation prevents graph replay corruption
- Plan recreation handles memory pressure gracefully

### Negative
- C++ header change triggers full rebuild
- Additional memory for tracking structures (~1-2 MB typical)
- Complexity in migration logic

### Neutral
- Requires integration into `DynamicShapePlanExecutor` for full benefit
- Tests required in `platform-tests` module only

## Migration Path

Existing code continues to work without changes. New features are opt-in via system properties. Integration points:

1. **Transfer tracking**: Wrap `replicateToDevice()` calls with timing + `transferSubsystem.record()`
2. **Pinning**: Check `pinningManager.resolveDevice()` before migration
3. **Replica tracking**: Call `replicaLeakDetector.registerReplica()` when caching, `unregisterReplica()` when clearing
4. **Pointer stability**: Check `pointerStabilityGuard.isFrozen()` before migration attempts
5. **Plan recreation**: Register `MemoryPressureCallback` and set `needsPlanRecreation` flag

## Testing

All tests in `platform-tests/src/test/java/.../DeviceTransferManagementTest.java`:
- 24 tests covering P1-P5 functionality
- Parameterized tests for all configuration combinations
- DSP diagnostics validation via `-Dnd4j.dsp.diagnostics=TRANSFER`

## References

- DSP Diagnostics: `libnd4j/include/graph/DspDiagnostics.h`
- Device Subsystem: `nd4j/.../linalg/framework/device/DeviceSubsystem.java`
- Transfer Metrics: `libnd4j/include/helpers/TransferMetrics.h`
