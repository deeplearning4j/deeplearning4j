# ADR: GC Pressure Optimization

## Status

Implemented

Proposed by: Adam Gibson (February 2025)

Discussed with: Development Team

## Context

ND4J manages native (off-heap) memory via JavaCPP's `Pointer` system, with garbage collection serving as the safety net for unreleased native allocations. The `DeallocatorService` monitors a `PhantomReference` queue and triggers native deallocation when Java objects are collected. Additionally, `System.gc()` calls are used to encourage timely collection of unreachable native-backed objects.

During VLM inference workloads, this GC-based approach caused severe performance problems:

**OpaqueDataBuffer Retry Loops**: When a native allocation fails (GPU OOM), the `OpaqueDataBuffer` constructor retries up to `MAX_TRIES=5` times, calling `System.gc()` on each retry to free unreachable native objects. During autoregressive decode, temporary OOM spikes are normal (the pool needs to trim and reclaim memory). Each OOM spike triggered 5 retries × `System.gc()`, resulting in 1936+ Full GC events during a single decode session.

**DeallocatorService Blind Timer**: The `DeallocatorService` ran `System.gc()` every 100ms when the PhantomReference queue was empty, attempting to encourage object finalization. This created a steady stream of Full GC events regardless of actual memory pressure — 11,800 Full GCs during a 256-token decode run.

**Full GC Stop-the-World Pauses**: Each `System.gc()` call triggers a Full GC that pauses all application threads for 50-200ms. With 11,800 Full GCs, this added ~10-40 minutes of pure GC overhead to a decode run that should take ~5 minutes.

**PhantomRef Strong Reference Cycle**: The `DeallocatableReference` held a strong reference to its associated native resource, preventing the PhantomReference from being enqueued by the GC. This meant GC-based cleanup was fundamentally broken — arrays not explicitly closed would NEVER be freed, making the excessive GC calls doubly wasteful.

## Decision

We implement a memory-pressure-aware GC strategy that eliminates unnecessary `System.gc()` calls while preserving the safety net for genuine memory pressure situations.

### Conditional GC via Heap Pressure

Replace unconditional `System.gc()` calls with a heap-pressure check:

```java
public static void gcIfHeapPressured() {
    Runtime rt = Runtime.getRuntime();
    long used = rt.totalMemory() - rt.freeMemory();
    long max = rt.maxMemory();
    if ((double) used / max > HEAP_PRESSURE_THRESHOLD) { // default 0.75
        System.gc();
    }
}
```

**Threshold**: 75% of max heap. Below this threshold, the JVM has ample capacity for object allocation and GC will happen naturally during minor collections. Above this threshold, explicit GC may help recover unreachable objects before the JVM is forced into emergency Full GC.

This replaces `System.gc()` calls in:
- `OpaqueDataBuffer` constructor retry loops
- `DeallocatorService` periodic timer
- Any other allocation-failure recovery paths

### DeallocatorService Timer Optimization

The periodic GC timer is changed from blind invocation to memory-aware:

```java
// Before: System.gc() every 100ms when queue empty
// After:  gcIfHeapPressured() every 5000ms when queue empty

private void processQueue() {
    Reference<?> ref = queue.poll();
    if (ref == null) {
        long now = System.currentTimeMillis();
        if (now - lastGcTime > autoGcWindow) {  // default 5000ms
            gcIfHeapPressured();
            lastGcTime = now;
        }
        return;
    }
    // Process deallocation...
}
```

**Window increase**: 100ms → 5000ms. The 100ms window was far too aggressive — GC benefits are amortized over seconds, not milliseconds. The 5000ms window provides adequate GC frequency for long-running workloads while eliminating the 10 GC/second overhead.

### DSP Auto-GC Suppression

During DynamicShapePlan execution, GC is suppressed entirely because DSP manages array lifecycle explicitly:

```java
// At DSP execution start:
int savedWindow = Nd4j.getMemoryManager().getAutoGcWindow();
Nd4j.getMemoryManager().setAutoGcWindow(Integer.MAX_VALUE);

try {
    // Execute DSP steps — all arrays freed explicitly via liveness schedule
    executePlan(plan);
} finally {
    Nd4j.getMemoryManager().setAutoGcWindow(savedWindow);
}
```

**Rationale**: DSP's pre-computed liveness schedule frees intermediates immediately after their last consumer. GC-based cleanup is unnecessary and counterproductive — the stop-the-world pauses interrupt GPU execution and cause pipeline bubbles.

**Note**: `setAutoGcWindow(0)` throws `IllegalStateException` because `CudaConfiguration` rejects values < 1. `Integer.MAX_VALUE` effectively disables the timer without hitting this validation.

### PhantomRef Cycle Fix

The `setCloseable(false)` → `dataBuffer.setConstant(true)` poisoning was fixed:

```java
// Before: setCloseable(false) was permanent
public void setCloseable(boolean closeable) {
    if (!closeable) {
        dataBuffer.setConstant(true);  // Marks buffer as non-freeable
    }
    // No way to undo!
}

// After: setCloseable(true) reverses the poisoning
public void setCloseable(boolean closeable) {
    if (!closeable) {
        dataBuffer.setConstant(true);
    } else {
        dataBuffer.setConstant(false);  // Undo poisoning
    }
}
```

`SameDiffMemoryUtils.safeClose()` handles the full cleanup sequence for poisoned arrays:

```java
public static void safeClose(INDArray array) {
    if (array == null) return;
    array.setCloseable(true);     // Undo constant poisoning
    array.close();                // Free native memory
}
```

### Shutdown Hook

A shutdown flag prevents native deallocation during JVM exit:

```java
private final AtomicBoolean shutdownInProgress = new AtomicBoolean(false);

// Registered at construction:
Runtime.getRuntime().addShutdownHook(new Thread(() -> {
    shutdownInProgress.set(true);
}));

// In deallocation loop:
if (shutdownInProgress.get()) {
    return;  // Skip native free — OS will reclaim process memory
}
```

This prevents SIGABRT crashes caused by calling `free()` on corrupted heap metadata during JVM shutdown. The corruption originates from C++ ops overrunning host buffers (see ADR 0060 padding discussion), and manifests only when the GC processes unreachable arrays during the shutdown sequence.

## Consequences

### Advantages

**99.98% GC Reduction**: 11,800 Full GCs → 0-2 Full GCs during DSP execution. The remaining 0-2 GCs occur during non-DSP phases (model loading, result processing) when heap pressure is genuine.

**Eliminated GC Overhead**: ~10-40 minutes of stop-the-world pauses eliminated from a 5-minute decode run. Total VLM decode time drops proportionally.

**GPU Pipeline Continuity**: No stop-the-world pauses during GPU execution. CUDA streams run uninterrupted, maximizing GPU utilization.

**Clean Shutdown**: Shutdown hook eliminates SIGABRT crashes from GC-triggered frees of corrupted buffers.

### Disadvantages

**Delayed Native Cleanup**: With less frequent GC, unreachable native objects take longer to be collected. This increases peak native memory usage for workloads that rely on GC for cleanup (non-DSP execution paths).

**Manual Close Requirement**: DSP's GC suppression means arrays that escape explicit close tracking will leak permanently. All intermediate arrays must be closed via the liveness schedule or explicit `safeClose()` calls.

**Heap Threshold Sensitivity**: The 75% threshold is empirically tuned. Workloads with very large Java heap usage (e.g., large batch preprocessing) may need adjustment via system property.

## References

- DeallocatorService.java in nd4j-api
- OpaqueDataBuffer.java (gcIfHeapPressured integration)
- DynamicShapePlanExecutor.java (autoGcWindow suppression)
- SameDiffMemoryUtils.java (safeClose utility)
- BaseNDArray.java (setCloseable fix)
