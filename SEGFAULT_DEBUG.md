# SIGSEGV Debug Investigation: VMThread calls cudaStreamSynchronize during plan unpin

## Summary

The embedding subprocess in the kompile project crashes with SIGSEGV inside `libcuda.so.1` at `cudaStreamSynchronize`. Two independent crash logs are available. The crashes exhibit **two different stack patterns** that reveal distinct phases of the same underlying bug.

**Crash log locations (in the kompile repo):**
- `kompile-rag-builds/kompile-fpna-v3/project/hs_err_pid2253408.log` (first crash)
- `kompile-rag-builds/kompile-fpna-v3/project/hs_err_pid2272305.log` (second crash, respawned subprocess)

---

## Crash Pattern A — First crash (pid 2253408): VMThread crash

```
Current thread: VMThread "VM Thread"

Native frames:
C  [libcuda.so.1+0x2baeed]
C  [libcuda.so.1+0x2f1860]
C  [libcuda.so.1+0x380bf3]
C  [libcudart.so.12.9.37+0x128b3]
C  [libcudart.so.12.9.37+0x7bc18]  cudaStreamSynchronize+0x1d8

siginfo: si_signo: 11 (SIGSEGV), si_code: 128 (SI_KERNEL), si_addr: 0x0000000000000000

VM state: at safepoint (shutting down)
VM_Operation: Exit, mode: safepoint, requested by thread "SIGTERM handler"
```

The "main" thread was at:
```
"main" #1 _thread_in_native
  at org.nd4j.linalg.jcublas.bindings.Nd4jCuda.unpinNativePlan(Native Method)
  at org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor.redispatchForCurrentShapes(DynamicShapePlanExecutor.java:1540)
  at org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor.executeNative(DynamicShapePlanExecutor.java:2299)
  at org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor.execute(DynamicShapePlanExecutor.java:1935)
```

Other threads at crash time:
- `"JavaCPP Deallocator"` daemon: `_thread_blocked`
- `"DeallocatorServiceThread_0"`: `_thread_blocked`
- `"DeallocatorServiceThread_1"`: `_thread_blocked`
- `"SIGTERM handler"`: `_thread_blocked` (this thread **requested** the Exit VM_Operation)

## Crash Pattern B — Second crash (pid 2272305): Main thread crash in executeDynamicShapePlan

```
Current thread: JavaThread "main" [_thread_in_native]

Native frames:
C  [libcuda.so.1+0x2f17b3]
C  [libcuda.so.1+0x380bf3]
C  [libcudart.so.12.9.37+0x128b3]
C  [libcudart.so.12.9.37+0x7bc18]  cudaStreamSynchronize+0x1d8

Java frames:
j  org.nd4j.linalg.jcublas.bindings.Nd4jCuda.executeDynamicShapePlan(...)I+0
j  org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor.executeNative(...)Ljava/util/Map;+6637
j  org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor.execute(...)Ljava/util/Map;+340
j  org.nd4j.autodiff.samediff.internal.InferenceSession.executeDynamicShapePlanBased(...)Ljava/util/Map;+308
j  org.nd4j.autodiff.samediff.internal.InferenceSession.output(...)Lorg/nd4j/autodiff/samediff/config/ExecutionResult;+1916
j  org.nd4j.autodiff.samediff.SameDiff.directExecHelper(...)Lorg/nd4j/autodiff/samediff/config/ExecutionResult;+522
j  org.nd4j.autodiff.samediff.SameDiff.batchOutputHelper(...)Lorg/nd4j/autodiff/samediff/config/ExecutionResult;+254
j  io.anserini.encoder.samediff.GenericDenseSameDiffEncoder.encodeSingleInferenceBatchFromEncodings(...)Ljava/util/List;+787
j  io.anserini.encoder.samediff.GenericDenseSameDiffEncoder.encodeBatchWithDynamicSizing(...)Ljava/util/List;+395
j  ai.kompile.embedding.anserini.subprocess.EmbeddingSubprocessMain.main([Ljava/lang/String;)V+324

siginfo: si_signo: 11 (SIGSEGV), si_code: 1 (SEGV_MAPERR), si_addr: 0x00007f1fac03fc2d
VM state: not at safepoint (normal execution)
```

---

## Environment

- **Hardware:** AMD Ryzen 9 5950X, 32 cores, 125GB RAM
- **GPUs:** RTX 4090 (24GB) + RTX 3070 Ti (8GB)
- **Driver:** 570.144, CUDA Runtime: 12.8, nd4j-cuda-12.9
- **JVM:** Amazon Corretto 17.0.17 (OpenJDK 17), G1 GC
- **JVM flags used for embedding subprocess:**
  ```
  -Xmx4096m -Xms1024m -XX:+UseG1GC -XX:MaxGCPauseMillis=100
  -Dnd4j.environment.maxThreads=4
  -Dnd4j.environment.maxMasterThreads=2
  -Dnd4j.environment.cudaCurrentDevice=0
  -Dnd4j.cublas.captureWorkspace=1
  -Dnd4j.optimizer.enabled=true
  -Dnd4j.optimizer.fp16=false
  -Dnd4j.dsp.noFreeze=false
  -Dnd4j.dsp.noDirect=false
  -Dnd4j.dsp.noAttnOverride=false
  -Dnd4j.dsp.noNativeDecodeInputs=false
  ```
- **Input batch shape at crash:** `input_ids -> shape=[8, 512]` (first real inference batch after model load)
- **Time to crash:** ~8 minutes (first crash at 485 seconds elapsed, second at 900 seconds)
- **Context:** Another JVM process (the main kompile app) also has ND4J/CUDA initialized on the same GPUs. The embedding subprocess is a separate JVM targeting cuda device 0 (RTX 4090).

---

## Key Observation: Two Different Crash Locations, Same Root Cause

**Pattern A (pid 2253408):**
- The JVM is shutting down via a `VM_Operation: Exit` triggered by a `SIGTERM handler` thread.
- During the JVM safepoint/shutdown, the **VMThread itself** calls `cudaStreamSynchronize`.
- This should never happen — the VMThread is a JVM-internal coordination thread and must never make CUDA calls.
- The main thread is mid-execution of `unpinNativePlan()` (a JNI call from `redispatchForCurrentShapes`).
- The hypothesis: `unpinNativePlan` triggers LRU eviction of an old plan via `NativePlanCache::unpinPlan` → `evictIfOverBudgetLocked` → `delete victim->second` (i.e., `delete NativeDynamicShapePlan`) → `platformFreePlanResources` → CUDA resource teardown that calls `cudaStreamSynchronize`. If this JNI call coincides with the JVM entering a safepoint (the SIGTERM handler requested Exit), the VMThread could be in the middle of a GC operation that also interacts with the CUDA context.

**Pattern B (pid 2272305):**
- No VMThread involvement — the main thread crashes directly in `executeDynamicShapePlan` (the JNI call that executes the compiled native plan).
- `si_code: 1 (SEGV_MAPERR)` — memory mapping error, a mapped address was accessed after being unmapped.
- `si_addr: 0x00007f1fac03fc2d` — the address is clearly not NULL, it's a stale/freed pointer (compare with Pattern A's NULL dereference `si_addr: 0x0`).
- This points to use-after-free: a CUDA stream handle or CUDA context object that was freed (by plan eviction or by the other JVM process) is accessed during `executeDynamicShapePlan`.

---

## Critical Code Paths to Investigate

### 1. `DynamicShapePlanExecutor.java` — `redispatchForCurrentShapes` (around line 1540)

**File:** `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/DynamicShapePlanExecutor.java`

The relevant path is at line 1540:
```java
nativeOps.unpinNativePlan(cache, nativePlanHandle);
```
This is called when a **plan swap** occurs — when `redispatchForCurrentShapes` discovers the cache returned a different plan handle for the current shapes (the `else` branch, i.e., `wasEverFrozen` is false). The old plan is unpinned so it becomes eligible for LRU eviction.

**The chain to investigate:**
1. `unpinNativePlan` → calls `NativePlanCache::unpinPlan(plan)` (in `NativeOps_dsp.cpp`)
2. `NativePlanCache::unpinPlan` → removes from `pinnedPlans_` set, calls `evictIfOverBudgetLocked()`
3. `evictIfOverBudgetLocked` → if over budget, calls `delete victim->second` (i.e., destructs a `NativeDynamicShapePlan`)
4. `~NativeDynamicShapePlan()` → calls `platformFreePlanResources()` first (to free GPU resources before slot arrays)
5. On CUDA builds, `platformFreePlanResources()` likely destroys CUDA graph handles, frees workspace memory, and potentially calls `cudaStreamSynchronize` to drain pending work before freeing GPU memory

**The race condition for Pattern A:**
- Main thread is inside `unpinNativePlan()` (a JNI native call, so thread state is `_thread_in_native`)
- SIGTERM arrives → JVM wants to do a safepoint stop-the-world → VMThread sends a safepoint request
- `_thread_in_native` threads are considered "safe" by the JVM and don't have to stop immediately — HOWEVER, threads in JNI code can still have the VMThread interact with CUDA context objects that the JNI thread is actively using
- The VMThread's `cudaStreamSynchronize` call suggests the VMThread is executing a destructor or finalizer that touches a CUDA stream

**Look specifically at:** Whether `evictIfOverBudgetLocked` can trigger synchronous `cudaStreamSynchronize` calls from the calling thread, and whether those calls are safe when the JVM is at a safepoint.

### 2. Native C++ `unpinNativePlan` and `NativePlanCache::unpinPlan`

**Files:**
- `libnd4j/include/legacy/cpu/NativeOps_dsp.cpp` — `unpinNativePlan` implementation (line 342)
- `libnd4j/include/graph/impl/NativePlanCache.cpp` — `unpinPlan` and `evictIfOverBudgetLocked`
- `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp` — `~NativeDynamicShapePlan()` and `platformFreePlanResources()` (CPU stubs in `NativeDynamicShapePlan_cuda_stubs.cpp`)

The CPU-side `platformFreePlanResources()` (in `NativeDynamicShapePlan_cuda_stubs.cpp`) only resets replay handles and deletes `PlanExecutionContext`. It does NOT call `cudaStreamSynchronize`.

**Therefore, on CUDA builds, there must be a CUDA-specific `platformFreePlanResources` implementation.** Look for:
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_cuda.cu` or similar `.cu` file
- Search for `platformFreePlanResources` in all CUDA `.cu` files

The CUDA `platformFreePlanResources` implementation is the most likely location of the `cudaStreamSynchronize` call being made during plan eviction. Check if it:
1. Calls `cudaGraphExecDestroy` (which requires stream drain)
2. Calls `cudaStreamSynchronize` explicitly before freeing workspace memory
3. Uses `cudaFreeAsync` which requires a stream argument and may synchronize

### 3. `NativeDynamicShapePlan::~NativeDynamicShapePlan()` destructor

**File:** `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp` (around line 879)

The destructor calls (in order):
1. `platformFreePlanResources()` — frees CUDA graphs, replay workspaces, cuBLAS workspace
2. `releaseKvScatterResources()`
3. `releasePlanFrozenRefsForTeardown(...)` — releases frozen reference counts
4. Frees slot arrays and plan-owned NDArrays
   - Each owned NDArray's DataBuffer gets `deleteBuffers()` called before deletion
   
**Key:** When `deleteBuffers()` is called on a CUDA NDArray inside the plan destructor, does it call `cudaStreamSynchronize`? Check `DataBuffer::deleteBuffers()` in `libnd4j/include/array/cuda/DataBuffer.cu`.

### 4. `OpaqueDataBufferDeallocator.java` — does it call `cudaStreamSynchronize`?

**File:** `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/OpaqueDataBufferDeallocator.java`

The `BufferDeallocator.deallocate()` method calls:
```java
Nd4j.getExecutioner().commit();  // This may call cudaStreamSynchronize!
Nd4j.getNativeOps().dbClose(buffer);
```

`Nd4j.getExecutioner().commit()` is a full GPU sync (stream synchronize). If the `DeallocatorServiceThread` calls this during GC, and if the GC is also calling into CUDA context — there could be a deadlock or corruption.

**However:** In Pattern A, both `DeallocatorServiceThread_0` and `DeallocatorServiceThread_1` are `_thread_blocked`. So they are NOT executing `deallocate()` at crash time. The `commit()` call in the deallocator is NOT the direct cause.

**The VMThread calling `cudaStreamSynchronize` is the smoking gun.** The VMThread should never call CUDA functions directly. The question is HOW the VMThread ends up in `cudaStreamSynchronize`.

### 5. Check for CUDA context initialization/teardown interactions

The crash in Pattern A occurs while `VM state: at safepoint (shutting down)`. During JVM shutdown:
1. The SIGTERM handler thread requests a `VM_Operation: Exit`
2. The VMThread executes the Exit operation
3. During Exit, the JVM runs shutdown hooks, finalizers, and cleanup

**If ND4J registered a JVM shutdown hook that calls `cudaStreamSynchronize`**, and this hook runs on the VMThread during the Exit operation, it would produce exactly this crash.

Search for:
```java
Runtime.getRuntime().addShutdownHook(...)
```
in ND4J CUDA backend code. Look particularly for shutdown hooks in:
- `Nd4jCuda` class
- `CudaEnvironment` 
- `JCublasBackend`
- `CudaContext` or `LaunchContext`
- Any class that initializes the CUDA execution context

### 6. CUDA Graph Capture Mode Interaction (`-Dnd4j.cublas.captureWorkspace=1`)

The system property `nd4j.cublas.captureWorkspace=1` enables CUDA graph workspace capture. This is highly relevant because:

1. During CUDA graph capture, any `cudaStreamSynchronize` on the captured stream is **illegal** — it can corrupt capture state
2. If a plan eviction (triggered by `unpinNativePlan`) destroys a plan that is currently captured or mid-replay, and the destruction calls `cudaStreamSynchronize` on the execution stream, this could corrupt the CUDA context for the process
3. A corrupted CUDA context may then produce `SIGSEGV` on the **next** `cudaStreamSynchronize` call (Pattern B) or at an unexpected time (Pattern A)

**Look for:** Whether `NativePlanCache::evictIfOverBudgetLocked` has any guards against evicting a plan whose CUDA graph is currently being captured or replayed. There is a comment in `NativeDynamicShapePlan.cpp` (around line 201-215) about a mutex that protects capture phases:

```
// during warmup and CUDA graph capture, legacy host-blocking CUDA API calls
// capture on the same device (error 906 → cascade 901).
```

Is this mutex held when `unpinNativePlan` → `evictIfOverBudgetLocked` → `delete victim` is called? If not, there is a race between plan execution (which captures CUDA graphs) and plan eviction (which destroys CUDA graph handles).

---

## Files to Read (in Priority Order)

1. **`libnd4j/include/graph/impl/NativeDynamicShapePlan_cuda.cu`** (or equivalent `.cu` CUDA implementation file — search for it)
   - Find `platformFreePlanResources()` CUDA implementation
   - Find any `cudaStreamSynchronize` calls in destructors or resource teardown
   - Find the CUDA graph mutex referenced in the `.cpp` file (around line 201)

2. **`libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp`** — specifically:
   - Line 879: `~NativeDynamicShapePlan()` destructor
   - Line 891: `platformFreePlanResources()` call (the CPU stub is in `_cuda_stubs.cpp`)
   - Lines 200-215: The CUDA capture mutex / `GraphExecutionMode` state transitions
   - Line 1403: `delete plan` inside what appears to be a plan teardown function

3. **`libnd4j/include/graph/impl/NativePlanCache.cpp`** — specifically:
   - Lines 163-167: `unpinPlan` — does it hold any lock during `evictIfOverBudgetLocked`?
   - Lines 210-217: `delete victim->second` — this triggers `~NativeDynamicShapePlan`
   - Is the plan cache mutex (`mutex_`) the same mutex as the capture guard? It should NOT be. Check for deadlock.

4. **`libnd4j/include/array/cuda/DataBuffer.cu`** — specifically:
   - `deleteBuffers()` implementation: does it call `cudaStreamSynchronize`?
   - Line 326: `cudaStreamSynchronize(cudaStreamPerThread)` in D2D copy — when is this triggered?
   - Line 547: `cudaStreamSynchronize(*stream)` — in what path?
   - Line 1049: `cudaStreamSynchronize(stream)` in `syncToPrimary`

5. **`libnd4j/include/legacy/cpu/NativeOps_dsp.cpp`** — lines 342-347: the `unpinNativePlan` function

6. **`nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/DynamicShapePlanExecutor.java`** — specifically:
   - Lines 1419-1557: `redispatchForCurrentShapes` — the two `unpinNativePlan` call sites
   - Lines 2261-2310: `executeNative` entry point
   - Lines 958-1003: `releaseGpuIntermediates` call and comments about CUDA graph teardown

7. **Search all CUDA backend files for JVM shutdown hooks:**
   ```bash
   grep -rn "addShutdownHook\|Runtime.getRuntime" \
     nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/ \
     nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/jcublas/
   ```

---

## Hypotheses to Verify (Most Likely First)

### Hypothesis 1: Plan eviction during CUDA graph execution triggers `cudaStreamSynchronize` from an unsafe context

**Mechanism:**
1. Main thread calls `redispatchForCurrentShapes` (first inference with `input_ids=[8,512]`)
2. Cache lookup returns a new plan handle (cache miss or plan swap)
3. Old plan is unpinned: `nativeOps.unpinNativePlan(cache, nativePlanHandle)` (line 1540)
4. `NativePlanCache::unpinPlan` → `evictIfOverBudgetLocked` → `delete victim` (old plan)
5. `~NativeDynamicShapePlan` → `platformFreePlanResources()` (CUDA build) → **calls `cudaStreamSynchronize`** to drain pending CUDA work before freeing GPU memory
6. Meanwhile, SIGTERM arrives; JVM initiates shutdown; VMThread is executing Exit operation
7. The VMThread's native frame shows `cudaStreamSynchronize` — this is the destructor's sync being attributed to VMThread through some execution context sharing (or the destructor runs on a thread that the VMThread is coordinating)

**ALTERNATIVELY for Pattern A:** The destructor's `cudaStreamSynchronize` runs on the main thread, but simultaneously the JVM's safepoint mechanism causes the VMThread to also call into the CUDA context for cleanup. Two concurrent `cudaStreamSynchronize` calls on the same device can produce SEGV in `libcuda.so.1` due to internal CUDA locking.

### Hypothesis 2: Use-after-free of CUDA stream handle (Pattern B)

**Mechanism:**
1. First subprocess run (pid 2253408) crashes and creates core dump
2. Subprocess is respawned (pid 2272305)
3. The other JVM process (main kompile app) was also using the GPU; its CUDA context remained initialized
4. The respawned subprocess initializes a new CUDA context on device 0
5. During `executeDynamicShapePlan`, the C++ code dereferences a CUDA stream handle (`cudaStream_t`) that maps to a virtual address that was valid in the first process run but is now recycled/unmapped
6. `si_addr: 0x00007f1fac03fc2d` is a non-null stale pointer — consistent with a CUDA handle that was freed and reallocated

**OR:**
The plan cache retained a CUDA graph handle from a previous encode call. The handle was created during a CUDA graph capture with `captureWorkspace=1`. The handle became stale (pointing to freed GPU memory) because `deleteBuffers()` freed the underlying GPU buffer while the graph handle still referenced it. On replay, `cudaStreamSynchronize` after launching the graph encounters the stale reference → SEGV.

### Hypothesis 3: CUDA graph capture mutex not held during plan eviction

The comment in `NativeDynamicShapePlan.cpp` (around line 201-215) mentions a static mutex that guards CUDA graph capture. If `NativePlanCache::evictIfOverBudgetLocked` deletes a plan without holding this mutex, and the plan's CUDA graph is mid-replay on the main thread's stream, the `cudaGraphExecDestroy` call in the destructor would race with `cudaGraphLaunch` on the main thread — corrupting the CUDA context and producing SEGV on the next `cudaStreamSynchronize`.

---

## Specific Questions to Answer

1. **In the CUDA build of `NativeDynamicShapePlan`, does `platformFreePlanResources()` call `cudaStreamSynchronize`?** If so, this call runs synchronously on whichever thread calls `unpinNativePlan`, which may be the main thread while the JVM is in safepoint.

2. **Is there a race between `NativePlanCache::evictIfOverBudgetLocked` (which holds `mutex_`) and any CUDA graph capture/replay mutex?** If the capture mutex is separate from the cache mutex, eviction can delete a plan whose CUDA graphs are being replayed on the main thread.

3. **Does `DataBuffer::deleteBuffers()` on the CUDA backend call `cudaStreamSynchronize`?** If yes, and if `deleteBuffers()` is called from the plan destructor (which runs during eviction triggered by `unpinNativePlan` on the main thread), this could be the exact call that ends up attributed to the VMThread.

4. **Are there any JVM shutdown hooks in the ND4J CUDA backend that call `cudaStreamSynchronize`?** A shutdown hook running on the VMThread's coordinated shutdown would explain Pattern A perfectly.

5. **Is the `captureWorkspace=1` flag causing workspace interior pointers to be stored in CUDA graph nodes?** The comment in `DataBuffer.cu` (around line 1380) mentions: "Recording cudaFreeAsync for graph-external memory creates MemFree graph nodes." If workspace memory is freed while a CUDA graph is replaying it, SEGV follows.

---

## The Fix Direction

The root fix must ensure:

**Fix 1 — No CUDA stream operations in plan destructor called from the JNI thread during shutdown:**
Any CUDA resource teardown (`cudaStreamSynchronize`, `cudaGraphExecDestroy`, `cudaFreeAsync`) in `platformFreePlanResources()` or `~NativeDynamicShapePlan()` that is triggered by `unpinNativePlan` must be deferred to a safe background thread — specifically, a thread that is not the JVM's VMThread and not a thread that can be pre-empted at a JVM safepoint while holding CUDA resources.

The existing `DeallocatorServiceThread_0/1` threads are the right place for this. They are daemon threads that process deferred deallocations. Plan eviction could post the plan pointer to a queue consumed by `DeallocatorServiceThread`, similar to how `OpaqueDataBufferDeallocator` defers `dbClose` to avoid calling `free()` at the wrong time.

**Fix 2 — Guard plan eviction against active CUDA graph capture/replay:**
`NativePlanCache::evictIfOverBudgetLocked` must not evict (delete) a plan whose CUDA graph is currently being captured or replayed. Check whether the static CUDA capture mutex (referenced in `NativeDynamicShapePlan.cpp` around line 201) is acquired before deletion or whether a per-plan "is-executing" flag is checked.

**Fix 3 — Handle the concurrent-JVM CUDA context issue:**
Two JVM processes sharing GPUs without CUDA MPS (Multi-Process Service) can produce CUDA context corruption. This is an environmental issue but can be mitigated by ensuring the embedding subprocess initializes a fresh CUDA context and does not assume any CUDA state from a previous process invocation.

---

## How to Reproduce

Run the kompile embedding subprocess with a batch of 8 sequences of length 512, with `-Dnd4j.cublas.captureWorkspace=1 -Dnd4j.optimizer.enabled=true`, while the main kompile app is also running with ND4J/CUDA initialized on the same device. The crash occurs on the **first real inference batch** after model load, suggesting the plan cache starts empty and the first `redispatchForCurrentShapes` triggers a plan creation + (potentially) an eviction of a warmup plan.

To isolate: first try disabling CUDA graph capture (`-Dnd4j.cublas.captureWorkspace=0`) to see if the crash disappears. If it does, the capture workspace mechanism is the culprit.

---

## Relevant deeplearning4j Source Files

All paths are relative to the repo root:

```
nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/DynamicShapePlanExecutor.java
nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/OpaqueDataBufferDeallocator.java
nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/DeallocatorService.java
nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/NativeOps.java
libnd4j/include/graph/impl/NativePlanCache.cpp
libnd4j/include/graph/NativePlanCache.h
libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp
libnd4j/include/graph/impl/NativeDynamicShapePlan_cuda_stubs.cpp
libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp
libnd4j/include/legacy/cpu/NativeOps_dsp.cpp
libnd4j/include/array/cuda/DataBuffer.cu
libnd4j/include/array/cuda/NDArray_core.cu
libnd4j/include/execution/cuda/CudaGraphScheduler.h
libnd4j/include/execution/cpu/LaunchContext.cpp
libnd4j/include/graph/NativeDynamicShapePlan.h
```

Search for all CUDA-specific `.cu` files implementing the plan execution platform dispatch:
```bash
find libnd4j -name "NativeDynamicShapePlan_cuda*.cu" -o -name "NativeDynamicShapePlan*.cu" 2>/dev/null
```

---

## Important Context: `captureWorkspace` and CUDA Graph Interaction

The flag `-Dnd4j.cublas.captureWorkspace=1` enables a mode where the cuBLAS workspace is allocated inside a CUDA graph capture region, so its pointer is stable across CUDA graph replays. This is critical for getting CUDA graphs to work with cuBLAS operations, but it creates a tight coupling between the workspace buffer lifecycle and the CUDA graph lifecycle.

In `libnd4j/include/array/cuda/DataBuffer.cu` around line 1330:
```
// Recording cudaFreeAsync for graph-external memory creates MemFree graph nodes
```

And around line 1380:
```
// cudaFreeAsync on interior pointer → "invalid argument" → cudaFree fallback
```

If the workspace is freed (via plan eviction → destructor → `cudaFreeAsync`) while a CUDA graph that **captures that workspace** is replaying on another stream, the CUDA driver will encounter a freed pointer during graph node execution → SEGV.

The capture workspace mode (`captureWorkspace=1`) requires that the plan and its associated workspace stay alive for the entire lifetime of any CUDA graph that captured it. **Plan eviction must never destroy a plan whose workspace is referenced by a live CUDA graph.**

---

## What a Correct Fix Looks Like

The general principle: **No CUDA API calls (especially `cudaStreamSynchronize`, `cudaGraphExecDestroy`, `cudaFreeAsync`) should be made from:**
1. A JVM thread that is in `_thread_in_native` state during a JVM safepoint
2. The JVM's VMThread
3. Any context where the CUDA driver could be in a race with another CUDA API call on the same device

**Concrete fix options:**

**Option A — Deferred plan deletion via DeallocatorServiceThread:**
Instead of `delete victim->second` in `evictIfOverBudgetLocked`, post the plan pointer to a concurrent queue. The `DeallocatorServiceThread` drains this queue and performs actual deletion (which includes CUDA teardown) asynchronously and at a safe point. This requires the plan cache to track "pending deletion" plans separately from "pinned" and "unpinned live" plans.

**Option B — Pre-drain before eviction:**
Before calling `delete victim->second` in `evictIfOverBudgetLocked`, call `cudaStreamSynchronize` on all streams that could have pending work referencing the victim plan's GPU memory. This must be done while holding a lock that prevents new CUDA work from being submitted. Difficult to implement correctly in the presence of concurrent CUDA graph replays.

**Option C — Reference-count CUDA graph handles separately from plans:**
The CUDA graph handles (`cudaGraphExec_t`) should be reference-counted independently of the plan. The plan destructor decrements the ref count; when the ref count reaches zero, the CUDA graph is destroyed. The CUDA graph's memory references prevent GPU memory from being freed until all replays complete.

**Option D — Disable plan eviction during CUDA graph capture/replay:**
Add a process-global `atomic<int>` counter for "plans currently in CUDA graph capture or replay." When this counter is nonzero, `evictIfOverBudgetLocked` skips deletion and instead marks plans for deferred eviction. This prevents the race window without requiring per-plan ref-counting.

The simplest correct fix for the immediate crash is **Option A** (deferred deletion), since it matches the existing `DeallocatorServiceThread` pattern already used for `OpaqueDataBuffer` deallocation.
