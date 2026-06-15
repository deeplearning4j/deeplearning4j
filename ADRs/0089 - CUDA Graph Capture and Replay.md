# ADR: CUDA Graph Capture, Replay, and Gap Op Orchestration

## Status
Accepted

## Related ADRs
- [ADR 0082](0082%20-%20CUDA%20Graph%20Replay%20Pointer%20Stability%20and%20Frozen%20Steady-State.md) — resolves the known issues documented in this ADR

## Date
2026-02-18 (Original), 2026-03-30 (Gap Ops, Shared Resources, Phase Model)

## Context

DeepLearning4J's SameDiff execution engine processes computation graphs through a native `NativeDynamicShapePlan` executor. For inference workloads like LLM token generation, the same graph is executed thousands of times with identical structure but varying input data (e.g., growing KV cache sequences).

Each op execution incurs overhead from:
1. **CPU-side launch overhead**: Java→JNI→C++ traversal, context setup
2. **Kernel launch latency**: Each CUDA kernel requires a separate launch call (~5-10μs each)
3. **Memory operations**: Individual allocations and transfers for each op

For autoregressive LLM generation with hundreds of ops per forward pass, this overhead becomes significant.

### Requirements

1. **Transparent Integration**: Work with existing SameDiff execution without API changes
2. **Shape Dynamic Handling**: Support graphs with dynamic shapes (e.g., growing KV cache)
3. **Segmentation**: Handle non-capturable ops (shape-dependent, host callbacks) by segmenting the graph
4. **Gap Op Support**: Handle ops within a segment that Triton cannot compile (matmul/cuBLAS, softmax, etc.)
5. **Batch-Zero**: Efficient output buffer zeroing compatible with CUDA graph capture
6. **Debugging Support**: Provide visibility into capture status, node contributions, and replay statistics
7. **Visualization**: PyTorch-style Chrome trace export and HTML visualization

## Decision

Implement CUDA Graph capture and replay for the native execution plan with comprehensive visualization support, following PyTorch's API patterns. Segments contain a mix of Triton-compiled sub-kernels and native "gap ops" that are interleaved during execution and captured together into a single CUDA graph.

---

## Architecture Overview

### Execution Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Segment Execution Lifecycle                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  WARMUP (executionCount == 0)                                          │
│    └── executeSegmentSlotBySlot() — native path for all ops            │
│    └── Observe batch-zero targets (register output buffers)            │
│    └── Learn gap op structure + view producers                         │
│                                                                         │
│  COMPILE (executionCount == 1, shapes frozen)                          │
│    └── Triton compiles sub-kernels for compilable ops                  │
│    └── Non-compilable ops become "gap ops" (fallbackRanges)            │
│    └── Shape observation for symbolic shape matching                   │
│                                                                         │
│  CAPTURE (executionCount in [captureMinExec, captureMinExec+2])        │
│    └── Pre-capture warmup (slot-by-slot with capture buffers)          │
│    └── Set up capture buffers for external + cross-segment inputs      │
│    └── Batch-zero registration outside graph                           │
│    └── cudaStreamBeginCapture()                                        │
│    └── Execute: Triton sub-kernels + gap ops interleaved               │
│    └── Gap ops run on capture stream → captured INTO graph             │
│    └── cudaStreamEndCapture() + cudaGraphInstantiate()                 │
│    └── SKIP initial launch (warmup results serve this pass)            │
│                                                                         │
│  REPLAY (executionCount > capture window, has replayHandle)            │
│    └── Phase 1: Pre-replay setup (capture buffer D2D, ext sync)        │
│    └── Phase 2: Batch-zero outside graph (cudaMemsetAsync)             │
│    └── Phase 3: cuBLAS workspace zeroing                               │
│    └── Phase 4: Graph launch (cudaGraphLaunch)                         │
│    └── Phase 5: Post-replay gap ops (only if NOT captured in graph)    │
│    └── Phase 6: Output slot restoration from slotArrayCache            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Segment Internal Structure: Triton Sub-Kernels and Gap Ops

A segment is a contiguous range of slots `[startSlot, endSlot]`. The Triton compiler attempts to fuse consecutive ops into GPU kernels. Ops that cannot be compiled (matmul/cuBLAS, softmax, attention, concat, split) become **gap ops** — they run via the native ND4J CUDA path (cuBLAS, cuDNN, etc.).

**Example: seg[1320-1323] from SmolDocling vision encoder attention layer:**

```
slot 1320: matmul      → GAP OP (cuBLAS, not Triton-compilable)
slot 1321: multiply    → TRITON SUB-KERNEL #0
slot 1322: softmax     → GAP OP (cuBLAS/native)
slot 1323: matmul      → GAP OP (cuBLAS)
```

During `TritonGraphBackend::executeSegment()`, the execution is interleaved:

```
1. Leading gap [1320-1320]  → fallbackRangeExecutor_(1320, 1320)  → matmul via cuBLAS
2. Triton sub-kernel [1321] → cuLaunchKernel(multiply_kernel)     → GPU SM execution
3. Trailing gap [1322-1323] → fallbackRangeExecutor_(1322, 1323)  → softmax + matmul
```

**Gap Op Identification** (`TritonGraphBackend::getGapSlots()`):
- Slot is a gap if NOT covered by any compiled sub-kernel
- Identified at compile time, stored in `CompiledSegment::fallbackRanges`
- Gap count = total slots - slots covered by sub-kernels

### Gap Ops During Capture vs Replay

#### During CUDA Graph Capture

When `tritonAllowFallbackCapture=true` (default), gap ops execute ON the capture stream and are **recorded INTO the CUDA graph**:

```cpp
// fallbackRangeExecutor_ lambda (line 1608 in gpubackend.cpp):
if (streamIsCapturing) {
    // Sync Triton stream → gap stream via CUDA events (graph dependency edge)
    cudaEventRecord(syncEvent, tritonStr);
    cudaStreamWaitEvent(gapStr, syncEvent, 0);

    seg.gapOpsCapturedInGraph = true;  // Mark: gap ops are IN the graph
    // tl_graphExecutionActive stays true → ops allocate from capture workspace
}
executeSegmentSlotBySlot(gapSeg, ...);  // cuBLAS/cuDNN calls recorded into graph

if (streamIsCapturing) {
    // Sync gap stream → Triton stream via CUDA events
    cudaEventRecord(syncEvent, gapStr);
    cudaStreamWaitEvent(tritonStr, syncEvent, 0);
}
```

The captured graph contains ALL operations: Triton kernels + cuBLAS matmuls + stream sync events.

#### During Replay

The behavior depends on `gapOpsCapturedInGraph`:

| `gapOpsCapturedInGraph` | Replay Behavior |
|---|---|
| `true` | Gap ops SKIPPED — already baked into the CUDA graph. Graph launch replays everything. |
| `false` | Gap ops RE-EXECUTED slot-by-slot AFTER graph replay (with `cudaStreamSynchronize` first). |

**Critical:** When gap ops are captured, the graph contains cuBLAS operations that depend on:
1. The cuBLAS workspace state at capture time
2. The cuBLAS handle's stream binding
3. The exact GPU addresses of input/output buffers (via capture buffers)

---

## Shared Resources Audit

### Category 1: GPU Memory — Shared Across Segments

#### 1.1 cuBLAS Workspace (`cublasWorkspaceBuffer_`)

| Property | Value |
|---|---|
| **Type** | Device GPU memory (256 MB) |
| **Declared** | `NativeDynamicShapePlan.h:1161` |
| **Scope** | Per-plan, shared across ALL segments |
| **Allocated** | `ensureCublasWorkspace()` via `cudaMalloc` (cublas.cu:48) |
| **Set for capture** | `setCublasWorkspaceForCapture()` — binds to cuBLAS handle + stream |
| **Set for warmup** | `setCublasWorkspaceForWarmup()` — binds without stream override |
| **Zeroed before replay** | `cudaMemsetAsync(cublasWorkspaceBuffer_, 0, ...)` (gpubackend.cpp:1302) |
| **Restored after capture** | `restoreCublasWorkspaceAfterCapture()` — clears thread-local |

**Why shared is dangerous:** Each segment's cuBLAS matmul may leave different state in the workspace. If seg[1316-1317]'s replay writes cuBLAS workspace state, then seg[1320-1323]'s replay reads stale state from seg[1316-1317]'s cuBLAS GEMM — causing the strided batched kernel to hang.

**Current mitigation:** Zero workspace before EVERY segment replay (line 1301-1305). This ensures deterministic initial state regardless of prior segment.

**Thread-local mirrors:**
```cpp
extern thread_local void*  tl_cublasWorkspacePtr;   // Active workspace pointer
extern thread_local size_t tl_cublasWorkspaceSize;   // Active workspace size
```

#### 1.2 Capture Workspace (per-segment, allocated from plan)

| Property | Value |
|---|---|
| **Type** | Device GPU memory (32 MB per segment) |
| **Declared** | `GraphReplayHandle` internal, accessed via `getWorkspacePtr()` |
| **Scope** | Per-segment (each captured segment has its own workspace) |
| **Allocated** | During capture setup (gpubackend.cpp:1731) |
| **Used by** | Gap ops during capture allocate from this workspace |
| **Freed** | When replay handle is destroyed |

**Thread-local access:**
```cpp
extern thread_local void*  tl_captureWorkspace;       // Current capture workspace ptr
extern thread_local size_t tl_captureWorkspaceSize;    // Total workspace bytes
extern thread_local size_t tl_captureWorkspaceOffset;  // Bump allocator offset
```

When `tl_graphExecutionActive=true` and `tl_captureWorkspace!=nullptr`, CudaMemoryPool allocates from this workspace instead of calling `cudaMallocAsync`. This keeps allocations capture-compatible.

#### 1.3 Pinned Host Workspace (per-segment)

| Property | Value |
|---|---|
| **Type** | Pinned host memory (32 MB per segment) |
| **Declared** | Thread-local `tl_captureHostWorkspace` |
| **Scope** | Per-segment capture |
| **Used by** | H2D memcpy nodes in captured graph (pinned source for async copies) |
| **Lifetime** | Preserved in replay handle after capture (`tl_capturedHostPtrs`) |

```cpp
extern thread_local void*  tl_captureHostWorkspace;
extern thread_local size_t tl_captureHostWorkspaceSize;
extern thread_local size_t tl_captureHostWorkspaceOffset;
extern thread_local std::vector<void*> tl_capturedHostPtrs;
```

### Category 2: Array Caches — Plan-Wide

#### 2.1 Slot Array Cache (`slotArrayCache_`)

| Property | Value |
|---|---|
| **Type** | `NDArray**` (one per output slot) |
| **Declared** | `NativeDynamicShapePlan.h:968` |
| **Scope** | Plan-wide, ALL segments read/write |
| **Size** | `totalOutputSlots_` entries |

**Critical sharing pattern:** Segment A produces output in slot X → stored in `slotArrayCache_[X]`. Segment B reads slot X as cross-segment input. If slot X's DataBuffer is freed (Java closes the array), the pointer becomes dangling.

**Post-replay:** `outputSlots_[si] = slotArrayCache_[si]` (line 1546) — graph wrote to capture workspace addresses, so outputSlots must point to the cache (which holds the workspace arrays).

#### 2.2 Output Slots (`outputSlots_`)

| Property | Value |
|---|---|
| **Type** | `NDArray**` (one per output slot) |
| **Declared** | `NativeDynamicShapePlan.h:968` |
| **Scope** | Plan-wide, mutable per-segment |

**Difference from slotArrayCache_:** `outputSlots_` is the "live" view — points to the array currently holding this slot's data. `slotArrayCache_` is the "stable" cache — persists across executions. During replay, these may differ (outputSlots_ points to warmup array, slotArrayCache_ points to capture workspace array).

### Category 3: Batch-Zero System

#### 3.1 Batch-Zero Entry List (`batchZeroEntries_`)

| Property | Value |
|---|---|
| **Type** | `std::vector<BatchZeroEntry>` where `BatchZeroEntry = {void* ptr, int bytes}` |
| **Declared** | `NativeDynamicShapePlan.h:1183` |
| **Scope** | Per-segment (rebuilt each segment) |
| **Populated** | Registration-based: during warmup, ops call `registerBatchZeroBuffer()` |
| **Used** | Pre-replay: `cudaMemsetAsync(entry.ptr, 0, entry.bytes, cudaStr)` for each entry |

**Registration flow:**
1. `startBatchZeroRegistration()` — enables thread-local `tl_batchZeroRegistering`
2. During warmup, each op's output allocation calls `registerBatchZeroBuffer(ptr, bytes)`
3. `finishBatchZeroRegistration()` — copies from thread-local to `batchZeroEntries_`

**Fallback:** If registration yields nothing, `collectBatchZeroTargets()` scans all slots in the segment.

**Thread-local state:**
```cpp
thread_local bool tl_batchZeroActive;         // Suppresses individual nullify() calls
thread_local bool tl_batchZeroRegistering;    // Enables registration collection
thread_local std::vector<RegEntry> tl_batchZeroRegistered;  // Collected entries
```

**Critical invariant:** The `void* ptr` in each entry must still be valid at replay time. If the DataBuffer was freed and reallocated at a different address, the entry is stale → zeros wrong memory.

#### 3.2 Batch-Zero Device Arrays (GPU-side kernel launch)

| Property | Value |
|---|---|
| **Type** | Device pointer arrays for batch-zero kernel |
| **Declared** | `NativeDynamicShapePlan.h:1190-1200` |
| **Fields** | `batchZeroDevicePtrs_`, `batchZeroDeviceSizes_`, `batchZeroHostPtrs_`, `batchZeroHostSizes_` |

### Category 4: Capture Buffer System

#### 4.1 Capture Buffers (per-segment, in GraphReplayHandle)

| Property | Value |
|---|---|
| **Type** | `std::vector<CaptureBuffer>` in `GraphReplayHandle` |
| **Purpose** | Fixed-address GPU buffers for inputs whose addresses change between executions |
| **Two types** | PLACEHOLDER (external inputs) and CROSS-SEGMENT (output slots from prior segments) |

```cpp
struct CaptureBuffer {
    int externalInputIndex;      // >= 0 for external inputs
    int crossSegmentSlotIdx;     // >= 0 for cross-segment outputs
    NDArray* buffer;             // Fixed-address capture buffer
    size_t capturedSize;         // Size at capture time
    const void* lastSourcePtr;   // Last known source address
    bool directReference;        // If true, no D2D copy — graph reads original address
    bool initialCopyDone;        // Whether first D2D copy has occurred
};
```

**Lifecycle:**
1. **Created** during capture setup — one per external input + one per cross-segment slot
2. **D2D updated** before each replay — fresh data copied into fixed-address buffer
3. **directReference** entries: no copy, graph reads from original address (weights/constants)

**Replay D2D flow (line 1011-1069):**
```
For each capture buffer:
  if directReference → skip (address assumed stable)
  if externalInput → syncToDevice() then D2D copy
  if crossSegment → already on device, just D2D copy
  copyIntoCaptureBuffer(cb.buffer, src, cudaStr)
```

#### 4.2 Batch D2D System (optimized bulk copy)

| Property | Value |
|---|---|
| **Type** | Device arrays of src/dst/size for batched D2D kernel |
| **Declared** | `NativeDynamicShapePlan.h:1208-1223` |
| **Purpose** | Replace N individual `cudaMemcpyAsync` with 1 batch kernel |

### Category 5: Thread-Local Execution State

| Thread-Local | Type | Purpose | Set By | Read By |
|---|---|---|---|---|
| `tl_graphExecutionActive` | `bool` | Controls allocation path (workspace vs normal) | Capture begin/end, gap ops | CudaMemoryPool, DebugHelper, DspVerifyUtils |
| `tl_dspExecutionStream` | `cudaStream_t` | Stream for `syncToSpecial()` calls | `DspStreamGuard` RAII | `DataBuffer::syncToSpecial()` |
| `tl_graphCaptureStream` | `cudaStream_t` | Capture stream for cuDNN/cuBLAS | Capture setup | cuDNN ops (cudnnUtils.h:251) |
| `tl_captureWorkspace` | `void*` | Bump-allocated workspace for capture | Capture setup/cleanup | CudaMemoryPool |
| `tl_captureWorkspaceSize` | `size_t` | Workspace total size | Capture setup | CudaMemoryPool |
| `tl_captureWorkspaceOffset` | `size_t` | Current allocation offset | CudaMemoryPool | CudaMemoryPool |
| `tl_cublasWorkspacePtr` | `void*` | cuBLAS workspace pointer | `setCublasWorkspaceForCapture/Warmup` | MmulHelper |
| `tl_cublasWorkspaceSize` | `size_t` | cuBLAS workspace size | `setCublasWorkspaceForCapture/Warmup` | MmulHelper |
| `tl_batchZeroActive` | `bool` | Suppresses individual `nullify()` calls | `setDspBatchZero()` | `DataBuffer::setToZeroBuffers()` |
| `tl_batchZeroRegistering` | `bool` | Enables batch-zero target collection | `startBatchZeroRegistration()` | `registerBatchZeroBuffer()` |

### Category 6: Per-Segment Mutable State (in GraphSegment)

| Field | Type | Purpose | Shared Across Segments? |
|---|---|---|---|
| `executionCount` | `int` | Lifecycle stage (0=warmup, 1=compile, 2+=replay) | No (per-segment) |
| `captureFailed` | `bool` | Permanent capture failure flag | No |
| `replayHandle` | `unique_ptr<GraphReplayHandle>` | Captured CUDA graph + exec + capture buffers | No |
| `cachedShapeKey` | `LongType` | Shape hash for invalidation | No |
| `argTableStable` | `bool` | Fast-replay flag (NEVER set to true currently) | No |
| `gapOpsCapturedInGraph` | `bool` | Whether gap ops are baked into the captured graph | No |
| `compiledByBackend` | `string` | Which backend compiled this segment | No |
| `shapeKey` | `LongType` | Compiled shape key | No |

### Category 7: Fallback Range Executor

| Property | Value |
|---|---|
| **Type** | `std::function<Status(int startSlot, int endSlot)>` |
| **Declared** | `TritonGraphBackend.h:104` (static thread-local) |
| **Set** | Before each segment execution (gpubackend.cpp:1608) |
| **Cleared** | RAII guard `TritonFallbackGuard` on scope exit |
| **Called by** | `TritonGraphBackend::executeSegment()` for gap ranges |

The executor lambda captures `this`, `seg`, `externalArrays`, `numExt`, `stream` — it calls `executeSegmentSlotBySlot()` for the gap range with appropriate stream synchronization.

---

## Execution Phase Model

### Phase 1: Pre-Execution Cleanup (in `execute()`)

```
flushPendingClose(stream)         — Free deferred arrays
invalidateStaleGraphs()           — Check for freed DataBuffers in slotArrayCache_
Pre-populate outputSlots from slotArrayCache_ (if frozen + hasReplayHandle)
```

### Phase 2: Per-Segment Dispatch (in `execute()` loop)

For each segment, determine execution mode:
- `useGraph && isCapturable && !captureFailed` → `executeSegmentWithGpuGraph()`
- Otherwise → `executeSegmentSlotBySlot()` (pure native path)

### Phase 3: Segment Warmup (executionCount == 0)

```
executeSegmentSlotBySlot(seg, ...)  — All ops run natively
  └── For each slot: allocate output, execute op, store in slotArrayCache_
  └── Batch-zero registration: observe which buffers get zeroed
  └── View producer detection: learn reshape/permute patterns
executionCount → 1
```

### Phase 4: Segment Compile + Capture (executionCount == 1, frozen)

```
Triton compilation:
  └── Fuse compilable ops into sub-kernels
  └── Identify non-compilable ops as fallbackRanges (gap ops)
  └── Symbolic shape matching

CUDA Graph Capture:
  1. Pre-capture warmup (slot-by-slot with capture buffers in place)
  2. Create capture buffers for PLACEHOLDER inputs + cross-segment slots
  3. Batch-zero registration (outside graph) — cudaMemsetAsync for output buffers
  4. Save warmup output arrays
  5. Set tl_graphExecutionActive=true, tl_captureWorkspace
  6. cudaStreamBeginCapture()
  7. Execute segment (Triton sub-kernels + gap ops interleaved)
     - Gap ops: CUDA events for stream sync, cuBLAS/cuDNN recorded into graph
     - gapOpsCapturedInGraph = true
  8. cudaStreamEndCapture() → cudaGraphInstantiate()
  9. SKIP initial launch (warmup results serve this pass)
  10. Restore outputSlots_ to warmup arrays
  11. Update slotArrayCache_ with capture workspace arrays
  12. Reset thread-locals (tl_captureWorkspace, etc.)
```

### Phase 5: Segment Replay

```
Pre-replay setup:
  1. DspStreamGuard(cudaStr) — set tl_dspExecutionStream
  2. Lineage validation — check directReference addresses haven't drifted
  3. Capture buffer D2D copies — fresh data into fixed-address buffers
  4. Ext input sync — syncToDevice() for PLACEHOLDER external inputs
  5. Arg table refresh — update Triton pinned host buffers with capture buffer addresses
  6. Copy consolidated arg table to device

Pre-replay zeroing:
  7. Batch-zero: cudaMemsetAsync for each batchZeroEntry (fill engines, outside graph)
  8. cuBLAS workspace zero: cudaMemsetAsync(cublasWorkspaceBuffer_, 0, 256MB)

Graph launch:
  9. Pre-launch error check (cudaPeekAtLastError)
  10. cudaGraphLaunch(graphExec, cudaStr)

Post-replay:
  11. Timed sync (30s timeout for hang detection)
  12. Diagnostic output dump (if sync OK)
  13. executionCount++, totalGraphReplays_++
  14. Gap ops: SKIP if gapOpsCapturedInGraph=true, else re-execute slot-by-slot
  15. Output slot restoration: outputSlots_[si] = slotArrayCache_[si]
```

---

## Known Issues and Risks

### 1. cuBLAS Workspace State Mismatch Between Capture and Replay

**Problem:** The cuBLAS workspace (256MB) is shared across all segments. During CUDA graph capture:

1. `setCublasWorkspaceForCapture()` binds the workspace to the cuBLAS handle and sets `cublasSetStream_v2` to the capture stream
2. Warmup cuBLAS calls have already written data into the workspace (non-zero)
3. The workspace is **NOT zeroed** before `cudaStreamBeginCapture()`
4. cuBLAS GEMM ops during capture read/write workspace with this non-zero pre-state
5. The CUDA graph bakes the kernel launches that depend on workspace content

During replay:
1. Workspace is **zeroed** via `cudaMemsetAsync` (line 1302) — different from capture-time state
2. cuBLAS handle is **NOT rebound** to the workspace (`cublasSetWorkspace` not called)
3. cuBLAS handle stream is **NOT reset** to `cudaStr` (`cublasSetStream_v2` not called)
4. The graph replays cuBLAS kernels that may expect non-zero workspace pre-state

**Evidence (seg[1320-1323] GPU hang):**
- seg[1316-1317]: 2 kernels + 3 memcpyH2D → replays fine
- seg[1320-1323]: 4 kernels + 1 memcpyH2D → **hangs** on first replay
- The fewer H2D copies in seg[1320-1323] suggest cuBLAS uses workspace for data that seg[1316-1317] passes via H2D memcpy nodes
- Between the two segments, seg[1318-1319] runs slot-by-slot (shape manipulation ops, no cuBLAS)

**Current mitigation:** Zero entire workspace before each segment replay (line 1301-1305).

**Risk:** Zeroing alone is insufficient if the captured cuBLAS kernels depend on specific non-zero workspace content from capture time. The fix may require:
- (a) Zeroing workspace before capture too (so capture and replay match), OR
- (b) Rebinding cuBLAS handle workspace+stream before each replay, OR
- (c) Snapshotting workspace content after capture and restoring before replay

### 1b. cuBLAS Handle State Not Managed During Replay

**Problem:** After capture, `restoreCublasWorkspaceAfterCapture()` resets the cuBLAS handle:
```cpp
cublasSetWorkspace(*handlePtr, nullptr, 0);  // Remove managed workspace
tl_cublasWorkspacePtr = nullptr;              // Clear thread-local
tl_cublasWorkspaceSize = 0;
```

During subsequent slot-by-slot execution (non-captured segments), cuBLAS uses its own internal allocator. The replay path does NOT re-establish the managed workspace binding before graph launch. While graph replay doesn't call cuBLAS API (it replays recorded kernel launches), the workspace memory content matters because cuBLAS kernels read configuration data from it.

**Sequence that causes the hang:**
```
1. CAPTURE seg[1320-1323]:
   - setCublasWorkspaceForCapture() → workspace bound, stream set
   - Warmup cuBLAS writes workspace (non-zero content)
   - Capture records cuBLAS kernels that read workspace content
   - restoreCublasWorkspaceAfterCapture() → workspace unbound

2. REPLAY seg[1316-1317]: OK (workspace zeroed, but this segment's cuBLAS
   kernels are less sensitive to workspace content)

3. SLOT-BY-SLOT seg[1318-1319]: shape ops only, no cuBLAS

4. REPLAY seg[1320-1323]: HANGS
   - Workspace zeroed (different from capture-time non-zero state)
   - cuBLAS handle not rebound to workspace
   - Replayed GEMM kernels read zero workspace where they expect config data
```

### 2. Batch-Zero Entry Staleness

**Problem:** `batchZeroEntries_` stores raw `void* ptr` values. If a DataBuffer is freed and the GPU address is reallocated for a different purpose, the batch-zero phase zeros the WRONG memory.

**Risk:** Primarily when Java closes output arrays between executions (e.g., `setCloseable(true); close()` in PageReuse test).

### 3. Cross-Segment Capture Buffer Address Mismatch

**Problem:** ext[341] and ext[342] always show address mismatches between capture and replay in the test output. These are PLACEHOLDER external inputs handled by capture buffers (D2D copy). The mismatch itself is expected — the capture buffer D2D copy updates the fixed-address buffer. But if the D2D copy is incorrect or incomplete, the graph reads stale data.

### 4. Gap Op Stream Synchronization During Capture

**Problem:** Gap ops may run on a different CUDA stream (gapStr) than the Triton kernels (tritonStr). CUDA events create dependency edges in the graph, but this introduces multi-stream complexity into the captured graph. On replay, the event synchronization must work correctly with the replay stream.

### 5. `argTableStable` Never Set to True

**Finding:** `argTableStable` is initialized `false` and only ever SET to `false` in the codebase. The "fast replay" path (`useFastReplay`) that skips arg table refresh and ext input sync loop is dead code — it can never trigger.

---

---

## Shared Resource Lifecycle Per Phase

This table tracks what happens to each shared resource at each execution phase.

### cuBLAS Workspace (`cublasWorkspaceBuffer_`)

| Phase | Action | State After |
|---|---|---|
| Plan construction | Not allocated | nullptr |
| Warmup (execCount=0) | `setCublasWorkspaceForWarmup()` → allocate 256MB, bind to handle | Bound to handle, contains warmup GEMM residue |
| Pre-capture | `setCublasWorkspaceForCapture()` → bind to handle + set stream | Bound to handle+stream, **NOT zeroed** (warmup residue remains) |
| During capture | cuBLAS GEMMs read/write workspace (recorded into graph) | Contains capture-time GEMM state |
| Post-capture | `restoreCublasWorkspaceAfterCapture()` → unbind from handle | **Unbound**, `tl_cublasWorkspacePtr=nullptr`, buffer still allocated |
| Slot-by-slot (non-captured seg) | cuBLAS uses internal allocator (workspace unbound) | Workspace untouched by slot-by-slot cuBLAS |
| Pre-replay | `cudaMemsetAsync(buf, 0, 256MB)` | **Zeroed** (different from capture-time state!) |
| During replay | Graph replays cuBLAS kernels that read workspace | Kernels see zeros instead of capture-time data |

**FIX NEEDED:** Either (a) zero workspace BEFORE capture too, or (b) don't zero before replay, or (c) snapshot and restore.

### Batch-Zero Entries (`batchZeroEntries_`)

| Phase | Action | State After |
|---|---|---|
| Plan construction | Empty vector | Empty |
| Warmup (execCount=0) | `startBatchZeroRegistration()` → `registerBatchZeroBuffer()` → `finishBatchZeroRegistration()` | Contains observed output buffer ptrs/sizes |
| Pre-capture | `collectBatchZeroTargets()` (if registration empty) | Gap-only or all-slot targets collected |
| Before capture begin | `cudaMemsetAsync` for each entry (outside graph) | Entries zeroed, graph will NOT contain memset nodes |
| During capture | `tl_batchZeroActive=true` suppresses individual `nullify()` calls | No memset recorded in graph |
| Pre-replay | `cudaMemsetAsync` for each entry (outside graph) | Output buffers zeroed before graph launch |
| During replay | Graph kernels write to output buffers (already zeroed) | Correct output |

**RISK:** `void* ptr` in entries may be stale if DataBuffer was freed between executions.

### Capture Buffers (per-segment, in `GraphReplayHandle`)

| Phase | Action | State After |
|---|---|---|
| Plan construction | nullptr (no replay handle) | No capture buffers |
| Pre-capture | Allocate buffers for PLACEHOLDER ext inputs + cross-segment slots | Fixed-address buffers created |
| During capture | Slots swapped to capture buffer arrays → graph records these addresses | Graph bakes capture buffer GPU addresses |
| Post-capture | Restore original outputSlots_ (slots swapped back) | Capture buffers persist in replay handle |
| Pre-replay (D2D) | `copyIntoCaptureBuffer()` for each non-directReference buffer | Fresh data copied into fixed-address buffers |
| During replay | Graph reads from capture buffer addresses (now updated via D2D) | Correct data at correct addresses |

### Thread-Local State

| Phase | `tl_graphExecutionActive` | `tl_dspExecutionStream` | `tl_captureWorkspace` | `tl_graphCaptureStream` |
|---|---|---|---|---|
| Plan construction | false | nullptr | nullptr | nullptr |
| Warmup | false | set by DspStreamGuard | nullptr | nullptr |
| Pre-capture warmup | false | set by DspStreamGuard | set (32MB) | nullptr |
| Capture begin | **true** | set | set | set to cudaStr |
| During capture (gap ops) | **true** (stays true) | set | set | set |
| Capture end | false | set | **cleared** | **restored** |
| Post-capture | false | restored by DspStreamGuard | nullptr | previous value |
| Slot-by-slot (non-captured) | false | may be set | nullptr | nullptr |
| Replay | false | set by DspStreamGuard | nullptr | nullptr |

**RISK:** If `tl_graphExecutionActive` is not correctly reset after capture failure, subsequent ops may try to allocate from non-existent capture workspace.

### Output Slots and Cache

| Phase | `outputSlots_[si]` | `slotArrayCache_[si]` |
|---|---|---|
| Plan construction | nullptr | nullptr |
| Warmup | Set to op output array | Set to same op output array |
| Pre-capture | May be nullified by cleanup | Preserved (cleanup protects replay-managed slots) |
| During capture | Set to capture workspace arrays | Updated post-capture to capture workspace arrays |
| Post-capture | Restored to warmup arrays | Points to capture workspace arrays |
| Pre-replay | May have warmup arrays OR cache arrays | Points to capture workspace arrays |
| Post-replay | **Restored** to `slotArrayCache_[si]` | Unchanged (still capture workspace arrays) |

**CRITICAL:** Post-replay restoration (line 1546) is essential — without it, downstream segments read stale warmup data instead of fresh replay output.

### Segment Execution Count

| Phase | `seg.executionCount` | Effect |
|---|---|---|
| Plan construction | 0 | Warmup mode |
| After warmup | 1 (or 2 if frozen+already compiled) | Compile/capture candidate |
| After capture | Not incremented (SKIP initial launch) | Stays at capture-time value |
| After replay | `++` | Increments each replay |
| Shapes not frozen: end of execute | **Reset to 0** (line 1357 in NativeDynamicShapePlan.cpp) | Forces re-warmup next execution |
| Shapes frozen: end of execute | Preserved | Replay continues |

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `dspBatchZero` | false | Enable batch-zero output buffer zeroing |
| `dspBatchZeroKernel` | false | Enable batch-zero via custom kernel (vs cudaMemsetAsync) |
| `tritonCaptureMinExec` | 2 | Minimum execution count before capture attempted |
| `tritonAllowFallbackCapture` | true | Allow gap ops during CUDA graph capture |
| `tritonGraphReinstantiate` | false | Re-instantiate graphExec from template before each replay |
| `tritonForceRecapture` | false | Force re-capture every step (diagnostic mode) |
| `tritonVerifyKernels` | false | Run native + Triton and compare outputs |
| `shapesFrozen` | false | Skip shape invalidation checks |
| `dspFreezeRecompile` | false | Force recompilation after freeze |

---

## Files

### Core Execution
| File | Purpose |
|---|---|
| `libnd4j/include/graph/NativeDynamicShapePlan.h` | Plan structure, GraphSegment, all shared resource declarations |
| `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp` | Main `execute()` loop, segment iteration, cleanup, frozen logic |
| `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cpp` | `executeSegmentWithGpuGraph()` — warmup, capture, replay, gap orchestration |
| `libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp` | `executeSegmentSlotBySlot()` — native per-op execution |

### GPU-Specific
| File | Purpose |
|---|---|
| `libnd4j/include/graph/impl/NativeDynamicShapePlan_cublas.cu` | cuBLAS workspace allocation, stream binding, restore |
| `libnd4j/include/graph/impl/NativeDynamicShapePlan_batchzero.cu` | Batch-zero registration, kernel launch, device arrays |
| `libnd4j/include/graph/impl/NativeDynamicShapePlan_cuda.cu` | CUDA-specific capture/replay handle management |
| `libnd4j/include/graph/impl/NativeDynamicShapePlan_batchgemm.cu` | Batched GEMM group detection and execution |

### Triton Backend
| File | Purpose |
|---|---|
| `libnd4j/include/graph/gpu/TritonGraphBackend.h` | Backend interface, CompiledSegment, fallbackRanges |
| `libnd4j/include/graph/gpu/TritonGraphBackend_compile.cu` | Compilation, sub-kernel creation, gap identification |
| `libnd4j/include/graph/gpu/TritonGraphBackend_execute.cu` | Sub-kernel + gap interleaved execution, `getGapSlots()` |

### Support
| File | Purpose |
|---|---|
| `libnd4j/include/graph/DspStreamGuard.h` | RAII guard for `tl_dspExecutionStream` |
| `libnd4j/include/array/DataBuffer.h` | Thread-local declarations for capture state |
| `libnd4j/include/helpers/DebugHelper.h` | `tl_graphExecutionActive` declaration |

---

## Related Decisions

- **ADR-OpTimingTracker**: Complementary per-op timing for profiling
- **Triton Graph Backend**: Kernel fusion for elementwise + reduction ops
- **Dynamic Shape Plan Executor**: Java-side orchestration layer

## References

- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [cuBLAS Workspace Management](https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace)

## Authors

- Implementation: deeplearning4j team
- Gap ops & shared resources audit: 2026-03-30
