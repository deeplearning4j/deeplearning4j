# ND4J CUDA segfault investigation (nd4j-api, nd4j-cuda, libnd4j)

Scope
- Looked only at nd4j-api, nd4j-cuda, libnd4j (no deeplearning4j-nn, no code changes).
- Focused on op execution, CUDA backend, opaque buffer/array lifecycle, multithreading, and multi-device.

Spring server threading implications
- Spring uses pooled request threads; device affinity is per thread, so different request threads can map to different GPUs when multiple GPUs are visible. If arrays are cached in singletons/services and reused across requests, they can be accessed on a different thread/device than they were created on.
- Async execution (@Async, CompletableFuture, Reactor/WebFlux) can hop threads; CudaZeroHandler keeps a ThreadLocal CudaContext and native AffinityManager is thread-local, so cross-thread array use is a high-risk path for device mismatch.
- OpExecutionDelegator's transferExecutor runs in its own daemon threads; prefetch/ensureAvailableOn uses HybridDataBuffer without switching device, so transfer threads can sync/allocate on the wrong device relative to the target.
- Workspaces are thread-local; arrays produced inside a workspace scope and used on a different thread after the scope ends are at risk for use-after-free or reuse.
- DeallocatorService threads run with fixed device IDs; if targetDevice is stale due to cross-thread/device use, deallocation syncs the wrong device.

Given your environment (Spring MVC + 2 GPUs + cached native ndarray references)
- With 2 GPUs visible, native AffinityManager will distribute request threads across devices by default; cached native NDArray pointers can be used on a different device/thread than they were created on, even if you never intended multi-GPU.
  - libnd4j/include/execution/cuda/AffinityManager.cu:34
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/jita/concurrency/CudaAffinityManager.java:69
- If you cache OpaqueNDArray or OpaqueDataBuffer objects directly, their deallocators pin a targetDevice at creation time. Reusing those objects across threads/devices risks wrong-device sync on delete or stale pointers after underlying INDArray changes.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/OpaqueNDArray.java:110-170
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/OpaqueDataBuffer.java:120-170
- If you cache INDArray but call getOrCreateOpaqueNDArray() per request, the cached OpaqueNDArray is cleared by BaseNDArray.clearOpaqueNDArray when shape/data changes, and a new OpaqueNDArray is created; any previously cached native pointers become invalid.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ndarray/BaseNDArray.java:6541-6567

High-risk patterns (most likely to explain multithread-only segfaults)

1) Thread/device affinity mismatch -> cross-device pointer use
- C++ affinity assignment is per thread and round-robin across *all* GPUs. If more than one GPU is present, different threads will be on different devices by default even if the user did not explicitly enable multi-GPU. This is the single biggest multiplier for multithread crashes.
  - libnd4j/include/execution/cuda/AffinityManager.cu:34 (currentDeviceId assigns device per thread), 94 (setCurrentDevice syncs and resets buffers)
- Java CudaAffinityManager reports current device by querying native getDevice (which uses AffinityManager::currentDeviceId), but does not enforce a single device for all threads by default.
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/jita/concurrency/CudaAffinityManager.java:69, 300
- AtomicAllocator/CudaZeroHandler return device pointers without checking that the current thread device matches the buffer device. If thread device differs from buffer device, kernels get pointers to memory from another GPU.
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/jita/handler/impl/CudaZeroHandler.java:492
- FlowController only conditionally relocates; relocateObject() is currently disabled (throws), so cross-device mismatch can silently persist when cross-device access is allowed or if device mismatch is not detected.
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/jita/handler/impl/CudaZeroHandler.java:579
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/jita/flow/impl/SynchronousFlowController.java:79-146
- Device-aware op routing does not switch the current thread device. It only attempts a logical transfer using HybridDataBuffer, then executes on the current thread device anyway.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DeviceAwareOpExecutioner.java:360-401, 730-781

Why this matters: if threads are on different GPUs, arrays created on GPU0 are routinely used on GPU1 without proper migration. That is consistent with single-thread OK, multithread crash.

2) HybridDataBuffer multi-device logic is incomplete
- BaseCudaDataBuffer tracks a single gpuValid flag and a single ownerDevice. There is no per-device validity map. Switching devices can mark GPU data as valid even if it is on a different GPU, which can cause stale/invalid device pointers to be used.
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/buffer/BaseCudaDataBuffer.java:90, 2038-2104
- ensureAvailableOn() calls syncToSpecial() but does not update the native DataBuffer deviceId. syncToSpecial() uses the *current* CUDA device without ensuring it matches the buffer device, so the physical allocation and the deviceId can diverge.
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/buffer/BaseCudaDataBuffer.java:2073
  - libnd4j/include/array/cuda/DataBuffer.cu:446 (syncToSpecial), 634 (migrate)
- getDeviceAddress() returns the same special pointer for any GPU device descriptor. That pointer is only valid on the GPU it was allocated on.
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/buffer/BaseCudaDataBuffer.java:2100
- allocateOnDevice() is effectively a no-op; multi-backend workspace hooks are present but not implemented.
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/buffer/BaseCudaDataBuffer.java:2132

Why this matters: Device-aware routing and HybridDataBuffer are currently unsafe for multiple GPUs. Even if you are not “actively” using multi-GPU, thread-based device assignment can implicitly create multi-GPU usage.

3) Deallocator/GC thread device mismatches + stale deviceId
- DeallocatorService selects a queue based on deallocatable.targetDevice(), which is captured at creation time from the current thread. If buffers/arrays migrate to other devices, targetDevice becomes stale.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/DeallocatorService.java:318-339, 360-380
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/OpaqueDataBuffer.java:120-160
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/OpaqueNDArray.java:110-170
- OpaqueDataBufferDeallocator/OpaqueNDArrayDeallocator use targetDevice for setDevice + commit before calling dbClose/deleteNDArray. If targetDevice is stale, they sync the wrong device.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/OpaqueDataBufferDeallocator.java:55-110
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/OpaqueNDArrayDeallocator.java:55-115
- Native dbClose/deleteNDArray set device using buffer deviceId. If deviceId is stale (because migration did not update it), then cudaDeviceSynchronize is on the wrong device and memory can be freed while kernels on another device still use it.
  - libnd4j/include/legacy/cuda/NativeOpsHelpers_DataBuffers_close.cu:131-175
  - libnd4j/include/legacy/cuda/NativeOpsHelpers_Arrays_delete.cu:90-125

4) Context/fastpath cleanup uses cudaDeviceSynchronize without device switching
- Context destructor and clearFastPath always call cudaDeviceSynchronize on the current device, but do not track the context's device. If the deallocation thread is on a different device, the sync is wrong.
  - libnd4j/include/graph/impl/Context.cpp:135, 899
- CudaOpContext.close() and purge() do not switch devices before ctxPurge/deleteGraphContext. (CudaOpContextDeallocator does, but manual close/purge does not.)
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/ops/executioner/CudaOpContext.java:85, 336

5) Ephemeral dimension arrays used in async ops (GC/UAF risk)
- Some CUDA exec paths allocate temporary dimension arrays via op.dimensions().castTo(LONG) and pass them to native ops without explicit synchronization. If kernels are async, those temp arrays can be GC’d while kernels still read them.
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/ops/executioner/CudaExecutioner.java:116, 551, 672

Medium-risk patterns / supporting issues

6) Device-aware caches are keyed by current thread device, not by array device
- Shape/TAD caches are device-aware but use AffinityManager::currentDeviceId (thread-local). If a thread's device does not match an array's device, cached special pointers for the wrong device can be used.
  - libnd4j/include/helpers/impl/DirectShapeTrie.cpp:95-133
  - libnd4j/include/helpers/impl/DirectTadTrie.cpp:40-83

7) LaunchContext ignores external stream pointers
- LaunchContext::getCudaStream always uses thread-local ContextBuffers and ignores the external stream pointer passed from Java. If Java expects a specific stream to be used (e.g., for op contexts), this can cause stream mismatch and synchronization gaps.
  - libnd4j/include/execution/cuda/LaunchContext.cu:168-181

8) ContextBuffers reinitialize streams when device changes
- ContextBuffers release/recreate streams when device changes. Any cached Java stream pointer (CudaContext) can become stale if not refreshed, leading to invalid stream handles.
  - libnd4j/include/execution/cuda/ContextBuffers.cu:171-317

9) Workspace/thread-local memory reuse risks across threads
- Workspaces are thread-local; arrays created in a workspace can be reused/freed when that thread exits its scope. Passing workspace arrays to other threads (common in async pipelines) can result in use-after-free or reuse while kernels are still running.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/provider/BasicWorkspaceManager.java:46-120
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/abstracts/Nd4jWorkspace.java:155-158
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/jita/workspace/CudaWorkspace.java:422-432

10) Device-aware ND4J helpers rely on HybridDataBuffer only
- DeviceAwareNd4j.ensureOnDevice uses HybridDataBuffer.ensureAvailableOn without device switching or migration. This can silently mark data as on a device without actually migrating it.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/DeviceAwareNd4j.java:477

Isolation checks (not fixes; useful to narrow root cause)

A) Force single-GPU per process/thread to validate root cause
- If multiple GPUs are present, the native AffinityManager will round-robin threads by default. Consider temporarily forcing all threads to a single device (or update AffinityManager to honor a configured device list) to see if the crash disappears.

B) Make device-aware execution actually set the thread device
- Wrap DeviceAwareOpExecutioner execution with Nd4j.getAffinityManager().unsafeSetDevice(targetDevice) for the duration of the op.
- Alternatively, disable device routing in multithreaded mode unless a full per-device migration system exists.

C) Implement real multi-device tracking in HybridDataBuffer
- Track validity per device (map of device -> valid), not a single gpuValid flag.
- ensureAvailableOn() should call native migrate (or equivalent) and update native deviceId.
- getDeviceAddress() should be device-specific or throw if the buffer is not allocated on that device.

D) Fix context cleanup device selection
- Context::clearFastPath and Context destructor should switch to the context's device before cudaDeviceSynchronize.
- CudaOpContext.close/purge should set deviceId to the context's device before calling ctxPurge/deleteGraphContext.

E) Protect ephemeral dimension arrays in CUDA exec paths
- Keep dimension arrays alive until kernel completion (store in OpContext or keep a strong reference list), or explicitly synchronize the device after ops that create temporary dimension arrays.

F) Deallocation should use the *actual* buffer device, not the creation device
- Opaque* deallocators should query the current deviceId from the native buffer/array at deallocation time (if available) and/or track per-buffer device usage. If cross-device access is used (P2P), sync all relevant devices before freeing.

G) Align native AffinityManager with Java configuration
- C++ AffinityManager uses cudaGetDeviceCount and ignores Java's availableDevices/forcedSingleGPU. Consider enforcing the configured device set in the native layer or expose a native API to set the allowed device list.

H) Add guardrails for workspace cross-thread use
- Optional: validate threadId/deviceId on workspace access, or detach arrays before cross-thread use.

What’s missing to make this correctly multi-thread + multi-device (proper fix path)

1) A single, explicit device‑selection contract shared by Java + native
- Today, native `AffinityManager::currentDeviceId()` assigns devices per thread and ignores Java’s allowed device list; Java `CudaAffinityManager` relies on native `getDevice()` which triggers that round‑robin assignment.
  - libnd4j/include/execution/cuda/AffinityManager.cu:24-66
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/jita/concurrency/CudaAffinityManager.java:59-121
- Proper fix: require explicit device selection on thread entry, or push Java’s configured device list into native (set allowed devices / per‑thread device) so native never “picks” a device on first use.

2) Device-aware ops must set the actual device (not just “logical routing”)
- Device-aware execution chooses a target device but does not call `unsafeSetDevice()` before executing; it relies on HybridDataBuffer syncs that don’t migrate.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DeviceAwareOpExecutioner.java:360-401
- Proper fix: set thread device to target before prepare/execute, restore after; or require ops execute on the device that owns inputs (and route accordingly).

3) Real multi-device buffer semantics (per-device validity + migration)
- Java `BaseCudaDataBuffer` tracks a single `gpuValid` flag and a single `ownerDevice`; it uses `syncToSpecial()` (host->device) rather than true migration, and does not update native `deviceId`.
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/buffer/BaseCudaDataBuffer.java:2038-2140
- Native `DataBuffer::migrate()` exists and updates deviceId, but Java paths do not call it.
  - libnd4j/include/array/cuda/DataBuffer.cu:700-760
- Proper fix: Java `ensureAvailableOn()` must call native migrate when device changes, and `getDeviceAddress()` should be device-specific (or throw if not present). Track validity per device (or multi-buffer strategy).

4) Cross-device access requires explicit policy + correct P2P semantics
- `allowCrossDeviceAccess` is treated as “ok to use pointer from another device”, but `relocateObject()` and `promoteObject()` are disabled (“Pew-pew”), so no safe fallback exists when P2P is off or incomplete.
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/jita/handler/impl/CudaZeroHandler.java:579-690
- Proper fix: implement relocation paths or enforce “run on owner device only” unless P2P is confirmed for the device pair; use `cudaMemcpyPeer` (or DataTransferManager) for cross-device migration.

5) Make deallocation device-correct even after migration
- Opaque deallocators use a device ID captured at creation time, which can be stale after migration. Native deallocation relies on buffer deviceId, but Java may not keep that deviceId accurate.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/OpaqueDataBufferDeallocator.java
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/OpaqueNDArrayDeallocator.java
  - libnd4j/include/legacy/cuda/NativeOpsHelpers_DataBuffers_close.cu
- Proper fix: query actual deviceId from native at deallocation time; ensure migrations update deviceId; sync the correct device/stream before free.

6) Make buffer migration thread-safe for shared arrays
- `DataBuffer::migrate()` updates `_specialBuffer` without locks; concurrent ops on the same array can race, leading to UAF/double free or stale pointers.
  - libnd4j/include/array/cuda/DataBuffer.cu:700-760
- Proper fix: add per-buffer locking or reference-counted “in‑flight” migration state; ensure op execution serializes migration for shared arrays.

7) Cache keys should use the array’s device, not the thread device
- `DirectShapeTrie`/`DirectTadTrie` are keyed by `AffinityManager::currentDeviceId()` (thread device), not by buffer device. If a thread device differs from array device, cached device pointers can be wrong.
  - libnd4j/include/helpers/impl/DirectShapeTrie.cpp:95-133
  - libnd4j/include/helpers/impl/DirectTadTrie.cpp:40-83
- Proper fix: include buffer deviceId in cache keys or require device switch before use.

8) Stream/context ownership should be consistent across Java + native
- Native `LaunchContext::getCudaStream` ignores external stream pointers, while Java caches stream pointers in `CudaContext`. ContextBuffers re-create streams when device changes.
  - libnd4j/include/execution/cuda/LaunchContext.cu:168-181
  - libnd4j/include/execution/cuda/ContextBuffers.cu:171-317
- Proper fix: either let Java supply the stream and honor it in native, or make Java treat streams as ephemeral and always query native per op.

9) OpaqueNDArray caching is not safe without reference counting
- OpaqueNDArray is a raw `sd::NDArray*`. If Java caches it and the underlying `INDArray` changes/clears the opaque pointer, you can get use‑after‑free.
  - libnd4j/include/legacy/NativeOps.h:54-66
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ndarray/BaseNDArray.java:6541-6567
- Proper fix: make opaque handles ref‑counted (shared_ptr) or versioned/invalidated with explicit checks; do not expose raw NDArray* across threads without lifetime tracking.

10) Unify multi-device transfer infra with ND4J execution path
- libnd4j has `DataTransferManager`/`DeviceManager` for P2P/transfer scheduling, but ND4J op execution doesn’t use it. This leaves migrations ad‑hoc and partially implemented.
  - libnd4j/include/execution/DataTransferManager.h
- Proper fix: route device transfers through `DataTransferManager` and align Java execution with native transfer scheduling.

Trace-specific: concat pointer-like scalar (strided_slice pipeline)

Observed error
- `concat` throws: “Input scalar at index 2 contains a value that looks like a pointer address …” which is produced by the pointer‑detection block in concat.
  - libnd4j/include/ops/declarable/generic/transforms/concat.cpp
- Input shapes in the log match scalar shapeInfo `[0,0,1,262145,1,99]` (rank‑0 scalar), so the *shape* is valid; the *data* is corrupted/stale.
  - libnd4j/include/helpers/impl/ShapeBuilders.cpp (scalar shapeInfo layout)

Why strided_slice is a plausible upstream source
- StridedSlice reads begin/end/stride arrays via host reads (`asVectorT`, `e<LongType>`), which depend on `syncToHost` and DataBuffer actuality flags.
  - libnd4j/include/ops/declarable/generic/tensor/strided_slice.cpp
  - libnd4j/include/array/NDArray.hXX (asVectorT/getBufferAsVector + e())
  - libnd4j/include/array/cuda/NDArray.cu (syncToHost -> DataBuffer::syncToPrimary)
- If begin/end/stride arrays were produced by earlier ops and reused from cache, stale host buffers can feed incorrect indices, producing corrupted scalar outputs that later hit concat’s pointer check.

Cache reuse can cause “host looks actual” even when it isn’t
- Cached INDArray reuse does NOT reset native DataBuffer counters. If a previous use left `_writePrimary > _writeSpecial`, then `isPrimaryActual()` stays true after reuse.
- `syncToHost()` returns early when `isPrimaryActual()` is true, so host reads can return stale data (including pointer-like values).
  - libnd4j/include/array/cuda/DataBuffer.cu (syncToPrimary early‑return)
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/memory/ArrayCacheMemoryMgr.java (array reuse without buffer reset)
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ndarray/BaseNDArray.java:6663 (assignNewId clears OpaqueNDArray but not DataBuffer counters)
- This fits the symptom: concat reads a scalar via `input->e<LongType>(0)` (host), gets stale host buffer containing pointer junk.
  - libnd4j/include/array/NDArray.hXX (e() uses preparePrimaryUse)

Multi-threaded SameDiff inference can free or reuse arrays early
- `InferenceSession` mutates shared state (`arrayUseTracker`, `freedArrays`, `dagCache`) per `output()` call; it is not thread‑safe. Concurrent calls can clear trackers and release arrays still in use by another request.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/InferenceSession.java
- Result: use‑after‑free at native level; scalar buffer reads can become pointer‑like values.

Other op‑specific hot spots for the same corruption pattern
- `reshape` and `broadcast_to` read shape/scalar inputs from host and already include pointer‑value detection; if concat consumes corrupted scalars, reshape will trip too.
  - libnd4j/include/ops/declarable/generic/shape/reshape.cpp
  - libnd4j/include/ops/declarable/generic/shape/broadcast_to.cpp
- `size_at` writes scalar via `p()` (host write only). If the scalar array is reused with stale device flags, later device use can see inconsistent state.
  - libnd4j/include/ops/declarable/generic/shape/size_at.cpp

Most likely missing pieces for this specific failure
- Reset/normalize DataBuffer actuality counters on INDArray reuse (or force sync on `syncToHost` for scalars used in shape/concat).
- Ensure SameDiff inference is isolated per thread (or add synchronization + per‑thread trackers).
- Enforce device consistency when reading scalars on GPU: `syncToHost` should validate `_deviceId` vs current device and update streams/contexts correctly when devices differ.

Notes
- The highest-likelihood crash path in multithreaded environments with multiple GPUs present is a device affinity mismatch leading to cross-device pointer use and incorrect deallocation synchronization.
- compute-sanitizer not catching the issue aligns with host-side pointer misuse (wrong device frees, stale opaque pointers, Java GC/deallocator timing).

New crash signature: cast + dbClose + JVM segfault
- Latest crash shows op execution completed (cast printed correct values) followed by immediate SIGSEGV in JVM with `dbClose: deallocating buffer ...` log lines.
- This pattern strongly suggests premature native buffer deallocation (use-after-free) on the host side, not an op-specific kernel bug.
  - libnd4j/include/legacy/cuda/NativeOpsHelpers_DataBuffers_close.cu
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/buffer/BaseDataBuffer.java

View output lifecycle looks unsafe (becomes fatal on CUDA + multithreading)
- `CustomOp.initializeOutputs` creates view outputs when C++ shape functions set `ARRAY_COPY_OFFSET_INPUT_X`, but the Java side does **not** mark these outputs as views.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/CustomOp.java
- `Nd4j.create(input.data(), shape, strides, offset, ordering)` does not set the view flag; the output is treated as closeable even though it shares the buffer.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/Nd4j.java
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ndarray/BaseNDArray.java
- `ArrayCacheMemoryMgr.release` attempts to close non-closeable arrays when `useCount==1`. Native `useCount` is stubbed to always return 1, so view arrays get closed anyway, freeing shared buffers still in use.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/memory/ArrayCacheMemoryMgr.java
  - libnd4j/include/array/impl/InteropDataBuffer.cpp (useCount returns 1)
- Result: base arrays can be released while view outputs still reference the same DataBuffer, yielding UAF and "pointer-as-scalar" corruption.
- This may not reproduce on CPU due to synchronous execution and deterministic lifetimes; CUDA async + multi-thread dealloc makes it visible.

Op-specific context (strided_slice + create_view + concat + cast pipeline)
- `SDVariable.get(...)` builds dynamic begin/end/stride arrays using `concat` of scalar ops (`sizeAt`, `constant`, `point`, `interval`), then calls `strided_slice`.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/SDVariable.java
- `CreateView.createPoint` and `createInterval` build index tensors via `concat` and `cast` on scalar values.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/shape/CreateView.java
- If any scalar buffer is freed/reused (view lifecycle bug), `concat` detects pointer-like values and `cast` can execute with corrupted buffers, matching the observed crash sequence.

Missing for correctness in CUDA multithreaded execution (keep simple, Java‑owned lifecycle)
1) Fix device affinity to avoid implicit multi‑GPU per thread
   - Native `AffinityManager::currentDeviceId()` assigns devices per thread (round‑robin) and ignores Java’s device list.
   - In a Spring thread pool, this makes different request threads land on different GPUs by default even if you never intended multi‑GPU.
   - Enforce a single device per process (or explicit per‑request device), and make native honor the configured device set.
     - libnd4j/include/execution/cuda/AffinityManager.cu
     - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/jita/concurrency/CudaAffinityManager.java
2) Always free/sync on the actual buffer device and stream
   - `dbClose` switches device based on buffer deviceId; if deviceId is stale (migration not updating it), sync/free can happen on the wrong device.
   - Deallocator threads and Context cleanup should set the correct device/stream for each buffer/context before sync/free.
     - libnd4j/include/legacy/cuda/NativeOpsHelpers_DataBuffers_close.cu
     - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/OpaqueDataBufferDeallocator.java
     - libnd4j/include/graph/impl/Context.cpp
3) Make device‑aware execution actually switch devices (not just “logical routing”)
   - Device‑aware execution picks a target device but does not switch the current thread device before op execution.
   - With multiple GPUs and thread pools, this means kernels run on whatever device the thread happened to get.
     - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DeviceAwareOpExecutioner.java
4) Keep CUDA buffer state consistent across devices
   - `BaseCudaDataBuffer` tracks only one `gpuValid` flag and one `ownerDevice`.
   - In multi‑GPU or thread‑hopping scenarios, this leads to stale device pointers and wrong‑device reads.
     - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/buffer/BaseCudaDataBuffer.java
5) Ensure stream/context alignment
   - Native `LaunchContext` ignores external stream pointers; `ContextBuffers` can recreate streams on device changes.
   - In multithreaded CUDA execution, mismatched streams reduce synchronization guarantees and can free buffers while kernels still run.
     - libnd4j/include/execution/cuda/LaunchContext.cu
     - libnd4j/include/execution/cuda/ContextBuffers.cu

Why this still explains the cast crash
- The cast op itself looks correct; the crash occurs after successful execution, during `dbClose`, consistent with memory being freed while still referenced elsewhere.
- The CUDA‑specific device/stream/deallocator issues above create the exact “free on wrong device/stream while still in use” pattern, and multithreading makes it far more likely.

New trace: cast datatype validation failure (output INT32 not in input types)

Observed error
- `cast` fails in native validation: “Op: [cast] failed check for output [0], DataType: [INT32] - not found in input types”.
  - This exact message comes from `DeclarableOp::validateDataTypes` when `_descriptor->isInherit(index)` is true.
  - libnd4j/include/ops/declarable/impl/DeclarableOp.cpp (validateDataTypes, inherit branch)

Why this is unexpected for cast
- `cast` explicitly allows ANY output type, so it should **not** run the “inherit output type from inputs” check.
  - libnd4j/include/ops/declarable/generic/datatypes/cast.cpp (DECLARE_TYPES sets allowed input/output to ANY)
- `OpDescriptor::isInherit()` only returns true if the allowed output type list contains INHERIT.
  - libnd4j/include/ops/declarable/impl/OpDescriptor.cpp (isInherit)

Likely causes (host‑side, multi‑thread amplified)
- OpDescriptor corruption or overwrite: for `isInherit` to be true on `cast`, the descriptor’s allowed output types must have been mutated (or memory‑scribbled) to include INHERIT. That points to host‑side memory corruption rather than a kernel bug.
- Stale/wrong output array pointer in OpContext: if `fastpath_out` contains a wrong/old INDArray (e.g., reused across threads), the dtype read from its shapeInfo can be unrelated to the input arrays, triggering the inherit check even when inputs are correct.
  - libnd4j/include/ops/declarable/impl/DeclarableOp.cpp (fastpath_out validation uses array->dataType())
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/InferenceSession.java (per‑op OpContext wiring)
- OpContext populated from stale op arguments (alternate exec path): `DefaultOpExecutioner.initOpContext` uses `op.inputArguments()`/`op.outputArguments()` (not the per‑execution arrays). If a shared `DynamicCustomOp` instance is reused across threads, those lists can be stale and feed the wrong outputs into native validation.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DefaultOpExecutioner.java
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/DynamicCustomOp.java (mutable input/output argument lists)

Secondary smell: cast IArg count mismatch
- `DECLARE_CUSTOM_OP(cast, ..., IARGS=1)` but `CUSTOM_OP_IMPL(cast, ..., IARGS=-2)` (variadic) don’t match.
  - libnd4j/include/ops/declarable/headers/datatypes.h
  - libnd4j/include/ops/declarable/generic/datatypes/cast.cpp
- This shouldn’t directly cause the inherit‑type failure, but it’s a registration inconsistency worth correcting once the memory‑safety issues are resolved.

Implication
- This error is strong evidence of **host‑side state corruption** (descriptor or op‑context inputs/outputs), consistent with the earlier pointer‑as‑scalar failures and CUDA‑only multithread crashes.

1) Call paths to DefaultOpExecutioner.initOpContext (stale input/output risk)

Where initOpContext is called
- CUDA path: `CudaExecutioner.exec(CustomOp op)` calls `op.setupOpContextFromCustomOp(context)` → `op.initializeOutputs(context)` → `DefaultOpExecutioner.initOpContext(op, shapeOverride, context)` → `exec(op, context)`.
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/ops/executioner/CudaExecutioner.java
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DefaultOpExecutioner.java
- CPU path mirrors the same call order in `NativeOpExecutioner.exec(CustomOp op)` (not your target, but the same API shape exists).
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cpu-backend-common/src/main/java/org/nd4j/linalg/cpu/nativecpu/ops/NativeOpExecutioner.java
- Device‑aware execution does **not** bypass this: `DeviceAwareOpExecutioner.exec(CustomOp op)` delegates to the target executioner’s `exec(op)` (CUDA) which calls `initOpContext`.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DeviceAwareOpExecutioner.java
- Multi‑backend execution also delegates to `executioner.exec(op)` (i.e., CUDA) and therefore uses `initOpContext`.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DefaultMultiBackendExecutioner.java

Where initOpContext is *not* used (safe path)
- `Nd4j.exec(CustomOp op, OpContext ctx)` delegates to `exec(op, ctx)`; the context arrays are used directly and `initOpContext` is not invoked.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/Nd4j.java
  - nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/ops/executioner/CudaExecutioner.java
- SameDiff `InferenceSession.executeCustomOp` builds an `OpContext`, sets input/output arrays, and calls `Nd4j.exec(dynOp, opContext)` (bypassing `initOpContext`), which is *good*.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/InferenceSession.java

Why initOpContext is risky in multithreaded SameDiff
- `DefaultOpExecutioner.initOpContext` resets inputs/outputs using `op.inputArguments()` and `op.outputArguments()` (shared, mutable lists on `DynamicCustomOp`).
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DefaultOpExecutioner.java
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/DynamicCustomOp.java
- If the same `DynamicCustomOp` instance is executed on multiple threads (e.g., SameDiff shared across requests), those lists can be mutated concurrently:
  - Thread A sets input/output arrays for op X; Thread B mutates them for op Y; `initOpContext` in thread A now picks B’s arrays.
  - This can feed `fastpath_out` with a stale/incorrect INDArray, producing dtype validation errors (including the cast inherit‑type error).
- This is fully consistent with "single‑thread OK, multithread crash."

2) Ops that explicitly use INHERIT output types (audit)

Ops with INHERIT output types (static in registerTypes)
- Transforms: `standardize`, `reverse`, `reverse_sequence`
  - libnd4j/include/ops/declarable/generic/transforms/standardize.cpp
  - libnd4j/include/ops/declarable/generic/transforms/reverse.cpp
  - libnd4j/include/ops/declarable/generic/transforms/reverseSequence.cpp
- Boolean: `select` (output index 1 inherits)
  - libnd4j/include/ops/declarable/generic/boolean/select.cpp
- Linalg: `sufficient_statistics` (all 3 outputs inherit)
  - libnd4j/include/ops/declarable/generic/linalg/sufficient_statistics.cpp
- Parity ops: `listdiff` (output 0 inherits)
  - libnd4j/include/ops/declarable/generic/parity_ops/listdiff.cpp
- Broadcastable math: `divide`, `divide_no_nan`, `truncatediv`, `floordiv`, `floormod`, `mod`, `atan2`,
  `subtract`, `reverse_subtract`, `squared_subtract`, `multiply`, `minimum`, `maximum`,
  `reverse_divide`, `reverse_mod`, `boolean_and`, `boolean_or`, `boolean_xor`, `percentile`
  - libnd4j/include/ops/declarable/generic/broadcastable/*.cpp
- Tensor/shape: `meshgrid` (all outputs inherit, same‑mode)
  - libnd4j/include/ops/declarable/generic/broadcastable/meshgrid.cpp
- NN convolution helpers: `im2col`, `col2im`
  - libnd4j/include/ops/declarable/generic/nn/convo/im2col.cpp
  - libnd4j/include/ops/declarable/generic/nn/convo/col2im.cpp

Audit result
- All INHERIT output types are set in `DECLARE_TYPES(...)` (registerTypes) and appear static; no runtime mutation was found in op implementations.
- Therefore, a *cast* op tripping the inherit‑type check strongly implies **descriptor corruption or stale op‑context arrays**, not a legitimate type rule.
- If other ops with INHERIT also start failing with “not found in input types,” that’s consistent with stale output arrays in `fastpath_out` or shared `DynamicCustomOp` mutation (not an op‑specific kernel bug).

3) Where Nd4j.exec(CustomOp op) is used (no OpContext) vs exec(op, ctx)

Within nd4j-api (non-SameDiff utility paths)
- Many convenience APIs call `Nd4j.exec(op)` directly and allocate outputs internally. These typically create a *new* op object per call, so they are not inherently unsafe unless the op is cached/reused across threads.
  - NDMath/NDLinalg/NDImage wrappers create new ops and call `Nd4j.exec(op)`.
  - BaseNDArray arithmetic helpers call `Nd4j.exec(op)` with freshly created op instances.
  - Loss/Updater/Activation helpers call `Nd4j.exec(op)` with new op instances.
  - Examples:
    - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/ops/transforms/Transforms.java
    - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/ops/NDMath.java
    - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/ops/NDLinalg.java
    - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/learning/*Updater.java

SameDiff/Inference paths (OpContext)
- InferenceSession executes ops via `Nd4j.exec(op, opContext)` after building an OpContext from per‑execution inputs.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/InferenceSession.java
- This bypasses `DefaultOpExecutioner.initOpContext` and does **not** use `op.inputArguments()`/`op.outputArguments()` for execution. That is the safe path for shared graphs.

Risk condition for multithreading
- Any app code that caches a `CustomOp` or `Op` instance and calls `Nd4j.exec(op)` concurrently is at risk, because:
  - `initOpContext` pulls inputs/outputs from the mutable op lists.
  - those lists are cleared/mutated by SameDiff after each execution (`clearArrays`).
- This can yield wrong `fastpath_out` arrays and type validation errors like the `cast` inherit‑type failure.

4) SameDiff/DynamicCustomOp shared state (thread safety hot spots)

Mutable op state (not thread-safe)
- `DynamicCustomOp.clearArrays()` clears `inputArguments` and `outputArguments`.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/DynamicCustomOp.java
- `BaseOp.clearArrays()` nulls `x/y/z`.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/BaseOp.java
- `DynamicCustomOp.calculateOutputShape(oc==null)` uses internal `inputArguments` count; if cleared concurrently, it can throw "not fully initialized".
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/DynamicCustomOp.java

Where SameDiff mutates ops
- SameDiff `ensureCustomOpInputsReady` uses `customOp.addInputArgument(arr)` when `numInputArguments() == 0`, mutating shared op state.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/SameDiff.java
- SameDiff `execCustomOp` path clears arrays then re-adds outputs before executing with a context built from `customOp.inputArguments()` / `outputArguments()`.
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/SameDiff.java
- InferenceSession clears arrays after every op execution (`op.getOp().clearArrays()`).
  - nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/InferenceSession.java

Why this matters for multithreaded SameDiff
- SameDiff stores op instances in the graph (`sameDiff.getOps()`), and those are shared across requests.
- If two threads execute the same graph concurrently, one thread can clear or mutate an op’s input/output lists while the other is building its OpContext (or invoking a no‑context exec path), leading to:
  - empty or mismatched input/output lists,
  - wrong output arrays in `fastpath_out`,
  - type validation failures (including the `cast` inherit‑type error).
