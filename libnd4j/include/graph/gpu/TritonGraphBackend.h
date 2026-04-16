/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_TRITON_GRAPH_BACKEND_H
#define LIBND4J_TRITON_GRAPH_BACKEND_H

#include <graph/GraphBackend.h>
#include <graph/NativeDynamicShapePlan.h>

#include <config.h>

#if HAVE_TRITON

// Forward declarations only — full headers included in .cpp/.cu files.
// TritonIRBuilder.h contains MLIR types (mlir::Value) that NVCC cannot compile.
#include <graph/gpu/TritonIRBuilder_types.h>
#include <graph/gpu/TritonTargetDispatch.h>

#include <atomic>
#include <functional>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

/**
 * Triton GPU compiler backend for the native plan executor.
 *
 * Compiles sequences of ops into fused GPU kernels using Triton's MLIR-based
 * compiler. Unlike CUDA Graphs (which replay separately-launched kernels),
 * Triton fuses ops at the compiler level — eliminating intermediate global
 * memory traffic and kernel launch overhead.
 *
 * For segments exceeding MAX_COMPILABLE_OPS (register pressure limit),
 * the backend automatically splits into sub-segments and compiles each
 * as a separate kernel. All sub-kernels execute sequentially on the same
 * stream — no fallback to CUDA graphs or slot-by-slot.
 *
 * Supports NVIDIA (PTX), AMD (AMDGCN), and Intel (SPIR-V) via Triton's
 * multi-target compiler backend.
 */
class TritonGraphBackend : public GraphBackend {
 public:
  TritonGraphBackend();
  ~TritonGraphBackend() override;

  const char* name() const override { return "Triton GPU"; }
  bool isAvailable() const override;
  bool canFuseSegment(NativeSlot* slots, int start, int end) override;

  bool compileSegment(GraphSegment& seg, NativeSlot* slots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots,
                      LongType shapeKey,
                      int totalSlots = 0,
                      int* requestedOutputSlotIndices = nullptr,
                      int numRequestedOutputs = 0) override;

  Status executeSegment(GraphSegment& seg, NativeSlot* slots,
                        NDArray** externalInputs, int numExternalInputs,
                        NDArray** outputSlots, int totalOutputSlots,
                        void* stream) override;

  void invalidateCache() override;

  /**
   * Clear only the failed-compilation cache (negative cache).
   * Called during plan recompilation to allow previously-failed segments
   * (e.g., attention with seqK=0 before KV setup) to retry compilation
   * with updated external input shapes.  Unlike invalidateCache(), this
   * does NOT free compiled kernels.
   */
  void clearFailedSegmentCache();

  /**
   * Remove cache entries whose slot range overlaps with the given segments.
   * Called from NativeDynamicShapePlan destructor to free compiled GPU modules
   * that would otherwise leak in the singleton cache across plan lifetimes.
   */
  void invalidateCacheForSegments(const std::vector<std::pair<int,int>>& segmentRanges);

  std::vector<CompilationAuditEntry> getLastCompilationAudit() const override;

  static TritonGraphBackend& getInstance();
  using OrderedRangeExecutor = std::function<Status(int, int)>;
  static void setOrderedRangeExecutor(OrderedRangeExecutor executor);
  static void clearOrderedRangeExecutor();

  // Refresh all indirect arg table pinned host buffers with current NDArray
  // specialBuffer() addresses. Must be called before CUDA graph replay so the
  // graph's H2D memcpy nodes transfer up-to-date buffer pointers to device.
  // @param execStream  The execution stream (from LaunchContext) — used for
  //                    cudaStreamSynchronize to ensure prior async work is visible.
  Status refreshArgTablesForReplay(GraphSegment& seg,
                                   NDArray** externalInputs, int numExternalInputs,
                                   NDArray** outputSlots, int totalOutputSlots,
                                   void* execStream = nullptr);

  /**
   * Copy consolidated arg table from host pinned buffer to device.
   * Called after refreshArgTablesForReplay to update device before graph replay.
   * This replaces ~N per-kernel cudaMemcpyAsync calls with ONE consolidated copy.
   */
  void copyConsolidatedArgTableToDevice(GraphSegment& seg, void* stream);

  // Get the set of slot indices NOT covered by any sub-kernel (ordered native ranges).
  // Used by batch-zero to only zero native-range outputs (Triton sub-kernel outputs are
  // NOT zeroed — they're fully written by the Triton kernel).
  std::unordered_set<int> getGapSlots(const GraphSegment& seg, NativeSlot* slots) const;

  // Counters for diagnostics and testing
  LongType getTotalKernelLaunches() const { return totalKernelLaunches_; }
  LongType getTotalCacheHits() const { return totalCacheHits_; }
  void resetCounters() { totalKernelLaunches_ = 0; totalCacheHits_ = 0; }

  // ── Auto-tune profiling ──
  // On first execution of a segment, record kernel execution time.
  // If the time exceeds a threshold for the problem size, the segment is
  // marked for recompilation with alternative tile configurations.
  // This enables profile-guided optimization without upfront multi-variant compile.
  struct AutoTuneEntry {
    float firstExecTimeMs = 0.0f;    // Time of first execution (ms)
    int configIndex = 0;              // Which tile config was used (0 = default)
    int numAttempts = 0;              // Number of configs tried so far
    bool settled = false;             // True once the best config is found
    static constexpr int MAX_ATTEMPTS = 3;
  };
  std::unordered_map<LongType, AutoTuneEntry> autoTuneCache_;  // shapeKey → entry

 private:
  LongType totalKernelLaunches_ = 0;
  LongType totalCacheHits_ = 0;
  // Compiled kernel: GPU module + kernel function + launch config
  struct CompiledKernel {
    void* gpuModule;            // Driver module (CUmodule / hipModule_t / ze_module_handle_t)
    void* kernelFunction;       // Kernel function handle
    unsigned int gridX, gridY, gridZ;
    unsigned int blockX, blockY, blockZ;
    unsigned int sharedMemBytes;
    unsigned int globalScratchBytes;      // Triton 3.6.0+ global scratch memory
    unsigned int globalScratchAlignment;  // Alignment for scratch allocation
    int numWarps;
    bool useCooperativeLaunch;  // true if kernel needs cuLaunchCooperativeKernel
    bool useDynamicGrid;        // true for simple 1D kernels that derive gridX from n_elements
    bool useIndirectArgs;       // true if kernel uses indirect arg table (>200 buffer args)
    bool useMultiPhaseLaunch;   // true if kernel uses multi-phase launch (phase_id arg)
    std::vector<TritonIRModule::LaunchPhase> launchPhases;  // Phase boundaries + grid sizes

    size_t estimatedModuleBytes;  // Approximate GPU memory for loaded CUmodule (binary size proxy)

    // ── Module residency LRU metadata ──
    // Used by ModuleResidencyCache to evict cold modules when per-device byte
    // budget is exceeded.  When evicted, gpuModule and kernelFunction are
    // nulled out and the kernel is reloaded from the disk cache on next use.
    std::string diskCacheHash;   // FNV-1a key for disk cache lookup (reload after eviction)
    std::string kernelName;      // Mangled kernel name (cuModuleGetFunction after reload)
    int loadedDeviceId;          // Device the module is currently loaded on (-1 if evicted)
    uint64_t lruTick;            // Monotonic LRU tick — bumped on every touchModule()

    // Sub-segment range (absolute slot indices)
    int startSlot_;
    int endSlot_;

    // Argument mapping: index in args -> {slotIndex, isOutput}
    std::vector<TritonKernelArg> argSlotMapping;

    // Compilation audit
    std::vector<CompilationAuditEntry> audit;

#ifdef SD_CUDA
    // Persistent launch workspace to avoid per-launch cudaMalloc/cudaFree churn.
    // These buffers are reused across executions for the same compiled kernel.
    void* cachedArgTableDevice;
    size_t cachedArgTableBytes;
    int cachedArgTableDeviceId;
    // Persistent pinned host buffer for the arg table — required for CUDA graph
    // capture. The graph records the cudaMemcpyAsync source address, so it must
    // remain valid across graph replays (stack-local vectors would be dead).
    void* cachedArgTableHostPinned;
    size_t cachedArgTableHostPinnedBytes;
    void* cachedSyncCounterDevice;
    int cachedSyncCounterDeviceId;
    void* cachedGlobalScratchDevice;
    size_t cachedGlobalScratchBytes;
    int cachedGlobalScratchDeviceId;
#endif

    CompiledKernel()
        : gpuModule(nullptr), kernelFunction(nullptr),
          gridX(1), gridY(1), gridZ(1),
          blockX(1), blockY(1), blockZ(1),
          sharedMemBytes(0), globalScratchBytes(0), globalScratchAlignment(128),
          numWarps(4),
          useCooperativeLaunch(false),
          useDynamicGrid(true),
          useIndirectArgs(false),
          useMultiPhaseLaunch(false),
          estimatedModuleBytes(0),
          loadedDeviceId(-1), lruTick(0),
          startSlot_(-1), endSlot_(-1)
#ifdef SD_CUDA
          , cachedArgTableDevice(nullptr), cachedArgTableBytes(0),
            cachedArgTableDeviceId(-1),
            cachedArgTableHostPinned(nullptr), cachedArgTableHostPinnedBytes(0),
            cachedSyncCounterDevice(nullptr),
            cachedSyncCounterDeviceId(-1),
            cachedGlobalScratchDevice(nullptr), cachedGlobalScratchBytes(0),
            cachedGlobalScratchDeviceId(-1)
#endif
    {}
  };

  // A compiled segment may contain multiple sub-kernels when the original
  // segment exceeds MAX_COMPILABLE_OPS. Each sub-kernel covers a contiguous
  // range of slots and is executed sequentially.
  struct SlotRange {
    int startSlot;
    int endSlot;
  };

  struct CompiledSegment {
    std::vector<CompiledKernel> subKernels;
    std::vector<SlotRange> orderedRanges;   // Native-executed ranges between Triton islands
    std::vector<CompilationAuditEntry> audit;  // Combined audit across all sub-kernels

#ifdef SD_CUDA
    // Consolidated arg table: single pinned host + device buffer for ALL sub-kernels.
    // Each sub-kernel references an offset in this buffer instead of its own buffer.
    // One cudaMemcpyAsync replaces N per-kernel copies (reduces graph nodes by ~N).
    bool useConsolidatedArgTable = false;
    void* consolidatedArgTableHostPinned = nullptr;
    void* consolidatedArgTableDevice = nullptr;
    size_t consolidatedArgTableBytes = 0;
    int consolidatedArgTableDeviceId = -1;
    // Per-kernel byte offsets into the consolidated buffer
    std::vector<size_t> consolidatedArgTableOffsets;
    // Per-kernel: whether this kernel has any dynamic (non-constant) args
    std::vector<bool> hasDynamicArgs;
#endif

    bool isValid() const {
      for (const auto& k : subKernels) {
        if (!k.gpuModule || !k.kernelFunction) return false;
      }
      return !subKernels.empty();
    }
  };

  // Per-segment cache (keyed by segment start/end + shape + runtime device).
  // GPU driver module handles are device/context-bound and cannot be shared
  // safely across different CUDA devices.
  struct SegmentCacheKey {
    int startSlot;
    int endSlot;
    LongType shapeKey;
    int deviceId;
    bool compileAll;         // Whether tritonCompileAll was enabled at compile time
    size_t excludeOpsHash;   // Hash of tritonExcludeOps string (0 if empty)
    bool operator==(const SegmentCacheKey& o) const {
      return startSlot == o.startSlot &&
             endSlot == o.endSlot &&
             shapeKey == o.shapeKey &&
             deviceId == o.deviceId &&
             compileAll == o.compileAll &&
             excludeOpsHash == o.excludeOpsHash;
    }
  };
  struct SegmentCacheHash {
    size_t operator()(const SegmentCacheKey& k) const {
      size_t h = std::hash<int>()(k.startSlot);
      h ^= std::hash<int>()(k.endSlot) << 1;
      h ^= std::hash<LongType>()(k.shapeKey) << 2;
      h ^= std::hash<int>()(k.deviceId) << 3;
      h ^= std::hash<bool>()(k.compileAll) << 4;
      h ^= std::hash<size_t>()(k.excludeOpsHash) << 5;
      return h;
    }
  };

  std::unordered_map<SegmentCacheKey, CompiledSegment, SegmentCacheHash> cache_;
  // Negative cache: segment/shape/device keys that previously failed Triton compilation.
  // This avoids repeating expensive compile attempts for known-bad shapes on a given device.
  std::unordered_set<SegmentCacheKey, SegmentCacheHash> failedCache_;
  mutable std::mutex cacheMtx_;

 public:
  // Per-device GPU memory consumed by compiled Triton modules (arg tables, scratch, modules)
  static constexpr int kMaxTritonDevices = 16;
  void recordModuleAlloc(int deviceId, size_t bytes);
  void recordModuleFree(int deviceId, size_t bytes);
  size_t getTritonModuleMemory(int deviceId) const;
  size_t getTotalTritonModuleMemory() const;

  // ── ModuleResidencyCache (LRU eviction for loaded CUmodules) ──
  // Bookkeeping is keyed off the per-device byte counters tritonDeviceMemory_
  // and the per-CompiledKernel lruTick / loadedDeviceId / estimatedModuleBytes
  // metadata.  When tritonDeviceMemory_[dev] exceeds
  // env.triton().moduleResidencyBudgetBytes() (if > 0), the least-recently
  // used kernels on that device are unloaded via cuModuleUnload until usage
  // drops below the budget.  Evicted kernels are reloaded from the disk
  // cache (which is atomic) on next launch.
  //
  // Thread safety: registerLoadedKernel/unregisterLoadedKernel/evictIfOverBudget
  // run under loadedKernelsMtx_.  touchModule writes lruTick with relaxed
  // atomicity (a stale tick is fine — eviction picks the lowest, not the
  // exact lowest).  reloadModuleIfEvicted is called on the launch thread
  // and serializes with eviction via loadedKernelsMtx_ when it bumps the
  // device memory counter.

  // Bump the LRU tick on a kernel launch.  Cheap, lock-free.
  void touchModule(CompiledKernel* k);

  // Register a freshly-loaded kernel with the residency cache.
  // Called immediately after cuModuleLoadDataEx + cuModuleGetFunction.
  // Holds loadedKernelsMtx_ briefly to add to the per-device tracking list
  // and trigger an eviction sweep if the device is over budget.
  void registerLoadedKernel(CompiledKernel* k, int deviceId);

  // Remove a kernel from residency tracking (e.g., when its CompiledSegment
  // is being deleted from cache_ via invalidateCache or
  // invalidateCacheForSegments).  Does NOT call cuModuleUnload — the caller
  // is responsible for that.
  void unregisterLoadedKernel(CompiledKernel* k);

  // Reload an evicted kernel from the disk cache.  Called on the launch
  // path when gpuModule == nullptr but diskCacheHash is populated.  Returns
  // Status::OK on success or KERNEL_FAILURE if the disk-cache read or
  // cuModuleLoadDataEx fails.  Updates k->gpuModule, k->kernelFunction,
  // k->loadedDeviceId, and k->estimatedModuleBytes; calls recordModuleAlloc
  // and registers the reloaded kernel with the residency cache.
  Status reloadModuleIfEvicted(CompiledKernel* k);

  // While tritonDeviceMemory_[deviceId] > moduleResidencyBudgetBytes(),
  // unload the least-recently-used kernel on that device.  Caller may
  // optionally exclude one kernel pointer from eviction (e.g., the kernel
  // that just got loaded so it doesn't immediately evict itself).
  // Caller must NOT hold loadedKernelsMtx_; the function takes the lock.
  void evictIfOverBudget(int deviceId, CompiledKernel* dontEvict = nullptr);

  // ── Batched preload (task #4) ─────────────────────────────────────────────
  // Force-load every CompiledKernel module in the cache into GPU memory in a
  // single pass.  Called at the end of platformPrecompileSegments so DSP
  // execution does not pay per-segment lazy-load latency on the first replay,
  // and so any disk-cache reload work happens up front rather than scattered
  // through the steady-state hot path.
  //
  // For each cached CompiledKernel:
  //   - If gpuModule != nullptr, count its bytes toward the projected total.
  //   - Otherwise (e.g., previously evicted), reload from the disk cache via
  //     reloadModuleIfEvicted() and count the freshly-loaded bytes.
  //
  // Before doing any reloads, the projected total is compared against
  // env.triton().moduleResidencyBudgetBytes() (when > 0); if the projection
  // exceeds the budget a loud sd_printf warning is emitted but the preload
  // still runs (the budget is advisory at preload time — eviction will trim
  // after the fact if needed).
  //
  // @param deviceId  Device that the caller has already activated via
  //                  cudaSetDevice — used both for the budget check and as
  //                  the device-id passed to recordModuleAlloc on reloads.
  // @return Status::OK on success, KERNEL_FAILURE if any reload fails.
  Status preloadAllModules(int deviceId);

 private:
  std::atomic<size_t> tritonDeviceMemory_[kMaxTritonDevices]{};

  // Residency LRU bookkeeping
  mutable std::mutex loadedKernelsMtx_;
  std::vector<CompiledKernel*> loadedKernels_[kMaxTritonDevices];
  std::atomic<uint64_t> lruTickCounter_{1};
  std::atomic<bool> residencyWarned_[kMaxTritonDevices]{};
  // Most recent compilation audit
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Minimum fraction of mappable ops required to attempt Triton compilation
  static constexpr float MIN_MAPPABLE_FRACTION = 0.5f;

  // Minimum number of mappable ops for Triton to be worthwhile
  static constexpr int MIN_MAPPABLE_OPS = 1;

  // Default max parallel compilations for inner sub-range parallelism
  // within a single segment's compileSegment() call.
  //
  // Thread safety is ensured by per-thread isolation of compilation state:
  //   - Each sub-segment creates its own MLIRContext (in TritonIRBuilder)
  //   - Each compile() call creates stack-local LLVMContext, TargetMachine,
  //     and PassManager instances
  //   - Shared global state is protected by targeted mutexes:
  //     * LLVM target init: std::once_flag
  //     * MLIR dialect registry: mlirRegistryMtx / mlirTranslationMtx
  //     * cuModuleLoadDataEx: loadModuleMtx (CUDA driver JIT is not thread-safe)
  //
  // Override with ND4J_TRITON_BUILD_THREADS=N environment variable.
  static constexpr int DEFAULT_MAX_PARALLEL_COMPILATIONS = 8;

  // Configurable max parallel compilations (set via ND4J_TRITON_BUILD_THREADS env var)
  static int maxParallelCompilations_;
  static std::mutex configMtx_;
  static thread_local OrderedRangeExecutor orderedRangeExecutor_;

 public:
  // Maximum number of ops that can be compiled into a single Triton kernel.
  // Larger segments exceed register limits (441K virtual regs for 3840 ops).
  // Segments exceeding this are split into sub-segments automatically.
  // Increased from 512 to 768 to reduce kernel launch overhead while staying
  // within register pressure limits for most decoder models.
  static constexpr int MAX_COMPILABLE_OPS = 768;

  // Maximum number of sub-kernels to compile in parallel.
  // Limited to avoid excessive memory usage during LLVM compilation.
  // Each parallel compilation uses ~1-2GB RAM.
  // Set via environment variable ND4J_TRITON_BUILD_THREADS (default: 4)
  static int getMaxParallelCompilations();
  static void setMaxParallelCompilations(int maxThreads);

  // Check if all ops in a range are Triton-mappable (without size limit check).
  // Used by sub-segment splitting to verify individual sub-segments.
  bool areAllOpsMappable(NativeSlot* slots, int start, int end);

 private:

  // Compile TTIR module to GPU binary, load, and extract kernel
  CompiledKernel compileToGpuBinary(NativeSlot* slots, int startSlot, int endSlot,
                                    int totalSlots,
                                    NDArray** externalInputs, int numExternalInputs,
                                    NDArray** outputSlots, int totalOutputSlots);

  // Disk cache helpers for compiled PTX
  std::string getDiskCacheDir() const;
  bool ensureDiskCacheDir(const std::string& cacheDir) const;
  std::string computeDiskCacheHash(const std::string& ttirText,
                                   int numWarps, int numStages) const;
  bool loadBinaryFromDiskCache(int startSlot, int endSlot,
                               const std::string& cacheHash,
                               const TritonIRModule& irModule,
                               TritonCompiledBinary& binary) const;
  // Reload-from-eviction variant.  Validates against the supplied kernelName
  // (instead of pulling it from a TritonIRModule) and is callable without an
  // MLIR context, since reloads happen long after IR build.
  bool loadBinaryFromDiskCacheByHash(const std::string& cacheHash,
                                     const std::string& kernelName,
                                     TritonCompiledBinary& binary) const;
  void writeBinaryToDiskCache(int startSlot, int endSlot,
                              const std::string& cacheHash,
                              const TritonIRModule& irModule,
                              const TritonCompiledBinary& binary) const;

  // Execute a single compiled sub-kernel.
  // When argTablePreCopied=true, skip per-kernel H2D memcpy (consolidated copy already done).
  Status executeSingleKernel(CompiledKernel& compiled, NativeSlot* slots,
                             NDArray** externalInputs, int numExternalInputs,
                             NDArray** outputSlots, int totalOutputSlots,
                             void* stream, bool argTablePreCopied = false,
                             NDArray** slotArrayCache = nullptr);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
#endif  // LIBND4J_TRITON_GRAPH_BACKEND_H
