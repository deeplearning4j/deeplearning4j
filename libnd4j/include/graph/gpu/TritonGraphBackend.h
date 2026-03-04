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

#include <graph/gpu/TritonIRBuilder.h>
#include <graph/gpu/TritonTargetDispatch.h>

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

  std::vector<CompilationAuditEntry> getLastCompilationAudit() const override;

  static TritonGraphBackend& getInstance();
  using FallbackRangeExecutor = std::function<Status(int, int)>;
  static void setFallbackRangeExecutor(FallbackRangeExecutor executor);
  static void clearFallbackRangeExecutor();

  // Refresh all indirect arg table pinned host buffers with current NDArray
  // specialBuffer() addresses. Must be called before CUDA graph replay so the
  // graph's H2D memcpy nodes transfer up-to-date buffer pointers to device.
  Status refreshArgTablesForReplay(GraphSegment& seg,
                                   NDArray** externalInputs, int numExternalInputs,
                                   NDArray** outputSlots, int totalOutputSlots);

  // Get the set of slot indices NOT covered by any sub-kernel (gap/fallback slots).
  // Used by batch-zero to only zero gap op outputs (Triton sub-kernel outputs are
  // NOT zeroed — they're fully written by the Triton kernel).
  std::unordered_set<int> getGapSlots(const GraphSegment& seg, NativeSlot* slots) const;

  // Counters for diagnostics and testing
  LongType getTotalKernelLaunches() const { return totalKernelLaunches_; }
  LongType getTotalCacheHits() const { return totalCacheHits_; }
  void resetCounters() { totalKernelLaunches_ = 0; totalCacheHits_ = 0; }

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
    std::vector<SlotRange> fallbackRanges;   // Slot ranges that must run slot-by-slot
    std::vector<CompilationAuditEntry> audit;  // Combined audit across all sub-kernels

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

  // Most recent compilation audit
  std::vector<CompilationAuditEntry> lastCompilationAudit_;

  // Minimum fraction of mappable ops required to attempt Triton compilation
  static constexpr float MIN_MAPPABLE_FRACTION = 0.5f;

  // Minimum number of mappable ops for Triton to be worthwhile
  static constexpr int MIN_MAPPABLE_OPS = 1;

  // Default max parallel compilations
  static constexpr int DEFAULT_MAX_PARALLEL_COMPILATIONS = 4;

  // Configurable max parallel compilations (set via ND4J_TRITON_BUILD_THREADS env var)
  static int maxParallelCompilations_;
  static std::mutex configMtx_;
  static thread_local FallbackRangeExecutor fallbackRangeExecutor_;

 public:
  // Maximum number of ops that can be compiled into a single Triton kernel.
  // Larger segments exceed register limits (441K virtual regs for 3840 ops).
  // Segments exceeding this are split into sub-segments automatically.
  static constexpr int MAX_COMPILABLE_OPS = 512;

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
                                    LongType segmentShapeKey,
                                    int totalSlots,
                                    NDArray** externalInputs, int numExternalInputs,
                                    NDArray** outputSlots, int totalOutputSlots);

  // Disk cache helpers for compiled PTX
  std::string getDiskCacheDir() const;
  bool ensureDiskCacheDir(const std::string& cacheDir) const;
  std::string computeDiskCacheHash(int startSlot, int endSlot,
                                   LongType segmentShapeKey,
                                   const std::string& ttirText,
                                   int numWarps, int numStages) const;
  bool loadBinaryFromDiskCache(int startSlot, int endSlot,
                               const std::string& cacheHash,
                               const TritonIRModule& irModule,
                               TritonCompiledBinary& binary) const;
  void writeBinaryToDiskCache(int startSlot, int endSlot,
                              const std::string& cacheHash,
                              const TritonIRModule& irModule,
                              const TritonCompiledBinary& binary) const;

  // Execute a single compiled sub-kernel
  Status executeSingleKernel(CompiledKernel& compiled, NativeSlot* slots,
                             NDArray** externalInputs, int numExternalInputs,
                             NDArray** outputSlots, int totalOutputSlots,
                             void* stream);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
#endif  // LIBND4J_TRITON_GRAPH_BACKEND_H
