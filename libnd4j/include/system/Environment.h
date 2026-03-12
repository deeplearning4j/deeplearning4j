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

//
// Created by raver119 on 06.10.2017.
//

#ifndef LIBND4J_ENVIRONMENT_H
#define LIBND4J_ENVIRONMENT_H
#include <array/DataType.h>
#include <types/pair.h>

#include <atomic>
#include <string>
#include <stdexcept>
#include <vector>
#include <config.h>

#ifdef SD_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaLimitType.h"
#endif

namespace sd {
class SD_LIB_EXPORT Environment {
 private:
  std::atomic<int> _tadThreshold;
  std::atomic<int> _elementThreshold;
  std::atomic<bool> _verbose;
  std::atomic<bool> _debug;
  std::atomic<bool> _leaks;
  std::atomic<bool> _profile;
  std::atomic<sd::DataType> _dataType;
  std::atomic<bool> _precBoost;
  std::atomic<bool> _useONEDNN{true};
  std::atomic<bool> _useMPS{true};
  std::atomic<bool> _allowHelpers{true};
  std::atomic<bool> funcTracePrintDeallocate;
  std::atomic<bool> funcTracePrintAllocate;

  // NDArray lifecycle tracking fields
  // Prevents backward-cpp crashes during early JVM initialization
  // Can be enabled via SD_LIFECYCLE_TRACKING=1 env var after JVM is ready
  std::atomic<bool> _lifecycleTracking{false};
  std::atomic<bool> _trackViews{true};
  std::atomic<bool> _trackDeletions{true};
  std::atomic<int> _stackDepth{32};
  std::atomic<int> _reportInterval{300};
  std::atomic<size_t> _maxDeletionHistory{10000};
  std::atomic<bool> _snapshotFiles{false};  // Default off - only write snapshots on demand
  std::atomic<bool> _trackOperations{false};  // Default off - operation tracking adds overhead

  // Individual tracker enable flags
  std::atomic<bool> _ndArrayTracking{false};
  std::atomic<bool> _dataBufferTracking{false};
  std::atomic<bool> _tadCacheTracking{false};
  std::atomic<bool> _shapeCacheTracking{false};
  std::atomic<bool> _opContextTracking{false};

  std::atomic<int> _maxThreads;
  std::atomic<int> _maxMasterThreads;
  std::atomic<bool> deleteSpecial{true};
  std::atomic<bool> deletePrimary{true};
  std::atomic<bool> deleteShapeInfo{true};
  std::atomic<bool> _checkInputChange{false};
  std::atomic<bool> _checkOutputChange{false};
  std::atomic<bool> _logNDArrayEvenuts{false};
  std::atomic<bool> _logNativeNDArrayCreation{false};
  // these fields hold defaults
  std::atomic<int64_t> _maxTotalPrimaryMemory{-1};
  std::atomic<int64_t> _maxTotalSpecialMemory{-1};
  std::atomic<int64_t> _maxDeviceMemory{-1};
  bool _blasFallback = false;
  std::atomic<bool> _enableBlasFall{true};

  // BLAS call serialization to prevent OpenBLAS TLS corruption and race conditions
  // Default true for safety with OpenBLAS in multi-threaded Java applications
  std::atomic<bool> _serializeBlasCallsSet{false};  // tracks if explicitly set
  std::atomic<int> _openBlasThreads{0};  // 0 = use default

#ifdef SD_EXPERIMENTAL_ENABLED
  const bool _experimental = true;
#else
  const bool _experimental = false;
#endif

  // device compute capability for CUDA
  std::vector<Pair> _capabilities;

  // CUDA specific environment configurations
  std::atomic<int> _cudaDeviceCount{0};
  std::atomic<int> _cudaCurrentDevice{0};
  std::atomic<bool> _cudaMemoryPinned{false};
  std::atomic<bool> _cudaUseManagedMemory{false};
  std::atomic<int> _cudaMemoryPoolSize{0};  // in MB
  std::atomic<bool> _cudaForceP2P{false};
  std::atomic<bool> _cudaAllocatorEnabled{true};
  std::atomic<int> _cudaMaxBlocks{0};
  std::atomic<int> _cudaMaxThreadsPerBlock{0};
  std::atomic<bool> _cudaAsyncExecution{true};
  std::atomic<int> _cudaStreamLimit{4};
  std::atomic<bool> _cudaUseDeviceHost{false};
  std::atomic<int> _cudaEventLimit{4};
  std::atomic<int> _cudaCachingAllocatorLimit{0}; // in MB
  std::atomic<int64_t> _cudaPinnedHostLimit{8LL * 1024}; // in MB, default 8 GB
  std::atomic<bool> _cudaUseUnifiedMemory{false};
  std::atomic<int> _cudaPrefetchSize{0}; // in MB
  std::atomic<bool> _cudaGraphOptimization{false};
  std::atomic<bool> _cudaTensorCoreEnabled{true};
  std::atomic<int> _cudaBlockingSync{0};
  std::atomic<int> _cudaDeviceSchedule{0}; // 0: default, 1: spin, 2: yield, 3: block

  // NDArray print options (similar to NumPy's printoptions)
  std::atomic<int> _printEdgeItems{3};       // Number of elements at each edge when summarizing
  std::atomic<int> _printThreshold{1000};    // Total elements before switching to summarized output
  std::atomic<int> _printLineWidth{75};      // Characters per line for output
  std::atomic<int> _printPrecision{8};       // Floating point precision (digits after decimal)

  // CUDA Device Limit configurations
  std::atomic<size_t> _cudaStackSize{0};            // cudaLimitStackSize
  std::atomic<size_t> _cudaMallocHeapSize{0};       // cudaLimitMallocHeapSize
  std::atomic<size_t> _cudaPrintfFifoSize{0};       // cudaLimitPrintfFifoSize
  std::atomic<size_t> _cudaDevRuntimeSyncDepth{0};  // cudaLimitDevRuntimeSyncDepth
  std::atomic<size_t> _cudaDevRuntimePendingLaunchCount{0}; // cudaLimitDevRuntimePendingLaunchCount
  std::atomic<size_t> _cudaMaxL2FetchGranularity{0}; // cudaLimitMaxL2FetchGranularity
  std::atomic<size_t> _cudaPersistingL2CacheSize{0}; // cudaLimitPersistingL2CacheSize

  // Triton GPU compilation settings
  std::atomic<int> _tritonBuildThreads{8};
  std::atomic<bool> _tritonCacheEnabled{true};
  std::atomic<bool> _tritonCooperativeLaunch{false};  // cooperative launch OFF by default
  std::atomic<int> _tritonCoopTargetBlocks{0};  // 0 = auto
  std::atomic<int> _tritonMaxSubsegmentOps{0};       // 0 = auto/adaptive
  std::atomic<int> _tritonMaxSubsegmentSections{0};  // 0 = auto/adaptive
  std::atomic<bool> _tritonVerbose{false};
  std::atomic<bool> _tritonDumpSections{false};
  std::atomic<bool> _tritonDumpArgs{false};
  std::atomic<bool> _tritonLogAllPatterns{false};
  std::atomic<bool> _tritonAlwaysCompile{false};
  std::atomic<bool> _tritonKernelDump{false};
  std::atomic<bool> _tritonKernelOverride{false};
  std::atomic<int> _tritonNumWarps{0};    // 0 = auto
  std::atomic<int> _tritonNumStages{0};   // 0 = auto
  std::atomic<int> _tritonNumCTAs{1};     // 1 = default (non-clustered)
  std::atomic<int> _tritonMaxNreg{0};     // 0 = unset
  std::atomic<bool> _tritonEnableFpFusion{true};
  std::atomic<bool> _tritonDisableLineInfo{false};
  std::string _tritonCacheDir;
  std::string _tritonDumpDir;
  std::string _tritonOverrideDir;
  std::string _tritonOverrideArch;

  // Triton + CUDA graph integration
  std::atomic<bool> _tritonAllowFallbackCapture{true};  // allow fallback executor during CUDA graph capture
  std::atomic<bool> _tritonGraphCapture{true};           // enable CUDA graph capture of Triton execution
  std::atomic<bool> _tritonDumpGraphDot{false};          // dump captured graph to DOT file for debugging
  std::atomic<bool> _tritonGraphCtxPush{false};          // push primary CUDA context during graph capture
  std::atomic<bool> _tritonGraphReinstantiate{false};    // re-instantiate graph exec before each replay
  std::atomic<bool> _tritonGraphAutoFree{false};         // use cudaGraphInstantiateFlagAutoFreeOnLaunch
  std::atomic<bool> _tritonGraphDotVerbose{false};       // use verbose flags for DOT export (may poison driver state)

  // Triton compilation scope — when true, Triton compiles ALL section types except
  // ops listed in the exclusion list. Default false = only ELEMENTWISE/IDENTITY compiled.
  std::atomic<bool> _tritonCompileAll{false};

  // Comma-separated list of nd4j op names to EXCLUDE from Triton compilation.
  // These ops fall back to cuBLAS/native execution. Typical exclusions:
  //   "matmul,mmul,tensormmul" — keep GEMMs on cuBLAS (usually faster)
  //   "matmul,softmax" — keep matmul on cuBLAS, softmax on native
  // Empty string = no exclusions (compile everything through Triton).
  // Set via ND4J_TRITON_EXCLUDE_OPS env var.
  std::string _tritonExcludeOps;

  // When tritonCompileAll=true, only compile these section types (plus ELEMENTWISE/IDENTITY).
  // Comma-separated type names: "CONST_GEN,SHAPE_MANIP,GATHER,CONCAT,SPLIT,STACK,REDUCTION,ATTENTION"
  // Empty = compile ALL types (original compileAll behavior).
  // Set via ND4J_TRITON_INCLUDE_TYPES env var.
  std::string _tritonIncludeTypes;

  // DSP batch-zero: replace per-slot memsets with a single batch kernel during graph capture
  std::atomic<bool> _dspBatchZero{false};           // ND4J_DSP_BATCH_ZERO (default: off)
  std::atomic<bool> _dspBatchZeroVerbose{false};    // ND4J_DSP_BATCH_ZERO_VERBOSE — log every buffer
  std::atomic<bool> _dspBatchZeroGapOnly{true};     // ND4J_DSP_BATCH_ZERO_GAP_ONLY — only zero gap slots (default: true)
  std::atomic<bool> _dspBatchZeroKernel{false};     // ND4J_DSP_BATCH_ZERO_KERNEL — use single kernel instead of N memsets

  // DSP batched GEMM: group consecutive same-shape matmul slots into single cublasGemmBatchedEx calls
  std::atomic<bool> _dspBatchedGemm{false};          // ND4J_DSP_BATCHED_GEMM (default: off)

  // Triton debugging flags
  std::atomic<bool> _tritonSkipKernels{false};       // skip Triton kernels, run native fallback instead
  std::atomic<bool> _tritonVerifyKernels{false};     // run both Triton and native, compare outputs
  std::atomic<bool> _tritonVerifyKeepNative{false};  // keep native outputs during verify (test error accumulation)
  std::atomic<int>  _tritonMaxSubKernelIndex{-1};    // max sub-kernel index to run via Triton (-1 = unlimited)
  std::atomic<bool> _tritonVerifyFullSnapshot{false}; // save/restore ALL outputSlots during verify (detect corruption)
  std::atomic<bool> _tritonForceRecapture{false};    // force CUDA graph re-capture every step (diagnostic)
  std::atomic<int>  _tritonCaptureMinExec{2};        // minimum execution count before graph capture

  // DSP optimization flags
  std::atomic<bool> _dspCastElimination{true};      // eliminate redundant cast pairs in FusionPass
  std::atomic<bool> _dspMatmulSegmentation{false};   // break segments at matmul boundaries
  std::atomic<bool> _dspFp16Compute{false};          // auto-cast FP32 matmul inputs to FP16 for TensorCore
  std::atomic<bool> _cublasTf32Enabled{false};       // enable TF32 math mode for cuBLAS on sm_80+
  std::atomic<bool> _dspCastSinkMatmul{false};       // sink FP16→FP32 casts through matmul ops
  std::atomic<bool> _tritonConsolidatedArgTable{false}; // consolidate arg tables into single H2D copy
  std::atomic<bool> _tritonArgDirtyTracking{false};  // skip arg table refresh for static-only sub-kernels
  std::atomic<bool> _tritonSectionFusion{false};     // merge non-EW sections into mega-kernels

  // Fusion scoring: cost-model-based section merge decisions
  std::atomic<bool> _tritonFusionScoring{true};      // ND4J_TRITON_FUSION_SCORING — use cost model for section merges
  std::atomic<float> _tritonFusionMinScore{1.0f};    // ND4J_TRITON_FUSION_MIN_SCORE — minimum score to merge sections

  // Symbolic shape ranges: avoid recompilation when dimensions change within observed bounds
  std::atomic<bool> _dspSymbolicShapes{true};         // ND4J_DSP_SYMBOLIC_SHAPES — enable range-based shape keys
  std::atomic<int>  _dspSymbolicShapeWarmup{2};       // ND4J_DSP_SYMBOLIC_SHAPE_WARMUP — observation steps before ranging

  // CUDA graph capture buffer pool sharing via CudaMemoryPool
  std::atomic<bool> _dspCapturePoolEnabled{true};     // ND4J_DSP_CAPTURE_POOL_ENABLED — route capture buffers through pool
  std::atomic<long long> _dspCapturePoolMaxBytes{1073741824LL}; // ND4J_DSP_CAPTURE_POOL_MAX_BYTES — 1GB default

  Environment();

 public:
  ~Environment();
  /**
   * These 3 fields are mostly for CUDA/cuBLAS version tracking
   */
  int _blasMajorVersion = 0;
  int _blasMinorVersion = 0;
  int _blasPatchVersion = 0;

  static Environment& getInstance();

  bool isEnableBlas() {
    return _enableBlasFall.load();
  }

  void setEnableBlas(bool reallyEnable) {
    _enableBlasFall.store(reallyEnable);
  }

  /**
   * When log ndarray evens is true in c++
   * certain features of ndarray logging will trigger such as what ndarray constructors are being called.
   *  A great use case for this is for detecting subtle changes in ndarrays like move constructor calls
   *  which  can cause the underlying data to change.
   * @return
   */
  bool isLogNativeNDArrayCreation();
  void setLogNativeNDArrayCreation(bool logNativeNDArrayCreation);

  /**
   * This is mostly a java feature. We can use this to build a framework
   * for logging ndarray events from c++ later.
   * @return
   */
  bool isLogNDArrayEvents();

  void setLogNDArrayEvents(bool logNDArrayEvents);

  /**
   * This is mainly for debugging. This toggles
   * deletion of shape info descriptors.
   * This can be used to isolate potential issues with shape info
   * memory management.
   * The next concern is why have this at all?
   * Historically, we had issues with shape descriptors and shape info
   * buffers being deallocated when they shouldn't be due to stack based deallocation.
   * By controlling everything with normal heap allocation, manual deletes and configurable behavior
   * we can keep memory management consistent and predictable.
   */

  bool isDeleteSpecial();
  void setDeleteSpecial(bool reallyDelete);
  bool isDeletePrimary();
  void setDeletePrimary(bool reallyDelete);


  /**
   * Checks whether the outputs of the op have changed
   * by duplicating them before and after the op runs
   * if it doesn't change it throws an exception.
   * @return
   */
  bool isCheckOutputChange();

  void setCheckOutputChange(bool reallyCheck);

  /**
   * Checks whether immutable ops changed their inputs by
   * duplicating each input and ensuring they're still equal after the op runs.
   * @return
   */
  bool isCheckInputChange();
  void setCheckInputChange(bool reallyCheck);

  bool isVerbose();
  void setVerbose(bool reallyVerbose);
  bool isDebug();
  bool isProfiling();
  bool isDetectingLeaks();
  bool isDebugAndVerbose();
  void setDebug(bool reallyDebug);
  void setProfiling(bool reallyProfile);
  void setLeaksDetector(bool reallyDetect);
  bool helpersAllowed();
  void allowHelpers(bool reallyAllow);

  bool blasFallback();

  /**
   * Check if BLAS call serialization is enabled.
   * When enabled, external BLAS calls are serialized to prevent OpenBLAS
   * TLS corruption and race conditions in multi-threaded environments.
   * Default is true for safety.
   */
  bool isSerializeBlasCalls();

  /**
   * Enable or disable BLAS call serialization.
   * Disable only if using a thread-safe BLAS implementation (e.g., MKL).
   */
  void setSerializeBlasCalls(bool serialize);

  /**
   * Get the number of threads OpenBLAS should use.
   * Returns 0 if using OpenBLAS default.
   */
  int getOpenBlasThreads();

  /**
   * Set the number of threads OpenBLAS should use.
   * Set to 0 for OpenBLAS default, or a specific number for explicit control.
   */
  void setOpenBlasThreads(int threads);

  int tadThreshold();
  void setTadThreshold(int threshold);

  int elementwiseThreshold();
  void setElementwiseThreshold(int threshold);

  int maxThreads();
  void setMaxThreads(int max);

  int maxMasterThreads();
  void setMaxMasterThreads(int max);

  /*
   * Legacy memory limits API, still used in new API as simplified version
   */
  void setMaxPrimaryMemory(uint64_t maxBytes);
  void setMaxSpecialyMemory(uint64_t maxBytes);
  void setMaxDeviceMemory(uint64_t maxBytes);

  uint64_t maxPrimaryMemory();
  uint64_t maxSpecialMemory();
  ////////////////////////

  /*
   * Methods for memory limits/counters
   */
  void setGroupLimit(int group, sd::LongType numBytes);
  void setDeviceLimit(int deviceId, sd::LongType numBytes);

  sd::LongType getGroupLimit(int group);
  sd::LongType getDeviceLimit(int deviceId);

  sd::LongType getGroupCounter(int group);
  sd::LongType getDeviceCounter(int deviceId);
  ////////////////////////

  bool isUseONEDNN() { return _useONEDNN.load(); }
  void setUseONEDNN(bool useMKLDNN) { _useONEDNN.store(useMKLDNN); }

  bool isUseMPS() { return _useMPS.load(); }
  void setUseMPS(bool useMPS) { _useMPS.store(useMPS); }

  sd::DataType defaultFloatDataType();
  void setDefaultFloatDataType(sd::DataType dtype);

  bool precisionBoostAllowed();
  void allowPrecisionBoost(bool reallyAllow);

  bool isExperimentalBuild();

  bool isCPU();

  int blasMajorVersion();
  int blasMinorVersion();
  int blasPatchVersion();

  std::vector<Pair>& capabilities();


  bool isFuncTracePrintDeallocate();
  void setFuncTracePrintDeallocate(bool reallyPrint);
  bool isFuncTracePrintAllocate();
  void setFuncTracePrintAllocate(bool reallyPrint);

  // NDArray lifecycle tracking methods
  bool isLifecycleTracking();
  void setLifecycleTracking(bool enabled);
  bool isTrackViews();
  void setTrackViews(bool track);
  bool isTrackDeletions();
  void setTrackDeletions(bool track);
  int getStackDepth();
  void setStackDepth(int depth);
  int getReportInterval();
  void setReportInterval(int seconds);
  size_t getMaxDeletionHistory();
  void setMaxDeletionHistory(size_t max);
  bool isSnapshotFiles();
  void setSnapshotFiles(bool enabled);
  bool isTrackOperations();
  void setTrackOperations(bool enabled);

  // Individual tracker enable/disable methods
  bool isNDArrayTracking();
  void setNDArrayTracking(bool enabled);
  bool isDataBufferTracking();
  void setDataBufferTracking(bool enabled);
  bool isTADCacheTracking();
  void setTADCacheTracking(bool enabled);
  bool isShapeCacheTracking();
  void setShapeCacheTracking(bool enabled);
  bool isOpContextTracking();
  void setOpContextTracking(bool enabled);

  bool isDeleteShapeInfo();
  void setDeleteShapeInfo(bool deleteShapeInfo);

  // CUDA specific getters/setters
  int cudaDeviceCount() { return _cudaDeviceCount.load(); }
  int cudaCurrentDevice() { return _cudaCurrentDevice.load(); }
  void setCudaCurrentDevice(int device);
  bool cudaMemoryPinned() { return _cudaMemoryPinned.load(); }
  void setCudaMemoryPinned(bool pinned);
  bool cudaUseManagedMemory() { return _cudaUseManagedMemory.load(); }
  void setCudaUseManagedMemory(bool managed);
  int cudaMemoryPoolSize() { return _cudaMemoryPoolSize.load(); }
  void setCudaMemoryPoolSize(int sizeInMB);
  bool cudaForceP2P() { return _cudaForceP2P.load(); }
  void setCudaForceP2P(bool forceP2P);
  bool cudaAllocatorEnabled() { return _cudaAllocatorEnabled.load(); }
  void setCudaAllocatorEnabled(bool enabled);
  int cudaMaxBlocks() { return _cudaMaxBlocks.load(); }
  void setCudaMaxBlocks(int blocks);
  int cudaMaxThreadsPerBlock() { return _cudaMaxThreadsPerBlock.load(); }
  void setCudaMaxThreadsPerBlock(int threads);
  bool cudaAsyncExecution() { return _cudaAsyncExecution.load(); }
  void setCudaAsyncExecution(bool async);
  int cudaStreamLimit() { return _cudaStreamLimit.load(); }
  void setCudaStreamLimit(int limit);
  bool cudaUseDeviceHost() { return _cudaUseDeviceHost.load(); }
  void setCudaUseDeviceHost(bool useDeviceHost);
  int cudaEventLimit() { return _cudaEventLimit.load(); }
  void setCudaEventLimit(int limit);
  int cudaCachingAllocatorLimit() { return _cudaCachingAllocatorLimit.load(); }
  void setCudaCachingAllocatorLimit(int limitInMB);
  int64_t cudaPinnedHostLimit() { return _cudaPinnedHostLimit.load(); }
  void setCudaPinnedHostLimit(int64_t limitInMB);
  bool cudaUseUnifiedMemory() { return _cudaUseUnifiedMemory.load(); }
  void setCudaUseUnifiedMemory(bool unified);
  int cudaPrefetchSize() { return _cudaPrefetchSize.load(); }
  void setCudaPrefetchSize(int sizeInMB);
  bool cudaGraphOptimization() { return _cudaGraphOptimization.load(); }
  void setCudaGraphOptimization(bool enabled);
  bool cudaTensorCoreEnabled() { return _cudaTensorCoreEnabled.load(); }
  void setCudaTensorCoreEnabled(bool enabled);
  int cudaBlockingSync() { return _cudaBlockingSync.load(); }
  void setCudaBlockingSync(int mode);
  int cudaDeviceSchedule() { return _cudaDeviceSchedule.load(); }
  void setCudaDeviceSchedule(int schedule);

  // CUDA Device Limit getters/setters
  size_t cudaStackSize() { return _cudaStackSize.load(); }
  void setCudaStackSize(size_t size);
  size_t cudaMallocHeapSize() { return _cudaMallocHeapSize.load(); }
  void setCudaMallocHeapSize(size_t size);
  size_t cudaPrintfFifoSize() { return _cudaPrintfFifoSize.load(); }
  void setCudaPrintfFifoSize(size_t size);
  size_t cudaDevRuntimeSyncDepth() { return _cudaDevRuntimeSyncDepth.load(); }
  void setCudaDevRuntimeSyncDepth(size_t depth);
  size_t cudaDevRuntimePendingLaunchCount() { return _cudaDevRuntimePendingLaunchCount.load(); }
  void setCudaDevRuntimePendingLaunchCount(size_t count);
  size_t cudaMaxL2FetchGranularity() { return _cudaMaxL2FetchGranularity.load(); }
  void setCudaMaxL2FetchGranularity(size_t size);
  size_t cudaPersistingL2CacheSize() { return _cudaPersistingL2CacheSize.load(); }
  void setCudaPersistingL2CacheSize(size_t size);

  bool setCudaDeviceLimit(int limitType, size_t value);

  // NDArray print options (NumPy-style printoptions)
  int printEdgeItems() { return _printEdgeItems.load(); }
  void setPrintEdgeItems(int edgeItems);
  int printThreshold() { return _printThreshold.load(); }
  void setPrintThreshold(int threshold);
  int printLineWidth() { return _printLineWidth.load(); }
  void setPrintLineWidth(int lineWidth);
  int printPrecision() { return _printPrecision.load(); }
  void setPrintPrecision(int precision);

  // Triton GPU compilation settings
  int tritonBuildThreads() { return _tritonBuildThreads.load(); }
  void setTritonBuildThreads(int threads);
  bool tritonCacheEnabled() { return _tritonCacheEnabled.load(); }
  void setTritonCacheEnabled(bool enabled);
  bool tritonCooperativeLaunch() { return _tritonCooperativeLaunch.load(); }
  void setTritonCooperativeLaunch(bool enabled);
  int tritonCoopTargetBlocks() { return _tritonCoopTargetBlocks.load(); }
  void setTritonCoopTargetBlocks(int blocks);
  int tritonMaxSubsegmentOps() { return _tritonMaxSubsegmentOps.load(); }
  void setTritonMaxSubsegmentOps(int ops);
  int tritonMaxSubsegmentSections() { return _tritonMaxSubsegmentSections.load(); }
  void setTritonMaxSubsegmentSections(int sections);
  bool tritonVerbose() { return _tritonVerbose.load(); }
  void setTritonVerbose(bool verbose);
  bool tritonDumpSections() { return _tritonDumpSections.load(); }
  void setTritonDumpSections(bool dumpSections);
  bool tritonDumpArgs() { return _tritonDumpArgs.load(); }
  void setTritonDumpArgs(bool dumpArgs);
  bool tritonLogAllPatterns() { return _tritonLogAllPatterns.load(); }
  void setTritonLogAllPatterns(bool logAllPatterns);
  bool tritonAlwaysCompile() { return _tritonAlwaysCompile.load(); }
  void setTritonAlwaysCompile(bool alwaysCompile);
  bool tritonKernelDump() { return _tritonKernelDump.load(); }
  void setTritonKernelDump(bool kernelDump);
  bool tritonKernelOverride() { return _tritonKernelOverride.load(); }
  void setTritonKernelOverride(bool kernelOverride);
  int tritonNumWarps() { return _tritonNumWarps.load(); }
  void setTritonNumWarps(int warps);
  int tritonNumStages() { return _tritonNumStages.load(); }
  void setTritonNumStages(int stages);
  int tritonNumCTAs() { return _tritonNumCTAs.load(); }
  void setTritonNumCTAs(int ctas);
  int tritonMaxNreg() { return _tritonMaxNreg.load(); }
  void setTritonMaxNreg(int maxNreg);
  bool tritonEnableFpFusion() { return _tritonEnableFpFusion.load(); }
  void setTritonEnableFpFusion(bool enableFpFusion);
  bool tritonDisableLineInfo() { return _tritonDisableLineInfo.load(); }
  void setTritonDisableLineInfo(bool disableLineInfo);
  std::string tritonCacheDir() const { return _tritonCacheDir; }
  void setTritonCacheDir(const std::string& cacheDir);
  std::string tritonDumpDir() const { return _tritonDumpDir; }
  void setTritonDumpDir(const std::string& dumpDir);
  std::string tritonOverrideDir() const { return _tritonOverrideDir; }
  void setTritonOverrideDir(const std::string& overrideDir);
  std::string tritonOverrideArch() const { return _tritonOverrideArch; }
  void setTritonOverrideArch(const std::string& overrideArch);

  // Triton + CUDA graph integration
  bool tritonAllowFallbackCapture() { return _tritonAllowFallbackCapture.load(); }
  void setTritonAllowFallbackCapture(bool allow);
  bool tritonGraphCapture() { return _tritonGraphCapture.load(); }
  void setTritonGraphCapture(bool enable) { _tritonGraphCapture.store(enable); }
  bool tritonDumpGraphDot() { return _tritonDumpGraphDot.load(); }
  void setTritonDumpGraphDot(bool dump) { _tritonDumpGraphDot.store(dump); }
  bool tritonGraphCtxPush() { return _tritonGraphCtxPush.load(); }
  void setTritonGraphCtxPush(bool v) { _tritonGraphCtxPush.store(v); }
  bool tritonGraphReinstantiate() { return _tritonGraphReinstantiate.load(); }
  void setTritonGraphReinstantiate(bool v) { _tritonGraphReinstantiate.store(v); }
  bool tritonGraphAutoFree() { return _tritonGraphAutoFree.load(); }
  void setTritonGraphAutoFree(bool v) { _tritonGraphAutoFree.store(v); }
  bool tritonGraphDotVerbose() { return _tritonGraphDotVerbose.load(); }
  void setTritonGraphDotVerbose(bool v) { _tritonGraphDotVerbose.store(v); }

  // DSP batch-zero
  bool dspBatchZero() { return _dspBatchZero.load(); }
  void setDspBatchZero(bool v) { _dspBatchZero.store(v); }
  bool dspBatchZeroVerbose() { return _dspBatchZeroVerbose.load(); }
  void setDspBatchZeroVerbose(bool v) { _dspBatchZeroVerbose.store(v); }
  bool dspBatchZeroGapOnly() { return _dspBatchZeroGapOnly.load(); }
  void setDspBatchZeroGapOnly(bool v) { _dspBatchZeroGapOnly.store(v); }
  bool dspBatchZeroKernel() { return _dspBatchZeroKernel.load(); }
  void setDspBatchZeroKernel(bool v) { _dspBatchZeroKernel.store(v); }

  // DSP batched GEMM
  bool dspBatchedGemm() { return _dspBatchedGemm.load(); }
  void setDspBatchedGemm(bool v) { _dspBatchedGemm.store(v); }

  // Triton compilation scope
  bool tritonCompileAll() { return _tritonCompileAll.load(); }
  void setTritonCompileAll(bool v) { _tritonCompileAll.store(v); }
  std::string tritonExcludeOps() const { return _tritonExcludeOps; }
  void setTritonExcludeOps(const std::string& ops) { _tritonExcludeOps = ops; }
  bool isTritonExcludedOp(const std::string& opName) const;

  std::string tritonIncludeTypes() const { return _tritonIncludeTypes; }
  void setTritonIncludeTypes(const std::string& types) { _tritonIncludeTypes = types; }

  // Triton debugging flags
  bool tritonSkipKernels() { return _tritonSkipKernels.load(); }
  void setTritonSkipKernels(bool skip);
  bool tritonVerifyKernels() { return _tritonVerifyKernels.load(); }
  void setTritonVerifyKernels(bool verify);
  bool tritonVerifyKeepNative() { return _tritonVerifyKeepNative.load(); }
  void setTritonVerifyKeepNative(bool v);
  int tritonMaxSubKernelIndex() { return _tritonMaxSubKernelIndex.load(); }
  void setTritonMaxSubKernelIndex(int idx);
  bool tritonVerifyFullSnapshot() { return _tritonVerifyFullSnapshot.load(); }
  void setTritonVerifyFullSnapshot(bool v);
  bool tritonForceRecapture() { return _tritonForceRecapture.load(); }
  void setTritonForceRecapture(bool v);
  int tritonCaptureMinExec() { return _tritonCaptureMinExec.load(); }
  void setTritonCaptureMinExec(int v);

  // DSP optimization flags
  bool dspCastElimination() { return _dspCastElimination.load(); }
  void setDspCastElimination(bool enabled);
  bool dspMatmulSegmentation() { return _dspMatmulSegmentation.load(); }
  void setDspMatmulSegmentation(bool enabled);
  bool dspFp16Compute() { return _dspFp16Compute.load(); }
  void setDspFp16Compute(bool enabled);
  bool cublasTf32Enabled() { return _cublasTf32Enabled.load(); }
  void setCublasTf32Enabled(bool enabled);
  bool dspCastSinkMatmul() { return _dspCastSinkMatmul.load(); }
  void setDspCastSinkMatmul(bool enabled);
  bool tritonConsolidatedArgTable() { return _tritonConsolidatedArgTable.load(); }
  void setTritonConsolidatedArgTable(bool enabled);
  bool tritonArgDirtyTracking() { return _tritonArgDirtyTracking.load(); }
  void setTritonArgDirtyTracking(bool enabled);
  bool tritonSectionFusion() { return _tritonSectionFusion.load(); }
  void setTritonSectionFusion(bool enabled);

  // Fusion scoring
  bool tritonFusionScoring() { return _tritonFusionScoring.load(); }
  void setTritonFusionScoring(bool enabled);
  float tritonFusionMinScore() { return _tritonFusionMinScore.load(); }
  void setTritonFusionMinScore(float score);

  // Symbolic shape ranges
  bool dspSymbolicShapes() { return _dspSymbolicShapes.load(); }
  void setDspSymbolicShapes(bool enabled);
  int dspSymbolicShapeWarmup() { return _dspSymbolicShapeWarmup.load(); }
  void setDspSymbolicShapeWarmup(int steps);

  // Capture buffer pool sharing
  bool dspCapturePoolEnabled() { return _dspCapturePoolEnabled.load(); }
  void setDspCapturePoolEnabled(bool enabled);
  long long dspCapturePoolMaxBytes() { return _dspCapturePoolMaxBytes.load(); }
  void setDspCapturePoolMaxBytes(long long bytes);

  // Process environment path helpers used by native backends.
  std::string homeDirectory() const;
  std::string cudaToolkitPath() const;

  // Initialize CUDA environment settings from environment variables
  void initCudaEnvironment();

  // Initialize CUDA device limits from environment variables
  void initCudaDeviceLimits();
};
}  // namespace sd

#endif  // LIBND4J_ENVIRONMENT_H
