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
// Environment.h exposes subsystem config classes directly. Both C++ and Java
// callers access options via env.triton(), env.dsp(), env.core(), etc.
// New config options MUST be added to the subsystem classes; do NOT add flat
// forwarding wrappers on Environment for new options. The existing flat methods
// below are preserved only for backward compatibility with legacy callers.
//

#ifndef LIBND4J_ENVIRONMENT_H
#define LIBND4J_ENVIRONMENT_H

// Subsystem config classes
#include <system/config/CoreConfig.h>
#include <system/config/CudaDeviceConfig.h>
#include <system/config/TritonConfig.h>
#include <system/config/DspConfig.h>
#include <system/config/LifecycleConfig.h>
#include <system/config/PrintConfig.h>

#include <config.h>

namespace sd {

/**
 * Environment is the central configuration singleton for libnd4j.
 *
 * Configuration is organized into subsystem groups accessible via typed accessors:
 *   - core()      : verbose/debug, threading, memory, BLAS, data type
 *   - cuda()      : CUDA device settings, limits, tensor cores
 *   - triton()    : Triton compiler settings, CUDA graph integration
 *   - dsp()       : DSP batch-zero, GEMM, cast elimination, capture pool
 *   - lifecycle() : NDArray/DataBuffer lifecycle tracking
 *   - print()     : NDArray print options (NumPy-style)
 *
 * New code should use subsystem accessors: env.triton().buildThreads()
 * Legacy flat accessors (env.tritonBuildThreads()) are preserved for backward
 * compatibility and forward to the subsystem.
 */
class SD_LIB_EXPORT Environment {
 private:
  config::CoreConfig _core;
  config::CudaDeviceConfig _cuda;
  config::TritonConfig _triton;
  config::DspConfig _dsp;
  config::LifecycleConfig _lifecycle;
  config::PrintConfig _print;

#ifdef SD_EXPERIMENTAL_ENABLED
  const bool _experimental = true;
#else
  const bool _experimental = false;
#endif

  Environment();

 public:
  ~Environment();

  static Environment& getInstance();

  // ===== Subsystem accessors (stable API — new code should use these) =====
  // Exposed to both C++ and JavaCPP. Java callers: env.triton().setCacheEnabled(v).
  // New config options should be added to the subsystem classes directly; do NOT
  // add flat forwarding wrappers on Environment for new options.
  config::CoreConfig& core() { return _core; }
  config::CudaDeviceConfig& cuda() { return _cuda; }
  config::TritonConfig& triton() { return _triton; }
  config::DspConfig& dsp() { return _dsp; }
  config::LifecycleConfig& lifecycle() { return _lifecycle; }
  config::PrintConfig& print() { return _print; }

  // ===== Legacy forwarding methods (backward compat — all delegate to subsystems) =====
  // These exist so that all 93+ files that call env.isVerbose(), env.tritonBuildThreads(),
  // etc. continue to compile without any changes. In Phase 3, callers migrate to subsystem
  // accessors and these methods can be removed.

  // --- Core ---
  /**
   * These 3 fields are mostly for CUDA/cuBLAS version tracking
   */
  int _blasMajorVersion = 0;
  int _blasMinorVersion = 0;
  int _blasPatchVersion = 0;

  bool isEnableBlas() { return _core.isEnableBlas(); }
  void setEnableBlas(bool v) { _core.setEnableBlas(v); }
  bool isLogNativeNDArrayCreation() { return _core.isLogNativeNDArrayCreation(); }
  void setLogNativeNDArrayCreation(bool v) { _core.setLogNativeNDArrayCreation(v); }
  bool isLogNDArrayEvents() { return _core.isLogNDArrayEvents(); }
  void setLogNDArrayEvents(bool v) { _core.setLogNDArrayEvents(v); }
  bool isDeleteSpecial() { return _core.isDeleteSpecial(); }
  void setDeleteSpecial(bool v) { _core.setDeleteSpecial(v); }
  bool isDeletePrimary() { return _core.isDeletePrimary(); }
  void setDeletePrimary(bool v) { _core.setDeletePrimary(v); }
  bool isCheckOutputChange() { return _core.isCheckOutputChange(); }
  void setCheckOutputChange(bool v) { _core.setCheckOutputChange(v); }
  bool isCheckInputChange() { return _core.isCheckInputChange(); }
  void setCheckInputChange(bool v) { _core.setCheckInputChange(v); }
  bool isVerbose() { return _core.isVerbose(); }
  void setVerbose(bool v) { _core.setVerbose(v); }
  bool isDebug() { return _core.isDebug(); }
  bool isProfiling() { return _core.isProfiling(); }
  bool isDetectingLeaks() { return _core.isDetectingLeaks(); }
  bool isDebugAndVerbose() { return _core.isDebugAndVerbose(); }
  void setDebug(bool v) { _core.setDebug(v); }
  void setProfiling(bool v) { _core.setProfiling(v); }
  void setLeaksDetector(bool v) { _core.setLeaksDetector(v); }
  bool helpersAllowed() { return _core.helpersAllowed(); }
  void allowHelpers(bool v) { _core.allowHelpers(v); }
  bool blasFallback() { return _core.blasFallback(); }
  bool isSerializeBlasCalls() { return _core.isSerializeBlasCalls(); }
  void setSerializeBlasCalls(bool v) { _core.setSerializeBlasCalls(v); }
  int getOpenBlasThreads() { return _core.getOpenBlasThreads(); }
  void setOpenBlasThreads(int v) { _core.setOpenBlasThreads(v); }
  int tadThreshold() { return _core.tadThreshold(); }
  void setTadThreshold(int v) { _core.setTadThreshold(v); }
  int elementwiseThreshold() { return _core.elementwiseThreshold(); }
  void setElementwiseThreshold(int v) { _core.setElementwiseThreshold(v); }
  int maxThreads() { return _core.maxThreads(); }
  void setMaxThreads(int v) { _core.setMaxThreads(v); }
  int maxMasterThreads() { return _core.maxMasterThreads(); }
  void setMaxMasterThreads(int v) { _core.setMaxMasterThreads(v); }
  void setMaxPrimaryMemory(uint64_t v) { _core.setMaxPrimaryMemory(v); }
  void setMaxSpecialyMemory(uint64_t v) { _core.setMaxSpecialMemory(v); }
  void setMaxDeviceMemory(uint64_t v) { _core.setMaxDeviceMemory(v); }
  uint64_t maxPrimaryMemory() { return _core.maxPrimaryMemory(); }
  uint64_t maxSpecialMemory() { return _core.maxSpecialMemory(); }
  bool isUseONEDNN() { return _core.isUseONEDNN(); }
  void setUseONEDNN(bool v) { _core.setUseONEDNN(v); }
  bool isUseMPS() { return _core.isUseMPS(); }
  void setUseMPS(bool v) { _core.setUseMPS(v); }
  sd::DataType defaultFloatDataType() { return _core.defaultFloatDataType(); }
  void setDefaultFloatDataType(sd::DataType dtype) { _core.setDefaultFloatDataType(dtype); }
  bool precisionBoostAllowed() { return _core.precisionBoostAllowed(); }
  void allowPrecisionBoost(bool v) { _core.allowPrecisionBoost(v); }
  bool isExperimentalBuild() { return _experimental; }
  bool isCPU();
  bool isFuncTracePrintDeallocate() { return _core.isFuncTracePrintDeallocate(); }
  void setFuncTracePrintDeallocate(bool v) { _core.setFuncTracePrintDeallocate(v); }
  bool isFuncTracePrintAllocate() { return _core.isFuncTracePrintAllocate(); }
  void setFuncTracePrintAllocate(bool v) { _core.setFuncTracePrintAllocate(v); }
  bool isDeleteShapeInfo() { return _core.isDeleteShapeInfo(); }
  void setDeleteShapeInfo(bool v) { _core.setDeleteShapeInfo(v); }
  std::string homeDirectory() const { return _core.homeDirectory(); }
  std::string cudaToolkitPath() const { return _core.cudaToolkitPath(); }

  // Memory limits/counters (delegate to MemoryCounter — keep in Environment for JNI)
  void setGroupLimit(int group, sd::LongType numBytes);
  void setDeviceLimit(int deviceId, sd::LongType numBytes);
  sd::LongType getGroupLimit(int group);
  sd::LongType getDeviceLimit(int deviceId);
  sd::LongType getGroupCounter(int group);
  sd::LongType getDeviceCounter(int deviceId);

  int blasMajorVersion() { return _cuda.blasMajorVersion(); }
  int blasMinorVersion() { return _cuda.blasMinorVersion(); }
  int blasPatchVersion() { return _cuda.blasPatchVersion(); }
  std::vector<Pair>& capabilities() { return _cuda.capabilities(); }

  // --- CUDA Device Config ---
  int cudaDeviceCount() { return _cuda.cudaDeviceCount(); }
  int cudaCurrentDevice() { return _cuda.cudaCurrentDevice(); }
  void setCudaCurrentDevice(int device) { _cuda.setCudaCurrentDevice(device); }
  bool cudaMemoryPinned() { return _cuda.cudaMemoryPinned(); }
  void setCudaMemoryPinned(bool v) { _cuda.setCudaMemoryPinned(v); }
  bool cudaUseManagedMemory() { return _cuda.cudaUseManagedMemory(); }
  void setCudaUseManagedMemory(bool v) { _cuda.setCudaUseManagedMemory(v); }
  int cudaMemoryPoolSize() { return _cuda.cudaMemoryPoolSize(); }
  void setCudaMemoryPoolSize(int v) { _cuda.setCudaMemoryPoolSize(v); }
  bool cudaForceP2P() { return _cuda.cudaForceP2P(); }
  void setCudaForceP2P(bool v) { _cuda.setCudaForceP2P(v); }
  bool cudaAllocatorEnabled() { return _cuda.cudaAllocatorEnabled(); }
  void setCudaAllocatorEnabled(bool v) { _cuda.setCudaAllocatorEnabled(v); }
  int cudaMaxBlocks() { return _cuda.cudaMaxBlocks(); }
  void setCudaMaxBlocks(int v) { _cuda.setCudaMaxBlocks(v); }
  int cudaMaxThreadsPerBlock() { return _cuda.cudaMaxThreadsPerBlock(); }
  void setCudaMaxThreadsPerBlock(int v) { _cuda.setCudaMaxThreadsPerBlock(v); }
  bool cudaAsyncExecution() { return _cuda.cudaAsyncExecution(); }
  void setCudaAsyncExecution(bool v) { _cuda.setCudaAsyncExecution(v); }
  int cudaStreamLimit() { return _cuda.cudaStreamLimit(); }
  void setCudaStreamLimit(int v) { _cuda.setCudaStreamLimit(v); }
  bool cudaUseDeviceHost() { return _cuda.cudaUseDeviceHost(); }
  void setCudaUseDeviceHost(bool v) { _cuda.setCudaUseDeviceHost(v); }
  int cudaEventLimit() { return _cuda.cudaEventLimit(); }
  void setCudaEventLimit(int v) { _cuda.setCudaEventLimit(v); }
  int cudaCachingAllocatorLimit() { return _cuda.cudaCachingAllocatorLimit(); }
  void setCudaCachingAllocatorLimit(int v) { _cuda.setCudaCachingAllocatorLimit(v); }
  int64_t cudaPinnedHostLimit() { return _cuda.cudaPinnedHostLimit(); }
  void setCudaPinnedHostLimit(int64_t v) { _cuda.setCudaPinnedHostLimit(v); }
  int cudaSoftLimitPercent() { return _cuda.cudaSoftLimitPercent(); }
  void setCudaSoftLimitPercent(int v) { _cuda.setCudaSoftLimitPercent(v); }
  bool cudaUseUnifiedMemory() { return _cuda.cudaUseUnifiedMemory(); }
  void setCudaUseUnifiedMemory(bool v) { _cuda.setCudaUseUnifiedMemory(v); }
  int cudaPrefetchSize() { return _cuda.cudaPrefetchSize(); }
  void setCudaPrefetchSize(int v) { _cuda.setCudaPrefetchSize(v); }
  bool cudaGraphOptimization() { return _cuda.cudaGraphOptimization(); }
  void setCudaGraphOptimization(bool v) { _cuda.setCudaGraphOptimization(v); }
  bool cudaTensorCoreEnabled() { return _cuda.cudaTensorCoreEnabled(); }
  void setCudaTensorCoreEnabled(bool v) { _cuda.setCudaTensorCoreEnabled(v); }
  int cudaBlockingSync() { return _cuda.cudaBlockingSync(); }
  void setCudaBlockingSync(int v) { _cuda.setCudaBlockingSync(v); }
  int cudaDeviceSchedule() { return _cuda.cudaDeviceSchedule(); }
  void setCudaDeviceSchedule(int v) { _cuda.setCudaDeviceSchedule(v); }

  // CUDA Device Limits
  size_t cudaStackSize() { return _cuda.cudaStackSize(); }
  void setCudaStackSize(size_t v) { _cuda.setCudaStackSize(v); }
  size_t cudaMallocHeapSize() { return _cuda.cudaMallocHeapSize(); }
  void setCudaMallocHeapSize(size_t v) { _cuda.setCudaMallocHeapSize(v); }
  size_t cudaPrintfFifoSize() { return _cuda.cudaPrintfFifoSize(); }
  void setCudaPrintfFifoSize(size_t v) { _cuda.setCudaPrintfFifoSize(v); }
  size_t cudaDevRuntimeSyncDepth() { return _cuda.cudaDevRuntimeSyncDepth(); }
  void setCudaDevRuntimeSyncDepth(size_t v) { _cuda.setCudaDevRuntimeSyncDepth(v); }
  size_t cudaDevRuntimePendingLaunchCount() { return _cuda.cudaDevRuntimePendingLaunchCount(); }
  void setCudaDevRuntimePendingLaunchCount(size_t v) { _cuda.setCudaDevRuntimePendingLaunchCount(v); }
  size_t cudaMaxL2FetchGranularity() { return _cuda.cudaMaxL2FetchGranularity(); }
  void setCudaMaxL2FetchGranularity(size_t v) { _cuda.setCudaMaxL2FetchGranularity(v); }
  size_t cudaPersistingL2CacheSize() { return _cuda.cudaPersistingL2CacheSize(); }
  void setCudaPersistingL2CacheSize(size_t v) { _cuda.setCudaPersistingL2CacheSize(v); }
  bool setCudaDeviceLimit(int limitType, size_t value) { return _cuda.setCudaDeviceLimit(limitType, value); }

  // --- Print Config ---
  int printEdgeItems() { return _print.edgeItems(); }
  void setPrintEdgeItems(int v) { _print.setEdgeItems(v); }
  int printThreshold() { return _print.threshold(); }
  void setPrintThreshold(int v) { _print.setThreshold(v); }
  int printLineWidth() { return _print.lineWidth(); }
  void setPrintLineWidth(int v) { _print.setLineWidth(v); }
  int printPrecision() { return _print.precision(); }
  void setPrintPrecision(int v) { _print.setPrecision(v); }

  // --- Triton Config ---
  int tritonBuildThreads() { return _triton.buildThreads(); }
  void setTritonBuildThreads(int v) { _triton.setBuildThreads(v); }
  bool tritonCacheEnabled() { return _triton.cacheEnabled(); }
  void setTritonCacheEnabled(bool v) { _triton.setCacheEnabled(v); }
  // NOTE: new Triton config accessors should go directly through the
  // triton() subsystem (see subsystem accessors above). Do not add more
  // flat forwarding wrappers here — they exist only for legacy callers.
  bool tritonCooperativeLaunch() { return _triton.cooperativeLaunch(); }
  void setTritonCooperativeLaunch(bool v) { _triton.setCooperativeLaunch(v); }
  int tritonCoopTargetBlocks() { return _triton.coopTargetBlocks(); }
  void setTritonCoopTargetBlocks(int v) { _triton.setCoopTargetBlocks(v); }
  int tritonMaxSubsegmentOps() { return _triton.maxSubsegmentOps(); }
  void setTritonMaxSubsegmentOps(int v) { _triton.setMaxSubsegmentOps(v); }
  int tritonMaxSubsegmentSections() { return _triton.maxSubsegmentSections(); }
  void setTritonMaxSubsegmentSections(int v) { _triton.setMaxSubsegmentSections(v); }
  bool tritonVerbose() { return _triton.isVerbose(); }
  void setTritonVerbose(bool v) { _triton.setVerbose(v); }
  bool tritonDumpSections() { return _triton.dumpSections(); }
  void setTritonDumpSections(bool v) { _triton.setDumpSections(v); }
  bool tritonDumpArgs() { return _triton.dumpArgs(); }
  void setTritonDumpArgs(bool v) { _triton.setDumpArgs(v); }
  bool tritonLogAllPatterns() { return _triton.logAllPatterns(); }
  void setTritonLogAllPatterns(bool v) { _triton.setLogAllPatterns(v); }
  bool tritonAlwaysCompile() { return _triton.alwaysCompile(); }
  void setTritonAlwaysCompile(bool v) { _triton.setAlwaysCompile(v); }
  bool tritonInvalidateOnPlanFree() { return _triton.invalidateOnPlanFree(); }
  void setTritonInvalidateOnPlanFree(bool v) { _triton.setInvalidateOnPlanFree(v); }
  bool tritonKernelDump() { return _triton.kernelDump(); }
  void setTritonKernelDump(bool v) { _triton.setKernelDump(v); }
  bool tritonKernelOverride() { return _triton.kernelOverride(); }
  void setTritonKernelOverride(bool v) { _triton.setKernelOverride(v); }
  int tritonNumWarps() { return _triton.numWarps(); }
  void setTritonNumWarps(int v) { _triton.setNumWarps(v); }
  int tritonNumStages() { return _triton.numStages(); }
  void setTritonNumStages(int v) { _triton.setNumStages(v); }
  int tritonNumCTAs() { return _triton.numCTAs(); }
  void setTritonNumCTAs(int v) { _triton.setNumCTAs(v); }
  int tritonMaxNreg() { return _triton.maxNreg(); }
  void setTritonMaxNreg(int v) { _triton.setMaxNreg(v); }
  int tritonAttentionBlockN() { return _triton.attentionBlockN(); }
  void setTritonAttentionBlockN(int v) { _triton.setAttentionBlockN(v); }
  bool tritonEnableFpFusion() { return _triton.enableFpFusion(); }
  void setTritonEnableFpFusion(bool v) { _triton.setEnableFpFusion(v); }
  bool tritonDisableLineInfo() { return _triton.disableLineInfo(); }
  void setTritonDisableLineInfo(bool v) { _triton.setDisableLineInfo(v); }
  std::string tritonCacheDir() const { return _triton.cacheDir(); }
  void setTritonCacheDir(const std::string& v) { _triton.setCacheDir(v); }
  std::string tritonDumpDir() const { return _triton.dumpDir(); }
  void setTritonDumpDir(const std::string& v) { _triton.setDumpDir(v); }
  std::string tritonOverrideDir() const { return _triton.overrideDir(); }
  void setTritonOverrideDir(const std::string& v) { _triton.setOverrideDir(v); }
  std::string tritonOverrideArch() const { return _triton.overrideArch(); }
  void setTritonOverrideArch(const std::string& v) { _triton.setOverrideArch(v); }
  bool tritonAllowFallbackCapture() { return _triton.allowFallbackCapture(); }
  void setTritonAllowFallbackCapture(bool v) { _triton.setAllowFallbackCapture(v); }
  bool tritonGraphCapture() { return _triton.graphCapture(); }
  void setTritonGraphCapture(bool v) { _triton.setGraphCapture(v); }
  bool tritonDumpGraphDot() { return _triton.dumpGraphDot(); }
  void setTritonDumpGraphDot(bool v) { _triton.setDumpGraphDot(v); }
  bool tritonGraphCtxPush() { return _triton.graphCtxPush(); }
  void setTritonGraphCtxPush(bool v) { _triton.setGraphCtxPush(v); }
  bool tritonGraphReinstantiate() { return _triton.graphReinstantiate(); }
  void setTritonGraphReinstantiate(bool v) { _triton.setGraphReinstantiate(v); }
  bool tritonGraphAutoFree() { return _triton.graphAutoFree(); }
  void setTritonGraphAutoFree(bool v) { _triton.setGraphAutoFree(v); }
  bool tritonGraphDotVerbose() { return _triton.graphDotVerbose(); }
  void setTritonGraphDotVerbose(bool v) { _triton.setGraphDotVerbose(v); }
  bool tritonCompileAll() { return _triton.compileAll(); }
  void setTritonCompileAll(bool v) { _triton.setCompileAll(v); }
  std::string tritonExcludeOps() const { return _triton.excludeOps(); }
  void setTritonExcludeOps(const std::string& v) { _triton.setExcludeOps(v); }
  bool isTritonExcludedOp(const std::string& opName) const { return _triton.isExcludedOp(opName); }
  std::string tritonIncludeTypes() const { return _triton.includeTypes(); }
  void setTritonIncludeTypes(const std::string& v) { _triton.setIncludeTypes(v); }
  bool tritonFuseIdentityShapes() { return _triton.fuseIdentityShapes(); }
  void setTritonFuseIdentityShapes(bool v) { _triton.setFuseIdentityShapes(v); }
  bool tritonFuseCastChains() { return _triton.fuseCastChains(); }
  void setTritonFuseCastChains(bool v) { _triton.setFuseCastChains(v); }
  bool tritonSpecializePermuteSeq1() { return _triton.specializePermuteSeq1(); }
  void setTritonSpecializePermuteSeq1(bool v) { _triton.setSpecializePermuteSeq1(v); }
  bool tritonFusedMatmul() { return _triton.fusedMatmul(); }
  void setTritonFusedMatmul(bool v) { _triton.setFusedMatmul(v); }
  bool tritonFuseAttentionNeighborhoods() { return _triton.fuseAttentionNeighborhoods(); }
  void setTritonFuseAttentionNeighborhoods(bool v) { _triton.setFuseAttentionNeighborhoods(v); }
  bool tritonSkipKernels() { return _triton.skipKernels(); }
  void setTritonSkipKernels(bool v) { _triton.setSkipKernels(v); }
  bool tritonVerifyKernels() { return _triton.verifyKernels(); }
  void setTritonVerifyKernels(bool v) { _triton.setVerifyKernels(v); }
  bool tritonVerifyKeepNative() { return _triton.verifyKeepNative(); }
  void setTritonVerifyKeepNative(bool v) { _triton.setVerifyKeepNative(v); }
  int tritonMaxSubKernelIndex() { return _triton.maxSubKernelIndex(); }
  void setTritonMaxSubKernelIndex(int v) { _triton.setMaxSubKernelIndex(v); }
  bool tritonVerifyFullSnapshot() { return _triton.verifyFullSnapshot(); }
  void setTritonVerifyFullSnapshot(bool v) { _triton.setVerifyFullSnapshot(v); }
  bool tritonForceRecapture() { return _triton.forceRecapture(); }
  void setTritonForceRecapture(bool v) { _triton.setForceRecapture(v); }
  int tritonCaptureMinExec() { return _triton.captureMinExec(); }
  void setTritonCaptureMinExec(int v) { _triton.setCaptureMinExec(v); }
  bool tritonTf32Enabled() { return _triton.tf32Enabled(); }
  void setTritonTf32Enabled(bool v) { _triton.setTf32Enabled(v); }
  bool tritonConsolidatedArgTable() { return _triton.consolidatedArgTable(); }
  void setTritonConsolidatedArgTable(bool v) { _triton.setConsolidatedArgTable(v); }
  bool tritonArgDirtyTracking() { return _triton.argDirtyTracking(); }
  void setTritonArgDirtyTracking(bool v) { _triton.setArgDirtyTracking(v); }
  bool tritonSectionFusion() { return _triton.sectionFusion(); }
  void setTritonSectionFusion(bool v) { _triton.setSectionFusion(v); }
  bool tritonFusionScoring() { return _triton.fusionScoring(); }
  void setTritonFusionScoring(bool v) { _triton.setFusionScoring(v); }
  float tritonFusionMinScore() { return _triton.fusionMinScore(); }
  void setTritonFusionMinScore(float v) { _triton.setFusionMinScore(v); }
  bool tritonMergedCaptureThroughViews() { return _triton.mergedCaptureThroughViews(); }
  void setTritonMergedCaptureThroughViews(bool v) { _triton.setMergedCaptureThroughViews(v); }

  // --- DSP Config ---
  bool dspBatchZero() { return _dsp.batchZero(); }
  void setDspBatchZero(bool v) { _dsp.setBatchZero(v); }
  bool dspBatchZeroVerbose() { return _dsp.batchZeroVerbose(); }
  void setDspBatchZeroVerbose(bool v) { _dsp.setBatchZeroVerbose(v); }
  bool dspBatchZeroGapOnly() { return _dsp.batchZeroGapOnly(); }
  void setDspBatchZeroGapOnly(bool v) { _dsp.setBatchZeroGapOnly(v); }
  bool dspBatchZeroKernel() { return _dsp.batchZeroKernel(); }
  void setDspBatchZeroKernel(bool v) { _dsp.setBatchZeroKernel(v); }
  bool dspBatchedGemm() { return _dsp.batchedGemm(); }
  void setDspBatchedGemm(bool v) { _dsp.setBatchedGemm(v); }
  int dspTrimInterval() { return _dsp.trimInterval(); }
  void setDspTrimInterval(int v) { _dsp.setTrimInterval(v); }
  bool dspCastElimination() { return _dsp.castElimination(); }
  void setDspCastElimination(bool v) { _dsp.setCastElimination(v); }
  bool dspMatmulSegmentation() { return _dsp.matmulSegmentation(); }
  void setDspMatmulSegmentation(bool v) { _dsp.setMatmulSegmentation(v); }
  bool dspFp16Compute() { return _dsp.fp16Compute(); }
  void setDspFp16Compute(bool v) { _dsp.setFp16Compute(v); }
  bool cublasTf32Enabled() { return _dsp.cublasTf32Enabled(); }
  void setCublasTf32Enabled(bool v) { _dsp.setCublasTf32Enabled(v); }
  bool cublasCaptureWorkspace() { return _dsp.cublasCaptureWorkspace(); }
  void setCublasCaptureWorkspace(bool v) { _dsp.setCublasCaptureWorkspace(v); }
  bool dspCastSinkMatmul() { return _dsp.castSinkMatmul(); }
  void setDspCastSinkMatmul(bool v) { _dsp.setCastSinkMatmul(v); }
  bool dspGapTensorCores() { return _dsp.gapTensorCores(); }
  void setDspGapTensorCores(bool v) { _dsp.setGapTensorCores(v); }
  int dspMaxCapturableGapSlots() { return _dsp.maxCapturableGapSlots(); }
  void setDspMaxCapturableGapSlots(int v) { _dsp.setMaxCapturableGapSlots(v); }
  bool dspGapCaptureBlockExternalWorkspace() { return _dsp.gapCaptureBlockExternalWorkspace(); }
  void setDspGapCaptureBlockExternalWorkspace(bool v) { _dsp.setGapCaptureBlockExternalWorkspace(v); }
  int dspGapCaptureTensorCoreWarmup() { return _dsp.gapCaptureTensorCoreWarmup(); }
  void setDspGapCaptureTensorCoreWarmup(int v) { _dsp.setGapCaptureTensorCoreWarmup(v); }
  bool dspSymbolicShapes() { return _dsp.symbolicShapes(); }
  void setDspSymbolicShapes(bool v) { _dsp.setSymbolicShapes(v); }
  int dspSymbolicShapeWarmup() { return _dsp.symbolicShapeWarmup(); }
  void setDspSymbolicShapeWarmup(int v) { _dsp.setSymbolicShapeWarmup(v); }
  bool dspFreezeMergeSegments() { return _dsp.freezeMergeSegments(); }
  void setDspFreezeMergeSegments(bool v) { _dsp.setFreezeMergeSegments(v); }
  bool dspFreezeRecompile() { return _dsp.freezeRecompile(); }
  void setDspFreezeRecompile(bool v) { _dsp.setFreezeRecompile(v); }
  bool dspCapturePoolEnabled() { return _dsp.capturePoolEnabled(); }
  void setDspCapturePoolEnabled(bool v) { _dsp.setCapturePoolEnabled(v); }
  long long dspCapturePoolMaxBytes() { return _dsp.capturePoolMaxBytes(); }
  void setDspCapturePoolMaxBytes(long long v) { _dsp.setCapturePoolMaxBytes(v); }
  int dspCaptureOomMaxRetries() { return _dsp.captureOomMaxRetries(); }
  void setDspCaptureOomMaxRetries(int v) { _dsp.setCaptureOomMaxRetries(v); }
  int dspCaptureOomRetryInterval() { return _dsp.captureOomRetryInterval(); }
  void setDspCaptureOomRetryInterval(int v) { _dsp.setCaptureOomRetryInterval(v); }
  int dspCublasWorkspaceMb() { return _dsp.cublasWorkspaceMb(); }
  void setDspCublasWorkspaceMb(int v) { _dsp.setCublasWorkspaceMb(v); }
  int dspGraphMetadataSafetyMb() { return _dsp.graphMetadataSafetyMb(); }
  void setDspGraphMetadataSafetyMb(int v) { _dsp.setGraphMetadataSafetyMb(v); }
  bool dspProactiveEvictBeforeCapture() { return _dsp.proactiveEvictBeforeCapture(); }
  void setDspProactiveEvictBeforeCapture(bool v) { _dsp.setProactiveEvictBeforeCapture(v); }
  bool dspLruEviction() { return _dsp.lruEviction(); }
  void setDspLruEviction(bool v) { _dsp.setLruEviction(v); }

  // --- Lifecycle Config ---
  bool isLifecycleTracking() { return _lifecycle.isLifecycleTracking(); }
  void setLifecycleTracking(bool v) { _lifecycle.setLifecycleTracking(v); }
  bool isTrackViews() { return _lifecycle.isTrackViews(); }
  void setTrackViews(bool v) { _lifecycle.setTrackViews(v); }
  bool isTrackDeletions() { return _lifecycle.isTrackDeletions(); }
  void setTrackDeletions(bool v) { _lifecycle.setTrackDeletions(v); }
  int getStackDepth() { return _lifecycle.getStackDepth(); }
  void setStackDepth(int v) { _lifecycle.setStackDepth(v); }
  int getReportInterval() { return _lifecycle.getReportInterval(); }
  void setReportInterval(int v) { _lifecycle.setReportInterval(v); }
  size_t getMaxDeletionHistory() { return _lifecycle.getMaxDeletionHistory(); }
  void setMaxDeletionHistory(size_t v) { _lifecycle.setMaxDeletionHistory(v); }
  bool isSnapshotFiles() { return _lifecycle.isSnapshotFiles(); }
  void setSnapshotFiles(bool v) { _lifecycle.setSnapshotFiles(v); }
  bool isTrackOperations() { return _lifecycle.isTrackOperations(); }
  void setTrackOperations(bool v) { _lifecycle.setTrackOperations(v); }
  bool isNDArrayTracking() { return _lifecycle.isNDArrayTracking(); }
  void setNDArrayTracking(bool enabled);
  bool isDataBufferTracking() { return _lifecycle.isDataBufferTracking(); }
  void setDataBufferTracking(bool enabled);
  bool isTADCacheTracking() { return _lifecycle.isTADCacheTracking(); }
  void setTADCacheTracking(bool enabled);
  bool isShapeCacheTracking() { return _lifecycle.isShapeCacheTracking(); }
  void setShapeCacheTracking(bool enabled);
  bool isOpContextTracking() { return _lifecycle.isOpContextTracking(); }
  void setOpContextTracking(bool enabled);

  // Initialize CUDA environment settings from environment variables
  // (kept for backward compat — delegates to cuda().initFromEnvironment())
  void initCudaEnvironment() { _cuda.initFromEnvironment(); }
  void initCudaDeviceLimits() { _cuda.initCudaDeviceLimits(); }
};

}  // namespace sd

#endif  // LIBND4J_ENVIRONMENT_H
