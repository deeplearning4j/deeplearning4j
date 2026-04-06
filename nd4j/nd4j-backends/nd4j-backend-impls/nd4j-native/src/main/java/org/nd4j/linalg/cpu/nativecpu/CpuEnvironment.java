/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.cpu.nativecpu;

import org.bytedeco.javacpp.BytePointer;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu;
import org.nd4j.linalg.factory.Environment;

public class CpuEnvironment implements Environment {

    // CUDA limit type definitions
    public static final int
        CUDA_LIMIT_STACK_SIZE = 0,
        CUDA_LIMIT_MALLOC_HEAP_SIZE = 1,
        CUDA_LIMIT_PRINTF_FIFO_SIZE = 2,
        CUDA_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
        CUDA_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
        CUDA_LIMIT_MAX_L2_FETCH_GRANULARITY = 5,
        CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE = 6;

    private static final CpuEnvironment INSTANCE = new CpuEnvironment(Nd4jCpu.Environment.getInstance());
    protected boolean funcTracePrintJavaOnly = false;
    protected boolean workspaceTrackOpenClose = false;
    protected int numEventsToKeep = -1;
    private final Nd4jCpu.Environment e;

    protected boolean truncateNDArrayLongStrings = false;
    
    // Variable origin tracing flag for debugging import issues
    protected boolean variableTracingEnabled = false;

    public static CpuEnvironment getInstance(){
        return INSTANCE;
    }

    protected CpuEnvironment(Nd4jCpu.Environment environment) {
        this.e = environment;
    }

    @Override
    public boolean isEnableBlas() {
        return e.isEnableBlas();
    }

    @Override
    public void setEnableBlas(boolean reallyEnable) {
        e.setEnableBlas(reallyEnable);
    }

    @Override
    public boolean isLogNativeNDArrayCreation() {
        return e.isLogNativeNDArrayCreation();
    }

    @Override
    public void setLogNativeNDArrayCreation(boolean logNativeNDArrayCreation) {
        e.setLogNativeNDArrayCreation(logNativeNDArrayCreation);
    }

    @Override
    public boolean isCheckOutputChange() {
        return e.isCheckOutputChange();
    }

    @Override
    public void setCheckOutputChange(boolean reallyCheck) {
        e.setCheckOutputChange(reallyCheck);
    }

    @Override
    public boolean isCheckInputChange() {
        return e.isCheckInputChange();
    }

    @Override
    public void setCheckInputChange(boolean reallyCheck) {
        e.setCheckInputChange(reallyCheck);
    }

    @Override
    public void setLogNDArrayEvents(boolean logNDArrayEvents) {
        e.setLogNDArrayEvents(logNDArrayEvents);
    }

    @Override
    public boolean isLogNDArrayEvents() {
        return e.isLogNDArrayEvents();
    }

    @Override
    public boolean isTruncateNDArrayLogStrings() {
        return truncateNDArrayLongStrings;
    }

    @Override
    public void setTruncateLogStrings(boolean truncateLogStrings) {
        this.truncateNDArrayLongStrings = truncateLogStrings;
    }

    @Override
    public int numWorkspaceEventsToKeep() {
        return numEventsToKeep;
    }

    @Override
    public boolean isTrackWorkspaceOpenClose() {
        return workspaceTrackOpenClose;
    }

    @Override
    public void setTrackWorkspaceOpenClose(boolean trackWorkspaceOpenClose) {
        this.workspaceTrackOpenClose = trackWorkspaceOpenClose;
    }

    @Override
    public boolean isFuncTracePrintJavaOnly() {
        return funcTracePrintJavaOnly;
    }

    @Override
    public void setFuncTracePrintJavaOnly(boolean reallyTrace) {
        this.funcTracePrintJavaOnly = reallyTrace;
    }

    @Override
    public boolean isDeleteShapeInfo() {
        return e.isDeleteShapeInfo();
    }

    @Override
    public void setDeleteShapeInfo(boolean reallyDelete) {
        e.setDeleteShapeInfo(reallyDelete);
    }

    @Override
    public int blasMajorVersion() {
        return e.blasMajorVersion();
    }

    @Override
    public int blasMinorVersion() {
        return e.blasMinorVersion();
    }

    @Override
    public int blasPatchVersion() {
        return e.blasMajorVersion();
    }

    @Override
    public boolean isVerbose() {
        return e.isVerbose();
    }

    @Override
    public void setVerbose(boolean reallyVerbose) {
        e.setVerbose(reallyVerbose);
    }

    @Override
    public boolean isDebug() {
        return e.isDebug();
    }

    @Override
    public boolean isProfiling() {
        return e.isProfiling();
    }

    @Override
    public boolean isDetectingLeaks() {
        return e.isDetectingLeaks();
    }

    @Override
    public boolean isDebugAndVerbose() {
        return e.isDebugAndVerbose();
    }

    @Override
    public void setDebug(boolean reallyDebug) {
        e.setDebug(reallyDebug);
    }

    @Override
    public void setProfiling(boolean reallyProfile) {
        e.setProfiling(reallyProfile);
    }

    @Override
    public void setLeaksDetector(boolean reallyDetect) {
        e.setLeaksDetector(reallyDetect);
    }

    @Override
    public boolean helpersAllowed() {
        return e.helpersAllowed();
    }

    @Override
    public void allowHelpers(boolean reallyAllow) {
        e.allowHelpers(reallyAllow);
    }

    @Override
    public int tadThreshold() {
        return e.tadThreshold();
    }

    @Override
    public void setTadThreshold(int threshold) {
        e.setTadThreshold(threshold);
    }

    @Override
    public int elementwiseThreshold() {
        return e.elementwiseThreshold();
    }

    @Override
    public void setElementwiseThreshold(int threshold) {
        e.setElementwiseThreshold(threshold);
    }

    @Override
    public int maxThreads() {
        return e.maxThreads();
    }

    @Override
    public void setMaxThreads(int max) {
        e.setMaxThreads(max);
    }

    @Override
    public int maxMasterThreads() {
        return e.maxMasterThreads();
    }

    @Override
    public void setMaxMasterThreads(int max) {
        e.setMaxMasterThreads(max);
    }

    @Override
    public void setMaxPrimaryMemory(long maxBytes) {
        e.setMaxPrimaryMemory(maxBytes);
    }

    @Override
    public void setMaxSpecialMemory(long maxBytes) {
        e.setMaxSpecialyMemory(maxBytes);
    }

    @Override
    public void setMaxDeviceMemory(long maxBytes) {
        e.setMaxDeviceMemory(maxBytes);
    }

    @Override
    public boolean isCPU() {
        return e.isCPU();
    }

    @Override
    public void setGroupLimit(int group, long numBytes) {
        e.setGroupLimit(group, numBytes);
    }

    @Override
    public void setDeviceLimit(int deviceId, long numBytes) {
        e.setDeviceLimit(deviceId, numBytes);
    }

    @Override
    public long getGroupLimit(int group) {
        return e.getGroupLimit(group);
    }

    @Override
    public long getDeviceLimit(int deviceId) {
        return e.getDeviceLimit(deviceId);
    }

    @Override
    public long getDeviceCounter(int deviceId) {
        return e.getDeviceCounter(deviceId);
    }

    @Override
    public boolean isFuncTracePrintDeallocate() {
        return e.isFuncTracePrintDeallocate();
    }

    @Override
    public boolean isFuncTracePrintAllocate() {
        return e.isFuncTracePrintAllocate();
    }

    @Override
    public void setFuncTraceForDeallocate(boolean reallyTrace) {
        e.setFuncTracePrintDeallocate(reallyTrace);
    }

    @Override
    public void setFuncTraceForAllocate(boolean reallyTrace) {
        e.setFuncTracePrintAllocate(reallyTrace);
    }

    @Override
    public boolean isDeletePrimary() {
        return e.isDeletePrimary();
    }

    @Override
    public boolean isDeleteSpecial() {
        return e.isDeleteSpecial();
    }

    @Override
    public void setDeletePrimary(boolean reallyDelete) {
        e.setDeletePrimary(reallyDelete);
    }

    @Override
    public void setDeleteSpecial(boolean reallyDelete) {
        e.setDeleteSpecial(reallyDelete);
    }

    @Override
    public boolean isVariableTracingEnabled() {
        return variableTracingEnabled;
    }

    @Override
    public void setVariableTracingEnabled(boolean enabled) {
        this.variableTracingEnabled = enabled;
    }
    
    // CUDA specific methods implementation (no-op for CPU)
    
    @Override
    public int cudaDeviceCount() {
        return -1;
    }

    @Override
    public int cudaCurrentDevice() {
        return -1;
    }

    @Override
    public void setCudaCurrentDevice(int device) {
        // No-op for CPU
    }

    @Override
    public boolean cudaMemoryPinned() {
        return false;
    }

    @Override
    public void setCudaMemoryPinned(boolean pinned) {
        // No-op for CPU
    }

    @Override
    public boolean cudaUseManagedMemory() {
        return false;
    }

    @Override
    public void setCudaUseManagedMemory(boolean managed) {
        // No-op for CPU
    }

    @Override
    public int cudaMemoryPoolSize() {
        return -1;
    }

    @Override
    public void setCudaMemoryPoolSize(int sizeInMB) {
        // No-op for CPU
    }

    @Override
    public boolean cudaForceP2P() {
        return false;
    }

    @Override
    public void setCudaForceP2P(boolean forceP2P) {
        // No-op for CPU
    }

    @Override
    public boolean cudaAllocatorEnabled() {
        return false;
    }

    @Override
    public void setCudaAllocatorEnabled(boolean enabled) {
        // No-op for CPU
    }

    @Override
    public int cudaMaxBlocks() {
        return -1;
    }

    @Override
    public void setCudaMaxBlocks(int blocks) {
        // No-op for CPU
    }

    @Override
    public int cudaMaxThreadsPerBlock() {
        return -1;
    }

    @Override
    public void setCudaMaxThreadsPerBlock(int threads) {
        // No-op for CPU
    }

    @Override
    public boolean cudaAsyncExecution() {
        return false;
    }

    @Override
    public void setCudaAsyncExecution(boolean async) {
        // No-op for CPU
    }

    @Override
    public int cudaStreamLimit() {
        return -1;
    }

    @Override
    public void setCudaStreamLimit(int limit) {
        // No-op for CPU
    }

    @Override
    public boolean cudaUseDeviceHost() {
        return false;
    }

    @Override
    public void setCudaUseDeviceHost(boolean useDeviceHost) {
        // No-op for CPU
    }

    @Override
    public int cudaEventLimit() {
        return -1;
    }

    @Override
    public void setCudaEventLimit(int limit) {
        // No-op for CPU
    }

    @Override
    public int cudaCachingAllocatorLimit() {
        return -1;
    }

    @Override
    public void setCudaCachingAllocatorLimit(int limitInMB) {
        // No-op for CPU
    }

    @Override
    public long cudaPinnedHostLimit() {
        return -1;
    }

    @Override
    public void setCudaPinnedHostLimit(long limitInMB) {
        // No-op for CPU
    }

    @Override
    public boolean cudaUseUnifiedMemory() {
        return false;
    }

    @Override
    public void setCudaUseUnifiedMemory(boolean unified) {
        // No-op for CPU
    }

    @Override
    public int cudaPrefetchSize() {
        return -1;
    }

    @Override
    public void setCudaPrefetchSize(int sizeInMB) {
        // No-op for CPU
    }

    @Override
    public boolean cudaGraphOptimization() {
        return false;
    }

    @Override
    public void setCudaGraphOptimization(boolean enabled) {
        // No-op for CPU
    }

    @Override
    public boolean cudaTensorCoreEnabled() {
        return false;
    }

    @Override
    public void setCudaTensorCoreEnabled(boolean enabled) {
        // No-op for CPU
    }

    @Override
    public int cudaBlockingSync() {
        return -1;
    }

    @Override
    public void setCudaBlockingSync(int mode) {
        // No-op for CPU
    }

    @Override
    public int cudaDeviceSchedule() {
        return -1;
    }

    @Override
    public void setCudaDeviceSchedule(int schedule) {
        // No-op for CPU
    }

    @Override
    public long cudaStackSize() {
        return -1;
    }

    @Override
    public void setCudaStackSize(long size) {
        // No-op for CPU
    }

    @Override
    public long cudaMallocHeapSize() {
        return -1;
    }

    @Override
    public void setCudaMallocHeapSize(long size) {
        // No-op for CPU
    }

    @Override
    public long cudaPrintfFifoSize() {
        return -1;
    }

    @Override
    public void setCudaPrintfFifoSize(long size) {
        // No-op for CPU
    }

    @Override
    public long cudaDevRuntimeSyncDepth() {
        return -1;
    }

    @Override
    public void setCudaDevRuntimeSyncDepth(long depth) {
        // No-op for CPU
    }

    @Override
    public long cudaDevRuntimePendingLaunchCount() {
        return -1;
    }

    @Override
    public void setCudaDevRuntimePendingLaunchCount(long count) {
        // No-op for CPU
    }

    @Override
    public long cudaMaxL2FetchGranularity() {
        return -1;
    }

    @Override
    public void setCudaMaxL2FetchGranularity(long size) {
        // No-op for CPU
    }

    @Override
    public long cudaPersistingL2CacheSize() {
        return -1;
    }

    @Override
    public void setCudaPersistingL2CacheSize(long size) {
        // No-op for CPU
    }
    
    @Override
    public int setCudaDeviceLimit(int limitType, long value) {
        // No-op for CPU
        return 0; // Return 0 to indicate operation not supported in CPU mode
    }

    // Lifecycle tracking methods (delegated to native Environment)

    @Override
    public boolean isLifecycleTracking() {
        return e.isLifecycleTracking();
    }

    @Override
    public void setLifecycleTracking(boolean enabled) {
        e.setLifecycleTracking(enabled);
    }

    @Override
    public boolean isTrackViews() {
        return e.isTrackViews();
    }

    @Override
    public void setTrackViews(boolean track) {
        e.setTrackViews(track);
    }

    @Override
    public boolean isTrackDeletions() {
        return e.isTrackDeletions();
    }

    @Override
    public void setTrackDeletions(boolean track) {
        e.setTrackDeletions(track);
    }

    @Override
    public int getStackDepth() {
        return e.getStackDepth();
    }

    @Override
    public void setStackDepth(int depth) {
        e.setStackDepth(depth);
    }

    @Override
    public int getReportInterval() {
        return e.getReportInterval();
    }

    @Override
    public void setReportInterval(int seconds) {
        e.setReportInterval(seconds);
    }

    @Override
    public long getMaxDeletionHistory() {
        return e.getMaxDeletionHistory();
    }

    @Override
    public void setMaxDeletionHistory(long max) {
        e.setMaxDeletionHistory(max);
    }

    @Override
    public boolean isSnapshotFiles() {
        return e.isSnapshotFiles();
    }

    @Override
    public void setSnapshotFiles(boolean enabled) {
        e.setSnapshotFiles(enabled);
    }

    @Override
    public boolean isTrackOperations() {
        return e.isTrackOperations();
    }

    @Override
    public void setTrackOperations(boolean enabled) {
        e.setTrackOperations(enabled);
    }

    @Override
    public boolean isNDArrayTracking() {
        return e.isNDArrayTracking();
    }

    @Override
    public void setNDArrayTracking(boolean enabled) {
        e.setNDArrayTracking(enabled);
    }

    @Override
    public boolean isDataBufferTracking() {
        return e.isDataBufferTracking();
    }

    @Override
    public void setDataBufferTracking(boolean enabled) {
        e.setDataBufferTracking(enabled);
    }

    @Override
    public boolean isTADCacheTracking() {
        return e.isTADCacheTracking();
    }

    @Override
    public void setTADCacheTracking(boolean enabled) {
        e.setTADCacheTracking(enabled);
    }

    @Override
    public boolean isShapeCacheTracking() {
        return e.isShapeCacheTracking();
    }

    @Override
    public void setShapeCacheTracking(boolean enabled) {
        e.setShapeCacheTracking(enabled);
    }

    @Override
    public boolean isOpContextTracking() {
        return e.isOpContextTracking();
    }

    @Override
    public void setOpContextTracking(boolean enabled) {
        e.setOpContextTracking(enabled);
    }

    // ─── Triton settings (delegated to native C++ Environment) ─────────────────
    @Override public int tritonBuildThreads() { return e.tritonBuildThreads(); }
    @Override public void setTritonBuildThreads(int v) { e.setTritonBuildThreads(v); }
    @Override public boolean tritonCacheEnabled() { return e.tritonCacheEnabled(); }
    @Override public void setTritonCacheEnabled(boolean v) { e.setTritonCacheEnabled(v); }
    @Override public boolean tritonCooperativeLaunch() { return e.tritonCooperativeLaunch(); }
    @Override public void setTritonCooperativeLaunch(boolean v) { e.setTritonCooperativeLaunch(v); }
    @Override public int tritonCoopTargetBlocks() { return e.tritonCoopTargetBlocks(); }
    @Override public void setTritonCoopTargetBlocks(int v) { e.setTritonCoopTargetBlocks(v); }
    @Override public int tritonMaxSubsegmentOps() { return e.tritonMaxSubsegmentOps(); }
    @Override public void setTritonMaxSubsegmentOps(int v) { e.setTritonMaxSubsegmentOps(v); }
    @Override public int tritonMaxSubsegmentSections() { return e.tritonMaxSubsegmentSections(); }
    @Override public void setTritonMaxSubsegmentSections(int v) { e.setTritonMaxSubsegmentSections(v); }
    @Override public boolean tritonVerbose() { return e.tritonVerbose(); }
    @Override public void setTritonVerbose(boolean v) { e.setTritonVerbose(v); }
    @Override public boolean tritonDumpSections() { return e.tritonDumpSections(); }
    @Override public void setTritonDumpSections(boolean v) { e.setTritonDumpSections(v); }
    @Override public boolean tritonDumpArgs() { return e.tritonDumpArgs(); }
    @Override public void setTritonDumpArgs(boolean v) { e.setTritonDumpArgs(v); }
    @Override public boolean tritonLogAllPatterns() { return e.tritonLogAllPatterns(); }
    @Override public void setTritonLogAllPatterns(boolean v) { e.setTritonLogAllPatterns(v); }
    @Override public boolean tritonAlwaysCompile() { return e.tritonAlwaysCompile(); }
    @Override public void setTritonAlwaysCompile(boolean v) { e.setTritonAlwaysCompile(v); }
    @Override public boolean tritonKernelDump() { return e.tritonKernelDump(); }
    @Override public void setTritonKernelDump(boolean v) { e.setTritonKernelDump(v); }
    @Override public boolean tritonKernelOverride() { return e.tritonKernelOverride(); }
    @Override public void setTritonKernelOverride(boolean v) { e.setTritonKernelOverride(v); }
    @Override public int tritonNumWarps() { return e.tritonNumWarps(); }
    @Override public void setTritonNumWarps(int v) { e.setTritonNumWarps(v); }
    @Override public int tritonNumStages() { return e.tritonNumStages(); }
    @Override public void setTritonNumStages(int v) { e.setTritonNumStages(v); }
    @Override public int tritonNumCTAs() { return e.tritonNumCTAs(); }
    @Override public void setTritonNumCTAs(int v) { e.setTritonNumCTAs(v); }
    @Override public int tritonMaxNreg() { return e.tritonMaxNreg(); }
    @Override public void setTritonMaxNreg(int v) { e.setTritonMaxNreg(v); }
    @Override public int tritonAttentionBlockN() { return e.tritonAttentionBlockN(); }
    @Override public void setTritonAttentionBlockN(int v) { e.setTritonAttentionBlockN(v); }
    @Override public boolean tritonEnableFpFusion() { return e.tritonEnableFpFusion(); }
    @Override public void setTritonEnableFpFusion(boolean v) { e.setTritonEnableFpFusion(v); }
    @Override public boolean tritonDisableLineInfo() { return e.tritonDisableLineInfo(); }
    @Override public void setTritonDisableLineInfo(boolean v) { e.setTritonDisableLineInfo(v); }
    @Override public String tritonCacheDir() { BytePointer p = e.tritonCacheDir(); return p == null ? "" : p.getString(); }
    @Override public void setTritonCacheDir(String v) { e.setTritonCacheDir(v); }
    @Override public String tritonDumpDir() { BytePointer p = e.tritonDumpDir(); return p == null ? "" : p.getString(); }
    @Override public void setTritonDumpDir(String v) { e.setTritonDumpDir(v); }
    @Override public String tritonOverrideDir() { BytePointer p = e.tritonOverrideDir(); return p == null ? "" : p.getString(); }
    @Override public void setTritonOverrideDir(String v) { e.setTritonOverrideDir(v); }
    @Override public String tritonOverrideArch() { BytePointer p = e.tritonOverrideArch(); return p == null ? "" : p.getString(); }
    @Override public void setTritonOverrideArch(String v) { e.setTritonOverrideArch(v); }
    @Override public boolean tritonAllowFallbackCapture() { return e.tritonAllowFallbackCapture(); }
    @Override public void setTritonAllowFallbackCapture(boolean v) { e.setTritonAllowFallbackCapture(v); }
    @Override public boolean tritonGraphCapture() { return e.tritonGraphCapture(); }
    @Override public void setTritonGraphCapture(boolean v) { e.setTritonGraphCapture(v); }
    @Override public boolean tritonDumpGraphDot() { return e.tritonDumpGraphDot(); }
    @Override public void setTritonDumpGraphDot(boolean v) { e.setTritonDumpGraphDot(v); }
    @Override public boolean tritonSkipKernels() { return e.tritonSkipKernels(); }
    @Override public void setTritonSkipKernels(boolean v) { e.setTritonSkipKernels(v); }
    @Override public boolean tritonVerifyKernels() { return e.tritonVerifyKernels(); }
    @Override public void setTritonVerifyKernels(boolean v) { e.setTritonVerifyKernels(v); }
    @Override public boolean tritonVerifyKeepNative() { return e.tritonVerifyKeepNative(); }
    @Override public void setTritonVerifyKeepNative(boolean v) { e.setTritonVerifyKeepNative(v); }
    @Override public int tritonMaxSubKernelIndex() { return e.tritonMaxSubKernelIndex(); }
    @Override public void setTritonMaxSubKernelIndex(int v) { e.setTritonMaxSubKernelIndex(v); }
    @Override public boolean tritonVerifyFullSnapshot() { return e.tritonVerifyFullSnapshot(); }
    @Override public void setTritonVerifyFullSnapshot(boolean v) { e.setTritonVerifyFullSnapshot(v); }
    @Override public boolean tritonForceRecapture() { return e.tritonForceRecapture(); }
    @Override public void setTritonForceRecapture(boolean v) { e.setTritonForceRecapture(v); }
    @Override public int tritonCaptureMinExec() { return e.tritonCaptureMinExec(); }
    @Override public void setTritonCaptureMinExec(int v) { e.setTritonCaptureMinExec(v); }
    @Override public boolean tritonCompileAll() { return e.tritonCompileAll(); }
    @Override public void setTritonCompileAll(boolean v) { e.setTritonCompileAll(v); }
    @Override public String tritonExcludeOps() { BytePointer p = e.tritonExcludeOps(); return p == null ? "" : p.getString(); }
    @Override public void setTritonExcludeOps(String v) { e.setTritonExcludeOps(v); }
    @Override public String tritonIncludeTypes() { BytePointer p = e.tritonIncludeTypes(); return p == null ? "" : p.getString(); }
    @Override public void setTritonIncludeTypes(String v) { e.setTritonIncludeTypes(v); }
    @Override public boolean tritonFusionScoring() { return e.tritonFusionScoring(); }
    @Override public void setTritonFusionScoring(boolean v) { e.setTritonFusionScoring(v); }
    @Override public float tritonFusionMinScore() { return e.tritonFusionMinScore(); }
    @Override public void setTritonFusionMinScore(float v) { e.setTritonFusionMinScore(v); }
    @Override public boolean tritonFuseIdentityShapes() { return e.tritonFuseIdentityShapes(); }
    @Override public void setTritonFuseIdentityShapes(boolean v) { e.setTritonFuseIdentityShapes(v); }
    @Override public boolean tritonFuseCastChains() { return e.tritonFuseCastChains(); }
    @Override public void setTritonFuseCastChains(boolean v) { e.setTritonFuseCastChains(v); }
    @Override public boolean tritonSpecializePermuteSeq1() { return e.tritonSpecializePermuteSeq1(); }
    @Override public void setTritonSpecializePermuteSeq1(boolean v) { e.setTritonSpecializePermuteSeq1(v); }
    @Override public boolean tritonFusedMatmul() { return e.tritonFusedMatmul(); }
    @Override public void setTritonFusedMatmul(boolean v) { e.setTritonFusedMatmul(v); }
    @Override public boolean tritonFuseAttentionNeighborhoods() { return e.tritonFuseAttentionNeighborhoods(); }
    @Override public void setTritonFuseAttentionNeighborhoods(boolean v) { e.setTritonFuseAttentionNeighborhoods(v); }
    @Override public boolean tritonSectionFusion() { return e.tritonSectionFusion(); }
    @Override public void setTritonSectionFusion(boolean v) { e.setTritonSectionFusion(v); }
    @Override public boolean tritonConsolidatedArgTable() { return e.tritonConsolidatedArgTable(); }
    @Override public void setTritonConsolidatedArgTable(boolean v) { e.setTritonConsolidatedArgTable(v); }
    @Override public boolean tritonArgDirtyTracking() { return e.tritonArgDirtyTracking(); }
    @Override public void setTritonArgDirtyTracking(boolean v) { e.setTritonArgDirtyTracking(v); }

    // ─── DSP settings ──────────────────────────────────────────────────────────
    @Override public boolean dspBatchZero() { return e.dspBatchZero(); }
    @Override public void setDspBatchZero(boolean v) { e.setDspBatchZero(v); }
    @Override public boolean dspBatchZeroVerbose() { return e.dspBatchZeroVerbose(); }
    @Override public void setDspBatchZeroVerbose(boolean v) { e.setDspBatchZeroVerbose(v); }
    @Override public boolean dspBatchZeroGapOnly() { return e.dspBatchZeroGapOnly(); }
    @Override public void setDspBatchZeroGapOnly(boolean v) { e.setDspBatchZeroGapOnly(v); }
    @Override public boolean dspBatchZeroKernel() { return e.dspBatchZeroKernel(); }
    @Override public void setDspBatchZeroKernel(boolean v) { e.setDspBatchZeroKernel(v); }
    @Override public boolean dspBatchedGemm() { return e.dspBatchedGemm(); }
    @Override public void setDspBatchedGemm(boolean v) { e.setDspBatchedGemm(v); }
    @Override public boolean dspCastElimination() { return e.dspCastElimination(); }
    @Override public void setDspCastElimination(boolean v) { e.setDspCastElimination(v); }
    @Override public boolean dspMatmulSegmentation() { return e.dspMatmulSegmentation(); }
    @Override public void setDspMatmulSegmentation(boolean v) { e.setDspMatmulSegmentation(v); }
    @Override public boolean dspFp16Compute() { return e.dspFp16Compute(); }
    @Override public void setDspFp16Compute(boolean v) { e.setDspFp16Compute(v); }
    @Override public boolean cublasTf32Enabled() { return e.cublasTf32Enabled(); }
    @Override public void setCublasTf32Enabled(boolean v) { e.setCublasTf32Enabled(v); }
    @Override public boolean tritonTf32Enabled() { return e.tritonTf32Enabled(); }
    @Override public void setTritonTf32Enabled(boolean v) { e.setTritonTf32Enabled(v); }
    @Override public boolean cublasCaptureWorkspace() { return e.cublasCaptureWorkspace(); }
    @Override public void setCublasCaptureWorkspace(boolean v) { e.setCublasCaptureWorkspace(v); }
    @Override public boolean dspCastSinkMatmul() { return e.dspCastSinkMatmul(); }
    @Override public void setDspCastSinkMatmul(boolean v) { e.setDspCastSinkMatmul(v); }
    @Override public boolean dspFreezeMergeSegments() { return e.dspFreezeMergeSegments(); }
    @Override public void setDspFreezeMergeSegments(boolean v) { e.setDspFreezeMergeSegments(v); }
    @Override public boolean dspFreezeRecompile() { return e.dspFreezeRecompile(); }
    @Override public void setDspFreezeRecompile(boolean v) { e.setDspFreezeRecompile(v); }
}
