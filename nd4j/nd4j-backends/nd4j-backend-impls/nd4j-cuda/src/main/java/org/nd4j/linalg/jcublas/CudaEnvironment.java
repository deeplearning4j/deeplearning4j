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
package org.nd4j.linalg.jcublas;

import org.bytedeco.javacpp.BytePointer;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.jcublas.bindings.Nd4jCuda;

/**
 * CUDA backend implementation of {@link Environment}
 *
 * @author Alex Black
 */
public class CudaEnvironment implements Environment {

    // CUDA limit type definitions
    public static final int
        CUDA_LIMIT_STACK_SIZE = 0,
        CUDA_LIMIT_MALLOC_HEAP_SIZE = 1,
        CUDA_LIMIT_PRINTF_FIFO_SIZE = 2,
        CUDA_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
        CUDA_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
        CUDA_LIMIT_MAX_L2_FETCH_GRANULARITY = 5,
        CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE = 6;

    private static final CudaEnvironment INSTANCE = new CudaEnvironment(Nd4jCuda.Environment.getInstance());
    protected boolean funcTracePrintJavaOnly = false;
    protected boolean workspaceTrackOpenClose = false;
    protected int numEventsToKeep = -1;
    protected boolean truncateNDArrayLongStrings = false;

    // Variable origin tracing flag for debugging import issues
    protected boolean variableTracingEnabled = false;

    private final Nd4jCuda.Environment e;
    public static CudaEnvironment getInstance(){
        return INSTANCE;
    }

    protected CudaEnvironment(Nd4jCuda.Environment environment){
        this.e = environment;
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

    // CUDA specific methods
    
    @Override
    public int cudaDeviceCount() {
        return e.cudaDeviceCount();
    }

    @Override
    public int cudaCurrentDevice() {
        return e.cudaCurrentDevice();
    }

    @Override
    public void setCudaCurrentDevice(int device) {
        e.setCudaCurrentDevice(device);
    }

    @Override
    public boolean cudaMemoryPinned() {
        return e.cudaMemoryPinned();
    }

    @Override
    public void setCudaMemoryPinned(boolean pinned) {
        e.setCudaMemoryPinned(pinned);
    }

    @Override
    public boolean cudaUseManagedMemory() {
        return e.cudaUseManagedMemory();
    }

    @Override
    public void setCudaUseManagedMemory(boolean managed) {
        e.setCudaUseManagedMemory(managed);
    }

    @Override
    public int cudaMemoryPoolSize() {
        return e.cudaMemoryPoolSize();
    }

    @Override
    public void setCudaMemoryPoolSize(int sizeInMB) {
        e.setCudaMemoryPoolSize(sizeInMB);
    }

    @Override
    public boolean cudaForceP2P() {
        return e.cudaForceP2P();
    }

    @Override
    public void setCudaForceP2P(boolean forceP2P) {
        e.setCudaForceP2P(forceP2P);
    }

    @Override
    public boolean cudaAllocatorEnabled() {
        return e.cudaAllocatorEnabled();
    }

    @Override
    public void setCudaAllocatorEnabled(boolean enabled) {
        e.setCudaAllocatorEnabled(enabled);
    }

    @Override
    public int cudaMaxBlocks() {
        return e.cudaMaxBlocks();
    }

    @Override
    public void setCudaMaxBlocks(int blocks) {
        e.setCudaMaxBlocks(blocks);
    }

    @Override
    public int cudaMaxThreadsPerBlock() {
        return e.cudaMaxThreadsPerBlock();
    }

    @Override
    public void setCudaMaxThreadsPerBlock(int threads) {
        e.setCudaMaxThreadsPerBlock(threads);
    }

    @Override
    public boolean cudaAsyncExecution() {
        return e.cudaAsyncExecution();
    }

    @Override
    public void setCudaAsyncExecution(boolean async) {
        e.setCudaAsyncExecution(async);
    }

    @Override
    public int cudaStreamLimit() {
        return e.cudaStreamLimit();
    }

    @Override
    public void setCudaStreamLimit(int limit) {
        e.setCudaStreamLimit(limit);
    }

    @Override
    public boolean cudaUseDeviceHost() {
        return e.cudaUseDeviceHost();
    }

    @Override
    public void setCudaUseDeviceHost(boolean useDeviceHost) {
        e.setCudaUseDeviceHost(useDeviceHost);
    }

    @Override
    public int cudaEventLimit() {
        return e.cudaEventLimit();
    }

    @Override
    public void setCudaEventLimit(int limit) {
        e.setCudaEventLimit(limit);
    }

    @Override
    public int cudaCachingAllocatorLimit() {
        return e.cudaCachingAllocatorLimit();
    }

    @Override
    public void setCudaCachingAllocatorLimit(int limitInMB) {
        e.setCudaCachingAllocatorLimit(limitInMB);
    }

    @Override
    public long cudaPinnedHostLimit() {
        return e.cudaPinnedHostLimit();
    }

    @Override
    public void setCudaPinnedHostLimit(long limitInMB) {
        e.setCudaPinnedHostLimit(limitInMB);
    }

    @Override
    public boolean cudaUseUnifiedMemory() {
        return e.cudaUseUnifiedMemory();
    }

    @Override
    public void setCudaUseUnifiedMemory(boolean unified) {
        e.setCudaUseUnifiedMemory(unified);
    }

    @Override
    public int cudaPrefetchSize() {
        return e.cudaPrefetchSize();
    }

    @Override
    public void setCudaPrefetchSize(int sizeInMB) {
        e.setCudaPrefetchSize(sizeInMB);
    }

    @Override
    public boolean cudaGraphOptimization() {
        return e.cudaGraphOptimization();
    }

    @Override
    public void setCudaGraphOptimization(boolean enabled) {
        e.setCudaGraphOptimization(enabled);
    }

    @Override
    public boolean cudaTensorCoreEnabled() {
        return e.cudaTensorCoreEnabled();
    }

    @Override
    public void setCudaTensorCoreEnabled(boolean enabled) {
        e.setCudaTensorCoreEnabled(enabled);
    }

    @Override
    public int cudaBlockingSync() {
        return e.cudaBlockingSync();
    }

    @Override
    public void setCudaBlockingSync(int mode) {
        e.setCudaBlockingSync(mode);
    }

    @Override
    public int cudaDeviceSchedule() {
        return e.cudaDeviceSchedule();
    }

    @Override
    public void setCudaDeviceSchedule(int schedule) {
        e.setCudaDeviceSchedule(schedule);
    }

    @Override
    public long cudaStackSize() {
        return e.cudaStackSize();
    }

    @Override
    public void setCudaStackSize(long size) {
        e.setCudaStackSize(size);
    }

    @Override
    public long cudaMallocHeapSize() {
        return e.cudaMallocHeapSize();
    }

    @Override
    public void setCudaMallocHeapSize(long size) {
        e.setCudaMallocHeapSize(size);
    }

    @Override
    public long cudaPrintfFifoSize() {
        return e.cudaPrintfFifoSize();
    }

    @Override
    public void setCudaPrintfFifoSize(long size) {
        e.setCudaPrintfFifoSize(size);
    }

    @Override
    public long cudaDevRuntimeSyncDepth() {
        return e.cudaDevRuntimeSyncDepth();
    }

    @Override
    public void setCudaDevRuntimeSyncDepth(long depth) {
        e.setCudaDevRuntimeSyncDepth(depth);
    }

    @Override
    public long cudaDevRuntimePendingLaunchCount() {
        return e.cudaDevRuntimePendingLaunchCount();
    }

    @Override
    public void setCudaDevRuntimePendingLaunchCount(long count) {
        e.setCudaDevRuntimePendingLaunchCount(count);
    }

    @Override
    public long cudaMaxL2FetchGranularity() {
        return e.cudaMaxL2FetchGranularity();
    }

    @Override
    public void setCudaMaxL2FetchGranularity(long size) {
        e.setCudaMaxL2FetchGranularity(size);
    }

    @Override
    public long cudaPersistingL2CacheSize() {
        return e.cudaPersistingL2CacheSize();
    }

    @Override
    public void setCudaPersistingL2CacheSize(long size) {
        e.setCudaPersistingL2CacheSize(size);
    }

    @Override
    public int setCudaDeviceLimit(int limitType, long value) {
        switch (limitType) {
            case CUDA_LIMIT_STACK_SIZE:
                setCudaStackSize(value);
                break;
            case CUDA_LIMIT_MALLOC_HEAP_SIZE:
                setCudaMallocHeapSize(value);
                break;
            case CUDA_LIMIT_PRINTF_FIFO_SIZE:
                setCudaPrintfFifoSize(value);
                break;
            case CUDA_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
                setCudaDevRuntimeSyncDepth(value);
                break;
            case CUDA_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
                setCudaDevRuntimePendingLaunchCount(value);
                break;
            case CUDA_LIMIT_MAX_L2_FETCH_GRANULARITY:
                setCudaMaxL2FetchGranularity(value);
                break;
            case CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE:
                setCudaPersistingL2CacheSize(value);
                break;
            default:
                return -1; // Unsupported limit type
        }
        return 0; // Success
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

    // Triton GPU settings
    @Override
    public int tritonBuildThreads() {
        return e.tritonBuildThreads();
    }

    @Override
    public void setTritonBuildThreads(int threads) {
        e.setTritonBuildThreads(threads);
    }

    @Override
    public boolean tritonCacheEnabled() {
        return e.tritonCacheEnabled();
    }

    @Override
    public void setTritonCacheEnabled(boolean enabled) {
        e.setTritonCacheEnabled(enabled);
    }

    @Override
    public boolean tritonCooperativeLaunch() {
        return e.tritonCooperativeLaunch();
    }

    @Override
    public void setTritonCooperativeLaunch(boolean enabled) {
        e.setTritonCooperativeLaunch(enabled);
    }

    @Override
    public int tritonCoopTargetBlocks() {
        return e.tritonCoopTargetBlocks();
    }

    @Override
    public void setTritonCoopTargetBlocks(int blocks) {
        e.setTritonCoopTargetBlocks(blocks);
    }

    @Override
    public int tritonMaxSubsegmentOps() {
        return e.tritonMaxSubsegmentOps();
    }

    @Override
    public void setTritonMaxSubsegmentOps(int ops) {
        e.setTritonMaxSubsegmentOps(ops);
    }

    @Override
    public int tritonMaxSubsegmentSections() {
        return e.tritonMaxSubsegmentSections();
    }

    @Override
    public void setTritonMaxSubsegmentSections(int sections) {
        e.setTritonMaxSubsegmentSections(sections);
    }

    @Override
    public boolean tritonVerbose() {
        return e.tritonVerbose();
    }

    @Override
    public void setTritonVerbose(boolean verbose) {
        e.setTritonVerbose(verbose);
    }

    @Override
    public boolean tritonDumpSections() {
        return e.tritonDumpSections();
    }

    @Override
    public void setTritonDumpSections(boolean dumpSections) {
        e.setTritonDumpSections(dumpSections);
    }

    @Override
    public boolean tritonDumpArgs() {
        return e.tritonDumpArgs();
    }

    @Override
    public void setTritonDumpArgs(boolean dumpArgs) {
        e.setTritonDumpArgs(dumpArgs);
    }

    @Override
    public boolean tritonLogAllPatterns() {
        return e.tritonLogAllPatterns();
    }

    @Override
    public void setTritonLogAllPatterns(boolean logAllPatterns) {
        e.setTritonLogAllPatterns(logAllPatterns);
    }

    @Override
    public boolean tritonAlwaysCompile() {
        return e.tritonAlwaysCompile();
    }

    @Override
    public void setTritonAlwaysCompile(boolean alwaysCompile) {
        e.setTritonAlwaysCompile(alwaysCompile);
    }

    @Override
    public boolean tritonKernelDump() {
        return e.tritonKernelDump();
    }

    @Override
    public void setTritonKernelDump(boolean kernelDump) {
        e.setTritonKernelDump(kernelDump);
    }

    @Override
    public boolean tritonKernelOverride() {
        return e.tritonKernelOverride();
    }

    @Override
    public void setTritonKernelOverride(boolean kernelOverride) {
        e.setTritonKernelOverride(kernelOverride);
    }

    @Override
    public int tritonNumWarps() {
        return e.tritonNumWarps();
    }

    @Override
    public void setTritonNumWarps(int numWarps) {
        e.setTritonNumWarps(numWarps);
    }

    @Override
    public int tritonNumStages() {
        return e.tritonNumStages();
    }

    @Override
    public void setTritonNumStages(int numStages) {
        e.setTritonNumStages(numStages);
    }

    @Override
    public int tritonNumCTAs() {
        return e.tritonNumCTAs();
    }

    @Override
    public void setTritonNumCTAs(int numCTAs) {
        e.setTritonNumCTAs(numCTAs);
    }

    @Override
    public int tritonMaxNreg() {
        return e.tritonMaxNreg();
    }

    @Override
    public void setTritonMaxNreg(int maxNreg) {
        e.setTritonMaxNreg(maxNreg);
    }

    @Override
    public boolean tritonEnableFpFusion() {
        return e.tritonEnableFpFusion();
    }

    @Override
    public void setTritonEnableFpFusion(boolean enableFpFusion) {
        e.setTritonEnableFpFusion(enableFpFusion);
    }

    @Override
    public boolean tritonDisableLineInfo() {
        return e.tritonDisableLineInfo();
    }

    @Override
    public void setTritonDisableLineInfo(boolean disableLineInfo) {
        e.setTritonDisableLineInfo(disableLineInfo);
    }

    @Override
    public String tritonCacheDir() {
        BytePointer p = e.tritonCacheDir();
        return p == null ? "" : p.getString();
    }

    @Override
    public void setTritonCacheDir(String cacheDir) {
        e.setTritonCacheDir(cacheDir);
    }

    @Override
    public String tritonDumpDir() {
        BytePointer p = e.tritonDumpDir();
        return p == null ? "" : p.getString();
    }

    @Override
    public void setTritonDumpDir(String dumpDir) {
        e.setTritonDumpDir(dumpDir);
    }

    @Override
    public String tritonOverrideDir() {
        BytePointer p = e.tritonOverrideDir();
        return p == null ? "" : p.getString();
    }

    @Override
    public void setTritonOverrideDir(String overrideDir) {
        e.setTritonOverrideDir(overrideDir);
    }

    @Override
    public String tritonOverrideArch() {
        BytePointer p = e.tritonOverrideArch();
        return p == null ? "" : p.getString();
    }

    @Override
    public void setTritonOverrideArch(String overrideArch) {
        e.setTritonOverrideArch(overrideArch);
    }

    // Triton + CUDA graph integration
    @Override
    public boolean tritonAllowFallbackCapture() {
        return e.tritonAllowFallbackCapture();
    }

    @Override
    public void setTritonAllowFallbackCapture(boolean allow) {
        e.setTritonAllowFallbackCapture(allow);
    }

    @Override
    public boolean tritonGraphCapture() {
        return e.tritonGraphCapture();
    }

    @Override
    public void setTritonGraphCapture(boolean enable) {
        e.setTritonGraphCapture(enable);
    }

    @Override
    public boolean tritonDumpGraphDot() {
        return e.tritonDumpGraphDot();
    }

    @Override
    public void setTritonDumpGraphDot(boolean dump) {
        e.setTritonDumpGraphDot(dump);
    }

    // Triton debugging flags
    @Override
    public boolean tritonSkipKernels() {
        return e.tritonSkipKernels();
    }

    @Override
    public void setTritonSkipKernels(boolean skip) {
        e.setTritonSkipKernels(skip);
    }

    @Override
    public boolean tritonVerifyKernels() {
        return e.tritonVerifyKernels();
    }

    @Override
    public void setTritonVerifyKernels(boolean verify) {
        e.setTritonVerifyKernels(verify);
    }

    @Override
    public boolean tritonVerifyKeepNative() {
        return e.tritonVerifyKeepNative();
    }

    @Override
    public void setTritonVerifyKeepNative(boolean v) {
        e.setTritonVerifyKeepNative(v);
    }

    @Override
    public int tritonMaxSubKernelIndex() {
        return e.tritonMaxSubKernelIndex();
    }

    @Override
    public void setTritonMaxSubKernelIndex(int idx) {
        e.setTritonMaxSubKernelIndex(idx);
    }

    @Override
    public boolean tritonVerifyFullSnapshot() {
        return e.tritonVerifyFullSnapshot();
    }

    @Override
    public void setTritonVerifyFullSnapshot(boolean v) {
        e.setTritonVerifyFullSnapshot(v);
    }

    @Override
    public boolean tritonForceRecapture() {
        return e.tritonForceRecapture();
    }

    @Override
    public void setTritonForceRecapture(boolean v) {
        e.setTritonForceRecapture(v);
    }

    @Override
    public int tritonCaptureMinExec() {
        return e.tritonCaptureMinExec();
    }

    @Override
    public void setTritonCaptureMinExec(int v) {
        e.setTritonCaptureMinExec(v);
    }

    // Triton compilation scope
    @Override
    public boolean tritonCompileAll() {
        return e.tritonCompileAll();
    }

    @Override
    public void setTritonCompileAll(boolean v) {
        e.setTritonCompileAll(v);
    }

    @Override
    public String tritonExcludeOps() {
        BytePointer p = e.tritonExcludeOps();
        return p == null ? "" : p.getString();
    }

    @Override
    public void setTritonExcludeOps(String ops) {
        e.setTritonExcludeOps(ops);
    }

    @Override
    public String tritonIncludeTypes() {
        BytePointer p = e.tritonIncludeTypes();
        return p == null ? "" : p.getString();
    }

    @Override
    public void setTritonIncludeTypes(String types) {
        e.setTritonIncludeTypes(types);
    }

    // DSP batch-zero flags
    @Override
    public boolean dspBatchZero() { return e.dspBatchZero(); }
    @Override
    public void setDspBatchZero(boolean v) { e.setDspBatchZero(v); }
    @Override
    public boolean dspBatchZeroVerbose() { return e.dspBatchZeroVerbose(); }
    @Override
    public void setDspBatchZeroVerbose(boolean v) { e.setDspBatchZeroVerbose(v); }
    @Override
    public boolean dspBatchZeroGapOnly() { return e.dspBatchZeroGapOnly(); }
    @Override
    public void setDspBatchZeroGapOnly(boolean v) { e.setDspBatchZeroGapOnly(v); }
    @Override
    public boolean dspBatchZeroKernel() { return e.dspBatchZeroKernel(); }
    @Override
    public void setDspBatchZeroKernel(boolean v) { e.setDspBatchZeroKernel(v); }

    // DSP batched GEMM
    @Override
    public boolean dspBatchedGemm() { return e.dspBatchedGemm(); }
    @Override
    public void setDspBatchedGemm(boolean v) { e.setDspBatchedGemm(v); }

    // DSP optimization flags
    @Override
    public boolean dspCastElimination() {
        return e.dspCastElimination();
    }

    @Override
    public void setDspCastElimination(boolean enabled) {
        e.setDspCastElimination(enabled);
    }

    @Override
    public boolean dspMatmulSegmentation() {
        return e.dspMatmulSegmentation();
    }

    @Override
    public void setDspMatmulSegmentation(boolean enabled) {
        e.setDspMatmulSegmentation(enabled);
    }

    @Override
    public boolean dspFp16Compute() {
        return e.dspFp16Compute();
    }

    @Override
    public void setDspFp16Compute(boolean enabled) {
        e.setDspFp16Compute(enabled);
    }

    @Override
    public boolean cublasTf32Enabled() {
        return e.cublasTf32Enabled();
    }

    @Override
    public void setCublasTf32Enabled(boolean enabled) {
        e.setCublasTf32Enabled(enabled);
    }

    @Override
    public boolean dspCastSinkMatmul() {
        return e.dspCastSinkMatmul();
    }

    @Override
    public void setDspCastSinkMatmul(boolean enabled) {
        e.setDspCastSinkMatmul(enabled);
    }

    @Override
    public boolean tritonConsolidatedArgTable() {
        return e.tritonConsolidatedArgTable();
    }

    @Override
    public void setTritonConsolidatedArgTable(boolean enabled) {
        e.setTritonConsolidatedArgTable(enabled);
    }

    @Override
    public boolean tritonArgDirtyTracking() {
        return e.tritonArgDirtyTracking();
    }

    @Override
    public void setTritonArgDirtyTracking(boolean enabled) {
        e.setTritonArgDirtyTracking(enabled);
    }

    @Override
    public boolean tritonSectionFusion() {
        return e.tritonSectionFusion();
    }

    @Override
    public void setTritonSectionFusion(boolean enabled) {
        e.setTritonSectionFusion(enabled);
    }
}
