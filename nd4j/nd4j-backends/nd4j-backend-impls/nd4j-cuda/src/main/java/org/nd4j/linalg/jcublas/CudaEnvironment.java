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

import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.jcublas.bindings.Nd4jCuda;

/**
 * CUDA backend implementation of {@link Environment}
 *
 * @author Alex Black
 */
public class CudaEnvironment implements Environment {

    private static final CudaEnvironment INSTANCE = new CudaEnvironment(Nd4jCuda.Environment.getInstance());
    protected boolean funcTracePrintJavaOnly = false;
    protected boolean workspaceTrackOpenClose = false;
    protected int numEventsToKeep = -1;

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
        return false;
    }

    @Override
    public void setTruncateLogStrings(boolean truncateLogStrings) {

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
}
