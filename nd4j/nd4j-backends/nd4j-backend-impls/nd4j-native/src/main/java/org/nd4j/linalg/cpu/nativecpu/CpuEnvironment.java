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
}
