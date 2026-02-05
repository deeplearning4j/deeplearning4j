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

package org.nd4j.linalg.jzluda;

import org.nd4j.linalg.factory.Environment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ZLUDA Environment Configuration
 *
 * Provides environment information for the ZLUDA backend, including
 * device information, memory limits, and runtime configuration.
 * ZLUDA translates CUDA API calls to other GPU backends (AMD HIP, Intel oneAPI).
 */
public class ZludaEnvironment implements Environment {

    private static final Logger log = LoggerFactory.getLogger(ZludaEnvironment.class);

    private static ZludaEnvironment INSTANCE;

    private final JZludaBackend.ZludaTarget target;
    private final String zludaPath;

    // Environment state
    protected boolean funcTracePrintJavaOnly = false;
    protected boolean workspaceTrackOpenClose = false;
    protected int numEventsToKeep = -1;
    protected boolean truncateNDArrayLongStrings = false;
    protected boolean variableTracingEnabled = false;

    // Debug/profiling state
    protected boolean debug = false;
    protected boolean verbose = false;
    protected boolean profiling = false;
    protected boolean leaksDetector = false;
    protected boolean helpersAllowed = true;

    // Thresholds
    protected int tadThreshold = 1;
    protected int elementwiseThreshold = 32768;
    protected int maxThreads = Runtime.getRuntime().availableProcessors();
    protected int maxMasterThreads = Runtime.getRuntime().availableProcessors();

    // Memory and deletion
    protected boolean deletePrimary = true;
    protected boolean deleteSpecial = true;
    protected boolean deleteShapeInfo = true;
    protected boolean enableBlas = true;
    protected boolean logNativeNDArrayCreation = false;
    protected boolean checkOutputChange = false;
    protected boolean checkInputChange = false;
    protected boolean logNDArrayEvents = false;

    // Lifecycle tracking
    protected boolean lifecycleTracking = false;
    protected boolean trackViews = false;
    protected boolean trackDeletions = false;
    protected int stackDepth = 32;
    protected int reportInterval = 300;
    protected long maxDeletionHistory = 10000;
    protected boolean snapshotFiles = false;
    protected boolean trackOperations = false;
    protected boolean ndArrayTracking = false;
    protected boolean dataBufferTracking = false;
    protected boolean tadCacheTracking = false;
    protected boolean shapeCacheTracking = false;
    protected boolean opContextTracking = false;

    private ZludaEnvironment() {
        this.zludaPath = System.getenv("ZLUDA_PATH");
        this.target = detectTarget();

        // Load settings from environment variables
        this.debug = Boolean.parseBoolean(System.getenv("ZLUDA_DEBUG"));
        this.verbose = Boolean.parseBoolean(System.getenv("ZLUDA_VERBOSE"));
        this.profiling = Boolean.parseBoolean(System.getenv("ZLUDA_PROFILING"));

        String tadThresholdEnv = System.getenv("ZLUDA_TAD_THRESHOLD");
        if (tadThresholdEnv != null) {
            try {
                this.tadThreshold = Integer.parseInt(tadThresholdEnv);
            } catch (NumberFormatException e) {
                log.warn("Invalid ZLUDA_TAD_THRESHOLD value: {}", tadThresholdEnv);
            }
        }

        String elementwiseThresholdEnv = System.getenv("ZLUDA_ELEMENTWISE_THRESHOLD");
        if (elementwiseThresholdEnv != null) {
            try {
                this.elementwiseThreshold = Integer.parseInt(elementwiseThresholdEnv);
            } catch (NumberFormatException e) {
                log.warn("Invalid ZLUDA_ELEMENTWISE_THRESHOLD value: {}", elementwiseThresholdEnv);
            }
        }

        String maxThreadsEnv = System.getenv("ZLUDA_MAX_THREADS");
        if (maxThreadsEnv != null) {
            try {
                this.maxThreads = Integer.parseInt(maxThreadsEnv);
                this.maxMasterThreads = this.maxThreads;
            } catch (NumberFormatException e) {
                log.warn("Invalid ZLUDA_MAX_THREADS value: {}", maxThreadsEnv);
            }
        }

        log.info("ZLUDA Environment initialized: target={}, path={}", target, zludaPath);
    }

    public static synchronized ZludaEnvironment getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new ZludaEnvironment();
        }
        return INSTANCE;
    }

    private JZludaBackend.ZludaTarget detectTarget() {
        String targetEnv = System.getenv("ZLUDA_TARGET");
        if (targetEnv != null) {
            if (targetEnv.equalsIgnoreCase("AMD")) {
                return JZludaBackend.ZludaTarget.AMD;
            } else if (targetEnv.equalsIgnoreCase("INTEL")) {
                return JZludaBackend.ZludaTarget.INTEL;
            }
        }
        return JZludaBackend.ZludaTarget.UNKNOWN;
    }

    // ============ Environment Interface Implementation ============

    @Override
    public boolean isEnableBlas() {
        return enableBlas;
    }

    @Override
    public void setEnableBlas(boolean reallyEnable) {
        this.enableBlas = reallyEnable;
    }

    @Override
    public boolean isLogNativeNDArrayCreation() {
        return logNativeNDArrayCreation;
    }

    @Override
    public void setLogNativeNDArrayCreation(boolean logNativeNDArrayCreation) {
        this.logNativeNDArrayCreation = logNativeNDArrayCreation;
    }

    @Override
    public boolean isCheckOutputChange() {
        return checkOutputChange;
    }

    @Override
    public void setCheckOutputChange(boolean reallyCheck) {
        this.checkOutputChange = reallyCheck;
    }

    @Override
    public boolean isCheckInputChange() {
        return checkInputChange;
    }

    @Override
    public void setCheckInputChange(boolean reallyCheck) {
        this.checkInputChange = reallyCheck;
    }

    @Override
    public void setLogNDArrayEvents(boolean logNDArrayEvents) {
        this.logNDArrayEvents = logNDArrayEvents;
    }

    @Override
    public boolean isLogNDArrayEvents() {
        return logNDArrayEvents;
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
        return deleteShapeInfo;
    }

    @Override
    public void setDeleteShapeInfo(boolean reallyDelete) {
        this.deleteShapeInfo = reallyDelete;
    }

    @Override
    public int blasMajorVersion() {
        return 0;  // ZLUDA doesn't have native BLAS version
    }

    @Override
    public int blasMinorVersion() {
        return 0;
    }

    @Override
    public int blasPatchVersion() {
        return 0;
    }

    @Override
    public boolean isVerbose() {
        return verbose;
    }

    @Override
    public void setVerbose(boolean reallyVerbose) {
        this.verbose = reallyVerbose;
    }

    @Override
    public boolean isDebug() {
        return debug;
    }

    @Override
    public boolean isProfiling() {
        return profiling;
    }

    @Override
    public boolean isDetectingLeaks() {
        return leaksDetector;
    }

    @Override
    public boolean isDebugAndVerbose() {
        return debug && verbose;
    }

    @Override
    public void setDebug(boolean reallyDebug) {
        this.debug = reallyDebug;
    }

    @Override
    public void setProfiling(boolean reallyProfile) {
        this.profiling = reallyProfile;
    }

    @Override
    public void setLeaksDetector(boolean reallyDetect) {
        this.leaksDetector = reallyDetect;
    }

    @Override
    public boolean helpersAllowed() {
        return helpersAllowed;
    }

    @Override
    public void allowHelpers(boolean reallyAllow) {
        this.helpersAllowed = reallyAllow;
    }

    @Override
    public int tadThreshold() {
        return tadThreshold;
    }

    @Override
    public void setTadThreshold(int threshold) {
        this.tadThreshold = threshold;
    }

    @Override
    public int elementwiseThreshold() {
        return elementwiseThreshold;
    }

    @Override
    public void setElementwiseThreshold(int threshold) {
        this.elementwiseThreshold = threshold;
    }

    @Override
    public int maxThreads() {
        return maxThreads;
    }

    @Override
    public void setMaxThreads(int max) {
        this.maxThreads = max;
    }

    @Override
    public int maxMasterThreads() {
        return maxMasterThreads;
    }

    @Override
    public void setMaxMasterThreads(int max) {
        this.maxMasterThreads = max;
    }

    @Override
    public void setMaxPrimaryMemory(long maxBytes) {
        // No-op for ZLUDA - memory is managed by underlying GPU driver
    }

    @Override
    public void setMaxSpecialMemory(long maxBytes) {
        // No-op for ZLUDA
    }

    @Override
    public void setMaxDeviceMemory(long maxBytes) {
        // No-op for ZLUDA
    }

    @Override
    public boolean isCPU() {
        return false;  // ZLUDA targets GPUs
    }

    @Override
    public void setGroupLimit(int group, long numBytes) {
        // No-op for ZLUDA
    }

    @Override
    public void setDeviceLimit(int deviceId, long numBytes) {
        // No-op for ZLUDA
    }

    @Override
    public long getGroupLimit(int group) {
        return 0;
    }

    @Override
    public long getDeviceLimit(int deviceId) {
        return 0;
    }

    @Override
    public long getDeviceCounter(int deviceId) {
        return 0;
    }

    @Override
    public boolean isFuncTracePrintDeallocate() {
        return false;
    }

    @Override
    public boolean isFuncTracePrintAllocate() {
        return false;
    }

    @Override
    public void setFuncTraceForDeallocate(boolean reallyTrace) {
        // No-op for ZLUDA
    }

    @Override
    public void setFuncTraceForAllocate(boolean reallyTrace) {
        // No-op for ZLUDA
    }

    @Override
    public boolean isLifecycleTracking() {
        return lifecycleTracking;
    }

    @Override
    public void setLifecycleTracking(boolean enabled) {
        this.lifecycleTracking = enabled;
    }

    @Override
    public boolean isTrackViews() {
        return trackViews;
    }

    @Override
    public void setTrackViews(boolean track) {
        this.trackViews = track;
    }

    @Override
    public boolean isTrackDeletions() {
        return trackDeletions;
    }

    @Override
    public void setTrackDeletions(boolean track) {
        this.trackDeletions = track;
    }

    @Override
    public int getStackDepth() {
        return stackDepth;
    }

    @Override
    public void setStackDepth(int depth) {
        this.stackDepth = depth;
    }

    @Override
    public int getReportInterval() {
        return reportInterval;
    }

    @Override
    public void setReportInterval(int seconds) {
        this.reportInterval = seconds;
    }

    @Override
    public long getMaxDeletionHistory() {
        return maxDeletionHistory;
    }

    @Override
    public void setMaxDeletionHistory(long max) {
        this.maxDeletionHistory = max;
    }

    @Override
    public boolean isSnapshotFiles() {
        return snapshotFiles;
    }

    @Override
    public void setSnapshotFiles(boolean enabled) {
        this.snapshotFiles = enabled;
    }

    @Override
    public boolean isTrackOperations() {
        return trackOperations;
    }

    @Override
    public void setTrackOperations(boolean enabled) {
        this.trackOperations = enabled;
    }

    @Override
    public boolean isNDArrayTracking() {
        return ndArrayTracking;
    }

    @Override
    public void setNDArrayTracking(boolean enabled) {
        this.ndArrayTracking = enabled;
    }

    @Override
    public boolean isDataBufferTracking() {
        return dataBufferTracking;
    }

    @Override
    public void setDataBufferTracking(boolean enabled) {
        this.dataBufferTracking = enabled;
    }

    @Override
    public boolean isTADCacheTracking() {
        return tadCacheTracking;
    }

    @Override
    public void setTADCacheTracking(boolean enabled) {
        this.tadCacheTracking = enabled;
    }

    @Override
    public boolean isShapeCacheTracking() {
        return shapeCacheTracking;
    }

    @Override
    public void setShapeCacheTracking(boolean enabled) {
        this.shapeCacheTracking = enabled;
    }

    @Override
    public boolean isOpContextTracking() {
        return opContextTracking;
    }

    @Override
    public void setOpContextTracking(boolean enabled) {
        this.opContextTracking = enabled;
    }

    @Override
    public boolean isDeletePrimary() {
        return deletePrimary;
    }

    @Override
    public boolean isDeleteSpecial() {
        return deleteSpecial;
    }

    @Override
    public void setDeletePrimary(boolean reallyDelete) {
        this.deletePrimary = reallyDelete;
    }

    @Override
    public void setDeleteSpecial(boolean reallyDelete) {
        this.deleteSpecial = reallyDelete;
    }

    @Override
    public boolean isVariableTracingEnabled() {
        return variableTracingEnabled;
    }

    @Override
    public void setVariableTracingEnabled(boolean enabled) {
        this.variableTracingEnabled = enabled;
    }

    // ============ CUDA-specific methods (ZLUDA emulates CUDA) ============

    @Override
    public int cudaDeviceCount() {
        return 1;  // ZLUDA typically exposes one device
    }

    @Override
    public int cudaCurrentDevice() {
        return 0;
    }

    @Override
    public void setCudaCurrentDevice(int device) {
        // No-op for ZLUDA with single device
    }

    @Override
    public boolean cudaMemoryPinned() {
        return false;
    }

    @Override
    public void setCudaMemoryPinned(boolean pinned) {
        // No-op for ZLUDA
    }

    @Override
    public boolean cudaUseManagedMemory() {
        return false;
    }

    @Override
    public void setCudaUseManagedMemory(boolean managed) {
        // No-op for ZLUDA
    }

    @Override
    public int cudaMemoryPoolSize() {
        return -1;
    }

    @Override
    public void setCudaMemoryPoolSize(int sizeInMB) {
        // No-op for ZLUDA
    }

    @Override
    public boolean cudaForceP2P() {
        return false;
    }

    @Override
    public void setCudaForceP2P(boolean forceP2P) {
        // No-op for ZLUDA
    }

    @Override
    public boolean cudaAllocatorEnabled() {
        return false;
    }

    @Override
    public void setCudaAllocatorEnabled(boolean enabled) {
        // No-op for ZLUDA
    }

    @Override
    public int cudaMaxBlocks() {
        return -1;
    }

    @Override
    public void setCudaMaxBlocks(int blocks) {
        // No-op for ZLUDA
    }

    @Override
    public int cudaMaxThreadsPerBlock() {
        return -1;
    }

    @Override
    public void setCudaMaxThreadsPerBlock(int threads) {
        // No-op for ZLUDA
    }

    @Override
    public boolean cudaAsyncExecution() {
        return false;
    }

    @Override
    public void setCudaAsyncExecution(boolean async) {
        // No-op for ZLUDA
    }

    @Override
    public int cudaStreamLimit() {
        return -1;
    }

    @Override
    public void setCudaStreamLimit(int limit) {
        // No-op for ZLUDA
    }

    @Override
    public boolean cudaUseDeviceHost() {
        return false;
    }

    @Override
    public void setCudaUseDeviceHost(boolean useDeviceHost) {
        // No-op for ZLUDA
    }

    @Override
    public int cudaEventLimit() {
        return -1;
    }

    @Override
    public void setCudaEventLimit(int limit) {
        // No-op for ZLUDA
    }

    @Override
    public int cudaCachingAllocatorLimit() {
        return -1;
    }

    @Override
    public void setCudaCachingAllocatorLimit(int limitInMB) {
        // No-op for ZLUDA
    }

    @Override
    public long cudaPinnedHostLimit() {
        return -1;
    }

    @Override
    public void setCudaPinnedHostLimit(long limitInMB) {
        // No-op for ZLUDA
    }

    @Override
    public boolean cudaUseUnifiedMemory() {
        return false;
    }

    @Override
    public void setCudaUseUnifiedMemory(boolean unified) {
        // No-op for ZLUDA
    }

    @Override
    public int cudaPrefetchSize() {
        return -1;
    }

    @Override
    public void setCudaPrefetchSize(int sizeInMB) {
        // No-op for ZLUDA
    }

    @Override
    public boolean cudaGraphOptimization() {
        return false;
    }

    @Override
    public void setCudaGraphOptimization(boolean enabled) {
        // No-op for ZLUDA
    }

    @Override
    public boolean cudaTensorCoreEnabled() {
        return false;
    }

    @Override
    public void setCudaTensorCoreEnabled(boolean enabled) {
        // No-op for ZLUDA
    }

    @Override
    public int cudaBlockingSync() {
        return -1;
    }

    @Override
    public void setCudaBlockingSync(int mode) {
        // No-op for ZLUDA
    }

    @Override
    public int cudaDeviceSchedule() {
        return -1;
    }

    @Override
    public void setCudaDeviceSchedule(int schedule) {
        // No-op for ZLUDA
    }

    @Override
    public long cudaStackSize() {
        return -1;
    }

    @Override
    public void setCudaStackSize(long size) {
        // No-op for ZLUDA
    }

    @Override
    public long cudaMallocHeapSize() {
        return -1;
    }

    @Override
    public void setCudaMallocHeapSize(long size) {
        // No-op for ZLUDA
    }

    @Override
    public long cudaPrintfFifoSize() {
        return -1;
    }

    @Override
    public void setCudaPrintfFifoSize(long size) {
        // No-op for ZLUDA
    }

    @Override
    public long cudaDevRuntimeSyncDepth() {
        return -1;
    }

    @Override
    public void setCudaDevRuntimeSyncDepth(long depth) {
        // No-op for ZLUDA
    }

    @Override
    public long cudaDevRuntimePendingLaunchCount() {
        return -1;
    }

    @Override
    public void setCudaDevRuntimePendingLaunchCount(long count) {
        // No-op for ZLUDA
    }

    @Override
    public long cudaMaxL2FetchGranularity() {
        return -1;
    }

    @Override
    public void setCudaMaxL2FetchGranularity(long size) {
        // No-op for ZLUDA
    }

    @Override
    public long cudaPersistingL2CacheSize() {
        return -1;
    }

    @Override
    public void setCudaPersistingL2CacheSize(long size) {
        // No-op for ZLUDA
    }

    @Override
    public int setCudaDeviceLimit(int limitType, long value) {
        // No-op for ZLUDA
        return 0;
    }

    // ============ ZLUDA-specific methods ============

    /**
     * Get the ZLUDA target backend
     */
    public JZludaBackend.ZludaTarget getTarget() {
        return target;
    }

    /**
     * Get the ZLUDA installation path
     */
    public String getZludaPath() {
        return zludaPath;
    }

    /**
     * Check if MIOpen is available (for AMD targets)
     */
    public boolean isMIOpenAvailable() {
        if (target != JZludaBackend.ZludaTarget.AMD) {
            return false;
        }
        String rocmPath = System.getenv("ROCM_PATH");
        if (rocmPath == null) {
            rocmPath = "/opt/rocm";
        }
        return new java.io.File(rocmPath, "lib/libMIOpen.so").exists();
    }

    /**
     * Check if oneDNN is available (for Intel targets)
     */
    public boolean isOneDNNAvailable() {
        if (target != JZludaBackend.ZludaTarget.INTEL) {
            return false;
        }
        String oneapiPath = System.getenv("ONEAPI_ROOT");
        if (oneapiPath == null) {
            oneapiPath = "/opt/intel/oneapi";
        }
        return new java.io.File(oneapiPath).exists();
    }

    @Override
    public String toString() {
        return "ZludaEnvironment{" +
                "target=" + target +
                ", zludaPath='" + zludaPath + '\'' +
                ", miopen=" + isMIOpenAvailable() +
                ", onednn=" + isOneDNNAvailable() +
                '}';
    }
}
