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
package org.nd4j.linalg.factory;

/**
 * This interface describes environment for ND4J.
 * It's used to control memory, profiling, debugging and other options.
 * It's also used to store backend-specific information, like BLAS version, etc
 * <p>
 *     PLEASE NOTE: This interface is NOT supposed to be used by users directly.
 * </p>
 *
 *
 *
 */
public interface Environment {

    // CUDA limit type definitions
    public static final int
        CUDA_LIMIT_STACK_SIZE = 0,
        CUDA_LIMIT_MALLOC_HEAP_SIZE = 1,
        CUDA_LIMIT_PRINTF_FIFO_SIZE = 2,
        CUDA_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
        CUDA_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
        CUDA_LIMIT_MAX_L2_FETCH_GRANULARITY = 5,
        CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE = 6;

    /**
     * Whether to enable blas library or not.
     * Disabling this should mainly be used for debugging
     * purposes.
     * @return
     */
    boolean isEnableBlas();

    void setEnableBlas(boolean reallyEnable);

    /**
     * Set this to true to
     * trigger logging of native c++ ndarray constructors.
     * Use this to debug behavior of individual ops
     * with confusing pointer issues like outputs not
     * updating due to some views being created.
     * @return
     */
    boolean isLogNativeNDArrayCreation();
    void setLogNativeNDArrayCreation(boolean logNativeNDArrayCreation);

    /**
     * If true exceptions will be thrown when an output is NOT changed
     * during ops that are not in place.
     * Note the overhead here can be significant.
     * Inputs are verified by duplicating the inputs and checking
     * for equality.
     * This defaults to false.
     * @return
     */
    boolean isCheckOutputChange();

    void setCheckOutputChange(boolean reallyCheck);

    /**
     * If true exceptions will be thrown when an input is changed
     * during ops that are not in place.
     * Note the overhead here can be significant.
     * Inputs are verified by duplicating the inputs and checking
     * for equality.
     * This defaults to false.
     * @return
     */
    boolean isCheckInputChange();

    void setCheckInputChange(boolean reallyCheck);

    /**
     * Sets whether to write ndarray log events or not.
     * @param logNDArrayEvents the logNDArrayWrites to set
     */
    void setLogNDArrayEvents(boolean logNDArrayEvents);


    /**
     * Returns whether to add write events
     * to ndarrays. A write event is an event
     * where an array is the output of an operation
     * or a put operation happens.
     * @return
     */
    boolean isLogNDArrayEvents();


    /**
     * This method returns whether to truncate ndarray
     * metadata strings or not when {@link #isLogNDArrayEvents()}
     * is true.
     * @return
     */
    boolean isTruncateNDArrayLogStrings();

    /**
     * This method sets whether to truncate
     * ndarray long strings when {@link #isLogNDArrayEvents()}
     * is true
     * @param truncateLogStrings
     */
    void setTruncateLogStrings(boolean truncateLogStrings);

    /**
     * This is the number of {@link WorkspaceUseMetaData} to keep
     * in the memory manager. The default is -1 (unlimited)
     * @return
     */
    int numWorkspaceEventsToKeep();

    /**
     * This is a java side environment flag
     * that controls whether the memory manager records
     * metadata about workspace usage.
     *
     * Note enabling this should only be for tracking down a quick workspace issue
     * in a very limited setting but should otherwise be turned off.
     * The metadata captured is very intensive including stack
     * traces and timestamps.
     * @return
     */
    boolean isTrackWorkspaceOpenClose();

    void setTrackWorkspaceOpenClose(boolean trackWorkspaceOpenClose);

    /**
     * This is a separate flag from {@link #isFuncTracePrintAllocate()}
     * that only records java stack traces rather than c++.
     * This exists due to the amount of overhead that printing c++ stack traces
     * can cause.
     * @return
     */
    boolean isFuncTracePrintJavaOnly();

    void setFuncTracePrintJavaOnly(boolean reallyTrace);

    /**
     * Whether to delete shape info descriptors or not.
     * This is mainly used to control deallocation of
     * shape info descriptors. Shape info descriptors
     * are heap allocated because they are often reused
     * as keys in ConstantSHapeBuffer.
     * Historically, they used to be deallocated
     * on the stack. Due to "smart" deallocation
     * by the stack allocation it would cause random
     * segfaults depending on how it was used.
     * This flag allows for debugging of that behavior
     * while maintaining control over shape descriptor
     * allocation.
     * @return
     */
    boolean isDeleteShapeInfo();
    void setDeleteShapeInfo(boolean reallyDelete);

    /** BLAS major version number (if applicable) */
    int blasMajorVersion();
    /** BLAS minor version number (if applicable) */
    int blasMinorVersion();
    /** BLAS patch version number (if applicable) */
    int blasPatchVersion();

    /** Returns true if ND4J is set to verbose mode */
    boolean isVerbose();
    /** Set verbose mode */
    void setVerbose(boolean reallyVerbose);
    /** Returns true if ND4J is set to debug mode */
    boolean isDebug();
    /** Returns true if ND4J is set to profiling mode */
    boolean isProfiling();
    /** Returns true if ND4J is set to detecting leaks mode */
    boolean isDetectingLeaks();
    /** Returns true if ND4J is set to debug and verbose mode */
    boolean isDebugAndVerbose();

    /** Set debug mode */
    void setDebug( boolean reallyDebug);
    /** Set profiling mode */
    void setProfiling( boolean reallyProfile);
    /** Set leaks detection mode */
    void setLeaksDetector( boolean reallyDetect);
    /** Returns true if helpers (cuDNN, DNNL/MKLDNN etc) are allowed */
    boolean helpersAllowed();
    /** Set whether helpers (cuDNN, DNNL/MKLDNN etc) are allowed */
    void allowHelpers(boolean reallyAllow);

    /** Returns the TAD (tensor along dimension) threshold for ops */
    int tadThreshold();
    /** Set the TAD (tensor along dimension) threshold for ops */
    void setTadThreshold(int threshold);

    /** Returns the elementwise threshold for ops */
    int elementwiseThreshold();
    /** Set the elementwise threshold for ops */
    void setElementwiseThreshold(int threshold);

    /** Returns the maximum number of threads for C++ op execution (if applicable) */
    int maxThreads();
    /** Set the maximum number of threads for C++ op execution (if applicable) */
    void setMaxThreads(int max);

    /** Returns the maximum number of master threads for C++ op execution (if applicable) */
    int maxMasterThreads();
    /** Set the maximum number of master threads for C++ op execution (if applicable) */
    void setMaxMasterThreads(int max);

    /** Set the maximum primary memory */
    void setMaxPrimaryMemory(long maxBytes);
    /** Set the maximum special memory */
    void setMaxSpecialMemory(long maxBytes);
    /** Set the maximum device memory */
    void setMaxDeviceMemory(long maxBytes);

    /** Return true if the backend is a CPU backend, or false otherwise */
    boolean isCPU();

    /**
     * This method allows to set memory limit for a specific group of devices. I.e. CUDA or CPU
     * @param group
     * @param numBytes
     */
    void setGroupLimit(int group, long numBytes);

    /**
     * This method allows to set memory limit for a specific device. I.e. GPU_0
     * @param deviceId
     * @param numBytes
     */
    void setDeviceLimit(int deviceId, long numBytes);

    /**
     * This method returns current group limit
     * @param group
     * @return
     */
    long getGroupLimit(int group);

    /**
     * This method returns current device limit
     * @param deviceId
     * @return
     */
    long getDeviceLimit(int deviceId);

    /**
     * This method returns current allocated amount for a specific device. I.e. GPU_0
     * @param deviceId
     * @return
     */
    long getDeviceCounter(int deviceId);

    /**
     * This function returns whether functrace deallocate is on or not.
     * This means that stack traces will be printed every time a data buffer deallocation happens.
     * This is used for debugging events like double frees
     * @return
     */
    boolean isFuncTracePrintDeallocate();

    /**
     * This function returns whether functrace allocate is on or not.
     * This means that stack traces will be printed every time a data buffer allocation happens
     * when a delete method is called. This is used for debugging events like double frees
     * tracing where a databuffer was created in the context of where it was deleted.
     * @return
     */
    boolean isFuncTracePrintAllocate();

    /**
     * This method sets whether to print stack traces on deallocate or not
     * See {@link #isFuncTracePrintAllocate()} for more information.

     * @param reallyTrace
     */
    void setFuncTraceForDeallocate(boolean reallyTrace);

    /**
     * This method sets whether to print stack traces on allocate or not
     * See {@link #isFuncTracePrintAllocate()} for more information.
     *
     * @param reallyTrace
     */
    void setFuncTraceForAllocate(boolean reallyTrace);

    /**
     * Returns whether NDArray lifecycle tracking is enabled.
     * When enabled (and libnd4j is compiled with SD_GCC_FUNCTRACE), tracks all NDArray
     * allocations and deallocations with stack traces for leak detection.
     * Automatically generates flamegraphs and leak reports at program exit.
     *
     * @return true if lifecycle tracking is enabled
     */
    boolean isLifecycleTracking();

    /**
     * Enable or disable NDArray lifecycle tracking.
     * Requires libnd4j to be compiled with SD_GCC_FUNCTRACE.
     * When enabled, tracks allocations/deallocations and generates leak reports.
     *
     * @param enabled true to enable tracking
     */
    void setLifecycleTracking(boolean enabled);

    /**
     * Returns whether view wrapper tracking is enabled.
     * Views share the underlying buffer but create new NDArray objects.
     * Tracking views can help identify wrapper leaks.
     *
     * @return true if view tracking is enabled
     */
    boolean isTrackViews();

    /**
     * Enable or disable tracking of view wrappers.
     *
     * @param track true to track views
     */
    void setTrackViews(boolean track);

    /**
     * Returns whether deletion stack trace tracking is enabled.
     * Useful for analyzing where deallocations happen and generating deletion flamegraphs.
     *
     * @return true if deletion tracking is enabled
     */
    boolean isTrackDeletions();

    /**
     * Enable or disable deletion stack trace tracking.
     *
     * @param track true to track deletions
     */
    void setTrackDeletions(boolean track);

    /**
     * Returns the stack trace depth for lifecycle tracking.
     * Deeper stacks provide more context but use more memory.
     *
     * @return stack depth (1-64), default 32
     */
    int getStackDepth();

    /**
     * Set the stack trace depth for allocation/deallocation tracking.
     *
     * @param depth stack depth (1-64)
     */
    void setStackDepth(int depth);

    /**
     * Returns the periodic report interval in seconds.
     * The lifecycle tracker prints statistics to stderr at this interval.
     *
     * @return interval in seconds, default 300 (5 minutes)
     */
    int getReportInterval();

    /**
     * Set the periodic report interval.
     *
     * @param seconds interval in seconds
     */
    void setReportInterval(int seconds);

    /**
     * Returns the maximum number of deletion records to keep in history.
     * Used for generating deletion flamegraphs.
     *
     * @return maximum deletion records, default 10000
     */
    long getMaxDeletionHistory();

    /**
     * Set the maximum deletion history size.
     *
     * @param max maximum records
     */
    void setMaxDeletionHistory(long max);

    /**
     * Returns whether periodic file-based snapshots are enabled.
     * When enabled, lifecycle tracker writes timestamped snapshot files
     * (not just stderr output) at each report interval.
     * Files are named: ndarray_snapshot_pid<PID>_<timestamp>_<sequence>.txt
     *
     * @return true if file snapshots are enabled
     */
    boolean isSnapshotFiles();

    /**
     * Enable or disable periodic file-based snapshots.
     * When enabled, snapshots are written to SD_LIFECYCLE_LOG_DIR (or /tmp if not set).
     *
     * @param enabled true to enable file snapshots
     */
    void setSnapshotFiles(boolean enabled);

    /**
     * Returns whether operation name tracking is enabled.
     * When enabled, tracks which operations (Conv2D, MatMul, etc.) are leaking.
     * Reports show top 20 leaking operations by memory.
     *
     * @return true if operation tracking is enabled
     */
    boolean isTrackOperations();

    /**
     * Enable or disable operation name tracking.
     * Useful for identifying which operations contribute most to leaks.
     *
     * @param enabled true to enable operation tracking
     */
    void setTrackOperations(boolean enabled);

    /**
     * Returns whether NDArray lifecycle tracking is enabled.
     * This controls the NDArrayLifecycleTracker specifically.
     *
     * @return true if NDArray tracking is enabled
     */
    boolean isNDArrayTracking();

    /**
     * Enable or disable NDArray lifecycle tracking.
     *
     * @param enabled true to enable NDArray tracking
     */
    void setNDArrayTracking(boolean enabled);

    /**
     * Returns whether DataBuffer lifecycle tracking is enabled.
     * This controls the DataBufferLifecycleTracker specifically.
     *
     * @return true if DataBuffer tracking is enabled
     */
    boolean isDataBufferTracking();

    /**
     * Enable or disable DataBuffer lifecycle tracking.
     *
     * @param enabled true to enable DataBuffer tracking
     */
    void setDataBufferTracking(boolean enabled);

    /**
     * Returns whether TAD cache lifecycle tracking is enabled.
     * This controls the TADCacheLifecycleTracker specifically.
     *
     * @return true if TAD cache tracking is enabled
     */
    boolean isTADCacheTracking();

    /**
     * Enable or disable TAD cache lifecycle tracking.
     *
     * @param enabled true to enable TAD cache tracking
     */
    void setTADCacheTracking(boolean enabled);

    /**
     * Returns whether shape cache lifecycle tracking is enabled.
     * This controls the ShapeCacheLifecycleTracker specifically.
     *
     * @return true if shape cache tracking is enabled
     */
    boolean isShapeCacheTracking();

    /**
     * Enable or disable shape cache lifecycle tracking.
     *
     * @param enabled true to enable shape cache tracking
     */
    void setShapeCacheTracking(boolean enabled);

    /**
     * Returns whether OpContext lifecycle tracking is enabled.
     * This controls the OpContextLifecycleTracker specifically.
     *
     * @return true if OpContext tracking is enabled
     */
    boolean isOpContextTracking();

    /**
     * Enable or disable OpContext lifecycle tracking.
     *
     * @param enabled true to enable OpContext tracking
     */
    void setOpContextTracking(boolean enabled);

    /**
     * This method returns whether to delete cpu side (host side in gpu terms)
     */
    boolean isDeletePrimary();


    /**
     * This method returns whether to delete special (device side in gpu terms)
     * @return
     */
    boolean isDeleteSpecial();

    /**
     * This method sets whether to deleted cpu side (host side in gpu terms)
     * databuffers. Disabling this should be for debugging double frees only.
     * @param reallyDelete
     */
    void setDeletePrimary(boolean reallyDelete);


    /**
     * This method sets whether to deleted special (device side in gpu terms)
     * databuffers. Disabling this should be for debugging double frees only.
     * @param reallyDelete
     */
    void setDeleteSpecial(boolean reallyDelete);

    /**
     * Returns whether variable origin tracing is enabled for debugging import issues.
     * When enabled, operations will trace variable resolution attempts to help debug
     * "unknown array" issues during ONNX graph import.
     * This defaults to false and should only be enabled for debugging purposes.
     * @return true if variable tracing is enabled
     */
    boolean isVariableTracingEnabled();

    /**
     * Set whether to enable variable origin tracing for debugging import issues.
     * When enabled, operations will trace variable resolution attempts which helps
     * debug "unknown array" issues during ONNX graph import by showing exactly
     * where variables come from and why they might be missing.
     * @param enabled true to enable tracing
     */
    void setVariableTracingEnabled(boolean enabled);
    
    // CUDA specific methods
    
    /** Returns the number of CUDA devices available */
    int cudaDeviceCount();
    
    /** Returns the current CUDA device */
    int cudaCurrentDevice();
    
    /** Set the current CUDA device */
    void setCudaCurrentDevice(int device);
    
    /** Returns whether pinned memory is used */
    boolean cudaMemoryPinned();
    
    /** Set whether to use pinned memory */
    void setCudaMemoryPinned(boolean pinned);
    
    /** Returns whether managed memory is used */
    boolean cudaUseManagedMemory();
    
    /** Set whether to use managed memory */
    void setCudaUseManagedMemory(boolean managed);
    
    /** Returns the memory pool size in MB */
    int cudaMemoryPoolSize();
    
    /** Set the memory pool size in MB */
    void setCudaMemoryPoolSize(int sizeInMB);
    
    /** Returns whether P2P memory access is forced */
    boolean cudaForceP2P();
    
    /** Set whether to force P2P memory access */
    void setCudaForceP2P(boolean forceP2P);
    
    /** Returns whether the CUDA allocator is enabled */
    boolean cudaAllocatorEnabled();
    
    /** Set whether the CUDA allocator is enabled */
    void setCudaAllocatorEnabled(boolean enabled);
    
    /** Returns the maximum number of CUDA blocks */
    int cudaMaxBlocks();
    
    /** Set the maximum number of CUDA blocks */
    void setCudaMaxBlocks(int blocks);
    
    /** Returns the maximum threads per block */
    int cudaMaxThreadsPerBlock();
    
    /** Set the maximum threads per block */
    void setCudaMaxThreadsPerBlock(int threads);
    
    /** Returns whether async execution is enabled */
    boolean cudaAsyncExecution();
    
    /** Set whether to use async execution */
    void setCudaAsyncExecution(boolean async);
    
    /** Returns the stream limit */
    int cudaStreamLimit();
    
    /** Set the stream limit */
    void setCudaStreamLimit(int limit);
    
    /** Returns whether device host is used */
    boolean cudaUseDeviceHost();
    
    /** Set whether to use device host */
    void setCudaUseDeviceHost(boolean useDeviceHost);
    
    /** Returns the event limit */
    int cudaEventLimit();
    
    /** Set the event limit */
    void setCudaEventLimit(int limit);
    
    /** Returns the caching allocator limit in MB */
    int cudaCachingAllocatorLimit();

    /** Set the caching allocator limit in MB */
    void setCudaCachingAllocatorLimit(int limitInMB);

    /** Returns the pinned host memory limit in MB (default 8192 = 8 GB) */
    long cudaPinnedHostLimit();

    /** Set the pinned host memory limit in MB. Controls the maximum host memory
     *  used as fallback when GPU memory is exhausted. */
    void setCudaPinnedHostLimit(long limitInMB);
    
    /** Returns whether unified memory is used */
    boolean cudaUseUnifiedMemory();
    
    /** Set whether to use unified memory */
    void setCudaUseUnifiedMemory(boolean unified);
    
    /** Returns the prefetch size in MB */
    int cudaPrefetchSize();
    
    /** Set the prefetch size in MB */
    void setCudaPrefetchSize(int sizeInMB);
    
    /** Returns whether graph optimization is enabled */
    boolean cudaGraphOptimization();
    
    /** Set whether to use graph optimization */
    void setCudaGraphOptimization(boolean enabled);
    
    /** Returns whether tensor core is enabled */
    boolean cudaTensorCoreEnabled();
    
    /** Set whether to use tensor core */
    void setCudaTensorCoreEnabled(boolean enabled);
    
    /** Returns the blocking sync mode */
    int cudaBlockingSync();
    
    /** Set the blocking sync mode */
    void setCudaBlockingSync(int mode);
    
    /** Returns the device schedule mode */
    int cudaDeviceSchedule();
    
    /** Set the device schedule mode */
    void setCudaDeviceSchedule(int schedule);
    
    /** Returns the stack size */
    long cudaStackSize();
    
    /** Set the stack size */
    void setCudaStackSize(long size);
    
    /** Returns the malloc heap size */
    long cudaMallocHeapSize();
    
    /** Set the malloc heap size */
    void setCudaMallocHeapSize(long size);
    
    /** Returns the printf fifo size */
    long cudaPrintfFifoSize();
    
    /** Set the printf fifo size */
    void setCudaPrintfFifoSize(long size);
    
    /** Returns the device runtime sync depth */
    long cudaDevRuntimeSyncDepth();
    
    /** Set the device runtime sync depth */
    void setCudaDevRuntimeSyncDepth(long depth);
    
    /** Returns the device runtime pending launch count */
    long cudaDevRuntimePendingLaunchCount();
    
    /** Set the device runtime pending launch count */
    void setCudaDevRuntimePendingLaunchCount(long count);
    
    /** Returns the maximum L2 fetch granularity */
    long cudaMaxL2FetchGranularity();
    
    /** Set the maximum L2 fetch granularity */
    void setCudaMaxL2FetchGranularity(long size);
    
    /** Returns the persisting L2 cache size */
    long cudaPersistingL2CacheSize();
    
    /** Set the persisting L2 cache size */
    void setCudaPersistingL2CacheSize(long size);
    
    /**
     * Sets a CUDA device limit
     * @param limitType the limit type (use CUDA_LIMIT_* constants)
     * @param value the limit value
     * @return status code (0 for success, non-zero for failure)
     */
    int setCudaDeviceLimit(int limitType, long value);
    
    // ============== Triton GPU Settings ==============
    
    /**
     * Returns the number of parallel threads for Triton kernel compilation.
     * Higher values compile more sub-kernels concurrently but use more memory (~1-2GB per thread).
     * @return number of parallel compilation threads (default: 1)
     */
    default int tritonBuildThreads() {
        return 1;
    }
    
    /**
     * Set the number of parallel threads for Triton kernel compilation.
     * @param threads number of parallel compilation threads (1-16)
     */
    default void setTritonBuildThreads(int threads) {}
    
    /**
     * Returns whether Triton disk cache is enabled for compiled kernels.
     * @return true if cache enabled, false otherwise
     */
    default boolean tritonCacheEnabled() {
        return true;
    }
    
    /**
     * Set whether Triton disk cache is enabled.
     * @param enabled true to enable cache, false to disable
     */
    default void setTritonCacheEnabled(boolean enabled) {}

    /**
     * Returns the cooperative-launch target grid size (in blocks) for Triton sectioned kernels.
     * Value {@code 0} means auto (device-dependent default).
     * @return cooperative launch target blocks
     */
    /**
     * Whether Triton cooperative launch (cuLaunchCooperativeKernel) is enabled.
     * Default is false — sections run without grid-wide sync barriers,
     * allowing arbitrary grid sizes.
     * @return true if cooperative launch is enabled
     */
    default boolean tritonCooperativeLaunch() {
        return false;
    }

    /**
     * Enable or disable Triton cooperative launch.
     * @param enabled true to enable cooperative launch (requires grid fits in SM capacity)
     */
    default void setTritonCooperativeLaunch(boolean enabled) {}

    default int tritonCoopTargetBlocks() {
        return 0;
    }

    /**
     * Set the cooperative-launch target grid size (in blocks) for Triton sectioned kernels.
     * Use {@code 0} for auto.
     * @param blocks target number of cooperative blocks (0 = auto)
     */
    default void setTritonCoopTargetBlocks(int blocks) {}

    /**
     * Returns the adaptive Triton sub-segment op cap. {@code 0} means auto/adaptive.
     */
    default int tritonMaxSubsegmentOps() {
        return 0;
    }

    /**
     * Set adaptive Triton sub-segment op cap. Use {@code 0} for auto/adaptive behavior.
     */
    default void setTritonMaxSubsegmentOps(int ops) {}

    /**
     * Returns the adaptive Triton sub-segment section cap. {@code 0} means auto/adaptive.
     */
    default int tritonMaxSubsegmentSections() {
        return 0;
    }

    /**
     * Set adaptive Triton sub-segment section cap. Use {@code 0} for auto/adaptive behavior.
     */
    default void setTritonMaxSubsegmentSections(int sections) {}

    /**
     * Returns whether verbose Triton diagnostics are enabled.
     */
    default boolean tritonVerbose() {
        return false;
    }

    /**
     * Enable or disable verbose Triton diagnostics.
     */
    default void setTritonVerbose(boolean verbose) {}

    /**
     * Returns whether section diagnostics dump is enabled.
     */
    default boolean tritonDumpSections() {
        return false;
    }

    /**
     * Enable or disable section diagnostics dump.
     */
    default void setTritonDumpSections(boolean dumpSections) {}

    /**
     * Returns whether argument mapping diagnostics dump is enabled.
     */
    default boolean tritonDumpArgs() {
        return false;
    }

    /**
     * Enable or disable argument mapping diagnostics dump.
     */
    default void setTritonDumpArgs(boolean dumpArgs) {}

    /**
     * Returns whether all Triton pattern matches are logged.
     */
    default boolean tritonLogAllPatterns() {
        return false;
    }

    /**
     * Enable or disable full Triton pattern logging.
     */
    default void setTritonLogAllPatterns(boolean logAllPatterns) {}

    /**
     * Returns whether Triton always recompiles kernels (bypassing disk-cache lookup).
     */
    default boolean tritonAlwaysCompile() {
        return false;
    }

    /**
     * Enable or disable always-compile mode for Triton.
     */
    default void setTritonAlwaysCompile(boolean alwaysCompile) {}

    /**
     * Returns whether Triton kernel artifact dumping is enabled.
     */
    default boolean tritonKernelDump() {
        return false;
    }

    /**
     * Enable or disable Triton kernel artifact dumping.
     */
    default void setTritonKernelDump(boolean kernelDump) {}

    /**
     * Returns whether Triton kernel override loading is enabled.
     */
    default boolean tritonKernelOverride() {
        return false;
    }

    /**
     * Enable or disable Triton kernel override loading.
     */
    default void setTritonKernelOverride(boolean kernelOverride) {}

    /**
     * Returns explicit Triton warp override (0 means auto).
     */
    default int tritonNumWarps() {
        return 0;
    }

    /**
     * Set explicit Triton warp override (0 means auto).
     */
    default void setTritonNumWarps(int numWarps) {}

    /**
     * Returns explicit Triton pipeline stage override (0 means auto).
     */
    default int tritonNumStages() {
        return 0;
    }

    /**
     * Set explicit Triton pipeline stage override (0 means auto).
     */
    default void setTritonNumStages(int numStages) {}

    /**
     * Returns Triton cluster CTA override used during TTIR->TTGIR conversion.
     */
    default int tritonNumCTAs() {
        return 1;
    }

    /**
     * Set Triton cluster CTA override used during TTIR->TTGIR conversion.
     */
    default void setTritonNumCTAs(int numCTAs) {}

    /**
     * Returns Triton max register override (0 means unset).
     */
    default int tritonMaxNreg() {
        return 0;
    }

    /**
     * Set Triton max register override (0 means unset).
     */
    default void setTritonMaxNreg(int maxNreg) {}

    /**
     * Returns explicit attention blockN override (0 means auto).
     * Controls the K-loop tile size in fused attention kernels.
     */
    default int tritonAttentionBlockN() {
        return 0;
    }

    /**
     * Set explicit attention blockN override (0 means auto).
     * Use 64 for better occupancy or 128 for better bandwidth efficiency.
     */
    default void setTritonAttentionBlockN(int blockN) {}

    /**
     * Returns whether LLVM floating-point fusion is enabled for Triton compilation.
     */
    default boolean tritonEnableFpFusion() {
        return true;
    }

    /**
     * Enable or disable LLVM floating-point fusion for Triton compilation.
     */
    default void setTritonEnableFpFusion(boolean enableFpFusion) {}

    /**
     * Returns whether line-info generation is disabled for CUDA JIT module load.
     */
    default boolean tritonDisableLineInfo() {
        return false;
    }

    /**
     * Enable or disable line-info generation for CUDA JIT module load.
     */
    default void setTritonDisableLineInfo(boolean disableLineInfo) {}

    /**
     * Returns Triton disk cache directory override, or empty string if unset.
     */
    default String tritonCacheDir() {
        return "";
    }

    /**
     * Set Triton disk cache directory override.
     */
    default void setTritonCacheDir(String cacheDir) {}

    /**
     * Returns Triton artifact dump directory override, or empty string if unset.
     */
    default String tritonDumpDir() {
        return "";
    }

    /**
     * Set Triton artifact dump directory override.
     */
    default void setTritonDumpDir(String dumpDir) {}

    /**
     * Returns Triton kernel override directory, or empty string if unset.
     */
    default String tritonOverrideDir() {
        return "";
    }

    /**
     * Set Triton kernel override directory.
     */
    default void setTritonOverrideDir(String overrideDir) {}

    /**
     * Returns Triton target architecture override, or empty string if unset.
     */
    default String tritonOverrideArch() {
        return "";
    }

    /**
     * Set Triton target architecture override.
     */
    default void setTritonOverrideArch(String overrideArch) {}

    // ============== Triton + CUDA Graph Integration ==============

    /**
     * Whether fallback executor (cuBLAS/native ops) is allowed during CUDA graph capture.
     * When true, segments with fallback ranges can be captured as a single CUDA graph
     * containing both Triton kernels and native cuBLAS/attention calls.
     * This is the key to minimizing kernel launch overhead (like pytorch.compile).
     * @return true if fallback during capture is allowed (default: true)
     */
    default boolean tritonAllowFallbackCapture() {
        return true;
    }

    /**
     * Set whether fallback executor is allowed during CUDA graph capture.
     */
    default void setTritonAllowFallbackCapture(boolean allow) {}

    /**
     * Whether CUDA graph capture of Triton execution is enabled.
     * When disabled, Triton kernels execute directly each step (no capture/replay).
     * @return true if graph capture is enabled (default: true)
     */
    default boolean tritonGraphCapture() {
        return true;
    }

    /**
     * Set whether CUDA graph capture of Triton execution is enabled.
     */
    default void setTritonGraphCapture(boolean enable) {}

    /**
     * Whether to dump captured Triton graph to DOT file for debugging.
     * @return true if DOT dump is enabled (default: false)
     */
    default boolean tritonDumpGraphDot() {
        return false;
    }

    /**
     * Set whether to dump captured Triton graph to DOT file.
     */
    default void setTritonDumpGraphDot(boolean dump) {}

    /**
     * Whether Triton sub-kernels should be skipped and native fallback used instead.
     * For debugging Triton accuracy issues.
     * @return true if Triton kernels are skipped (default: false)
     */
    default boolean tritonSkipKernels() {
        return false;
    }

    default void setTritonSkipKernels(boolean skip) {}

    /**
     * Whether Triton sub-kernel outputs should be verified against native execution.
     * Runs both Triton and native, compares outputs, logs mismatches.
     * @return true if verification is enabled (default: false)
     */
    default boolean tritonVerifyKernels() {
        return false;
    }

    default void setTritonVerifyKernels(boolean verify) {}

    /**
     * When true and tritonVerifyKernels is also true, keep native outputs for
     * continued execution instead of Triton outputs. Tests error accumulation.
     */
    default boolean tritonVerifyKeepNative() {
        return false;
    }

    default void setTritonVerifyKeepNative(boolean v) {}

    /**
     * Max sub-kernel index to run via Triton (-1 = unlimited).
     * Sub-kernels with index > max run native fallback instead.
     * Used for binary search to find which sub-kernel causes corruption.
     */
    default int tritonMaxSubKernelIndex() {
        return -1;
    }

    default void setTritonMaxSubKernelIndex(int idx) {}

    /**
     * When true and tritonVerifyKernels is also true, save/restore ALL outputSlots
     * (not just the sub-kernel's outputs) to detect memory corruption by Triton kernels.
     * Also diffs all slots after each sub-kernel to identify exactly which slot was corrupted.
     */
    default boolean tritonVerifyFullSnapshot() {
        return false;
    }

    default void setTritonVerifyFullSnapshot(boolean v) {}

    /**
     * Force CUDA graph re-capture every step (diagnostic).
     * When enabled, invalidate the cached graph after each replay/capture
     * so every step does a fresh capture instead of replaying.
     */
    default boolean tritonForceRecapture() { return false; }
    default void setTritonForceRecapture(boolean v) {}

    /**
     * Minimum execution count before CUDA graph capture begins.
     * Default=2 (capture on 3rd Triton execution). Set to 9999 to effectively disable capture.
     */
    default int tritonCaptureMinExec() { return 2; }
    default void setTritonCaptureMinExec(int v) {}

    /**
     * Whether Triton compiles ALL section types (not just elementwise).
     * When true, ops not in the exclusion list are compiled through Triton IR.
     * When false (default), only ELEMENTWISE/IDENTITY sections are compiled.
     * @return true if compile-all mode is enabled (default: false)
     */
    default boolean tritonCompileAll() {
        return false;
    }

    default void setTritonCompileAll(boolean v) {}

    /**
     * Comma-separated list of nd4j op names to EXCLUDE from Triton compilation.
     * These ops fall back to cuBLAS/native execution even when tritonCompileAll=true.
     * Example: "matmul,mmul,tensormmul" keeps GEMMs on cuBLAS.
     * @return the exclusion list (default: empty string = no exclusions)
     */
    default String tritonExcludeOps() {
        return "";
    }

    default void setTritonExcludeOps(String ops) {}

    // ============== Triton Segment Fusion Flags (Temporary Testing) ==============

    /**
     * Fuse identity reshape/expand_dims/squeeze into element-wise sections.
     * Reduces section count by ~200-300 for typical decoder models.
     * Controlled by ND4J_TRITON_FUSE_IDENTITY_SHAPES env var.
     * @return true if enabled (default: true)
     */
    default boolean tritonFuseIdentityShapes() { return true; }
    default void setTritonFuseIdentityShapes(boolean v) {}

    /**
     * Fuse consecutive cast ops into single cast kernel.
     * Reduces section count by ~50-100 when cast chains present.
     * Controlled by ND4J_TRITON_FUSE_CAST_CHAINS env var.
     * @return true if enabled (default: true)
     */
    default boolean tritonFuseCastChains() { return true; }
    default void setTritonFuseCastChains(boolean v) {}

    /**
     * Fuse trivial (identity/offset) gather with following element-wise ops.
     * Reduces section count by ~100-200 for seq=1 decode patterns.
     * Controlled by ND4J_TRITON_FUSE_TRIVIAL_GATHER env var.
     * @return true if enabled (default: true)
     */
    default boolean tritonFuseTrivialGather() { return true; }
    default void setTritonFuseTrivialGather(boolean v) {}

    /**
     * Treat identity permute patterns as no-op for seq=1 decode.
     * Reduces section count by ~60-100 for typical attention patterns.
     * Controlled by ND4J_TRITON_SPECIALIZE_PERMUTE_SEQ1 env var.
     * @return true if enabled (default: true)
     */
    default boolean tritonSpecializePermuteSeq1() { return true; }
    default void setTritonSpecializePermuteSeq1(boolean v) {}

    /**
     * Eliminate concat→split or split→concat pairs (canceling patterns).
     * Reduces section count by ~50-100 when patterns detected.
     * Controlled by ND4J_TRITON_ELIMINATE_CONCAT_SPLIT_PAIRS env var.
     * @return true if enabled (default: true)
     */
    default boolean tritonEliminateConcatSplitPairs() { return true; }
    default void setTritonEliminateConcatSplitPairs(boolean v) {}

    /**
     * Fuse matmul→bias→activation patterns (HIGH RISK - accuracy issues possible).
     * Disabled by default. Enable only for testing.
     * Controlled by ND4J_TRITON_FUSED_MATMUL env var.
     * @return true if enabled (default: false)
     */
    default boolean tritonFusedMatmul() { return false; }
    default void setTritonFusedMatmul(boolean v) {}

    /**
     * Fuse GATHER/CONCAT/STACK ops around attention neighborhoods.
     * Reduces section fragmentation around flash attention by merging
     * attention-adjacent data movement ops into larger Triton ranges.
     * Controlled by ND4J_TRITON_FUSE_ATTENTION_NEIGHBORHOODS env var.
     * @return true if enabled (default: true)
     */
    default boolean tritonFuseAttentionNeighborhoods() { return true; }
    default void setTritonFuseAttentionNeighborhoods(boolean v) {}

    // ============== Triton Include Types ==============

    /**
     * When tritonCompileAll=true, only compile these section types (plus ELEMENTWISE/IDENTITY).
     * Comma-separated: "CONST_GEN,SHAPE_MANIP,GATHER,CONCAT,SPLIT,STACK,REDUCTION,ATTENTION,MATMUL"
     * Empty = compile ALL types (original compileAll behavior).
     */
    default String tritonIncludeTypes() {
        return "";
    }

    default void setTritonIncludeTypes(String types) {}

    // ============== DSP Batch-Zero Flags ==============

    /**
     * Replace per-slot memsets with a single batch-zero kernel during CUDA graph capture.
     * Reduces graph node count by ~800. Controlled by ND4J_DSP_BATCH_ZERO env var.
     * @return true if batch-zero is enabled (default: false)
     */
    default boolean dspBatchZero() { return false; }
    default void setDspBatchZero(boolean v) {}

    /**
     * Log every buffer collected for batch-zero (very verbose).
     * Controlled by ND4J_DSP_BATCH_ZERO_VERBOSE env var.
     */
    default boolean dspBatchZeroVerbose() { return false; }
    default void setDspBatchZeroVerbose(boolean v) {}

    /**
     * When true (default), only zero gap (native fallback) slot outputs.
     * When false, zero ALL slot outputs including Triton sub-kernel outputs.
     * Controlled by ND4J_DSP_BATCH_ZERO_GAP_ONLY env var.
     */
    default boolean dspBatchZeroGapOnly() { return true; }
    default void setDspBatchZeroGapOnly(boolean v) {}

    /**
     * When true, use a single CUDA kernel to zero all buffers instead of N cudaMemsetAsync calls.
     * Reduces graph nodes by ~797 but may have memory ordering differences.
     * Controlled by ND4J_DSP_BATCH_ZERO_KERNEL env var.
     */
    default boolean dspBatchZeroKernel() { return false; }
    default void setDspBatchZeroKernel(boolean v) {}

    // ============== DSP Batched GEMM ==============

    /**
     * Group consecutive same-shape matmul slots into single cublasGemmBatchedEx calls.
     * Reduces CUDA graph node count by ~96 nodes for typical transformer models.
     * Controlled by ND4J_DSP_BATCHED_GEMM env var.
     * @return true if batched GEMM grouping is enabled (default: false)
     */
    default boolean dspBatchedGemm() { return false; }
    default void setDspBatchedGemm(boolean v) {}

    // ============== DSP Optimization Flags ==============

    /**
     * Whether cast elimination pass is enabled in FusionPass.
     * Removes redundant cast pairs (e.g., FP16->FP32 followed by FP32->FP16).
     * @return true if cast elimination is enabled (default: true)
     */
    default boolean dspCastElimination() {
        return true;
    }

    default void setDspCastElimination(boolean enabled) {}

    /**
     * Whether matmul segmentation is enabled in frozen-shapes rebuild.
     * Breaks mega-segments at matmul boundaries so element-wise chains between
     * matmuls get separate Triton fusion.
     * @return true if matmul segmentation is enabled (default: false)
     */
    default boolean dspMatmulSegmentation() {
        return false;
    }

    default void setDspMatmulSegmentation(boolean enabled) {}

    /**
     * Whether FP16 compute is enabled for matmuls.
     * Auto-casts FP32 matmul inputs to FP16 for TensorCore GEMM with FP32 accumulation.
     * @return true if FP16 compute is enabled (default: false)
     */
    default boolean dspFp16Compute() {
        return false;
    }

    default void setDspFp16Compute(boolean enabled) {}

    /**
     * Whether TF32 math mode is enabled for cuBLAS on sm_80+ (Ampere+).
     * Uses tensor cores with 10-bit mantissa for FP32 GEMMs.
     * @return true if TF32 is enabled (default: false)
     */
    default boolean cublasTf32Enabled() { return false; }
    default void setCublasTf32Enabled(boolean enabled) {}

    /**
     * Whether to pre-allocate an explicit cuBLAS workspace for CUDA graph capture.
     * Prevents cuBLAS from creating MemAlloc/MemFree graph nodes during capture,
     * but may cause algorithm divergence (split-K selection) for some configurations.
     * @return true if capture workspace is enabled (default: true)
     */
    default boolean cublasCaptureWorkspace() { return true; }
    default void setCublasCaptureWorkspace(boolean enabled) {}

    /**
     * Whether cast sinking through matmul is enabled in FusionPass.
     * Marks FP16→FP32 casts as identity ops when their only consumer is a matmul,
     * since MmulHelper already handles mixed-precision internally.
     * @return true if cast sink is enabled (default: false)
     */
    default boolean dspCastSinkMatmul() { return false; }
    default void setDspCastSinkMatmul(boolean enabled) {}

    /**
     * Whether consolidated arg table is enabled for Triton sub-kernels.
     * Replaces per-kernel arg table H2D copies with a single consolidated copy.
     * @return true if enabled (default: false)
     */
    default boolean tritonConsolidatedArgTable() { return false; }
    default void setTritonConsolidatedArgTable(boolean enabled) {}

    /**
     * Whether dirty tracking is enabled for Triton arg table refresh.
     * Skips refreshing arg tables for sub-kernels with only static (constant weight) args.
     * @return true if enabled (default: false)
     */
    default boolean tritonArgDirtyTracking() { return false; }
    default void setTritonArgDirtyTracking(boolean enabled) {}

    /**
     * Whether section fusion is enabled for Triton mega-kernel merging.
     * When enabled, non-elementwise section types (GATHER, CONCAT, CONST_GEN, SPLIT, STACK)
     * can merge with adjacent sections into mega-kernels via buildSectionedModule() DAG analysis.
     * @return true if section fusion is enabled (default: false)
     */
    default boolean tritonSectionFusion() { return false; }
    default void setTritonSectionFusion(boolean enabled) {}

    /**
     * Whether cost-model fusion scoring is enabled for Triton section merging.
     * @return true if enabled (default: true)
     */
    default boolean tritonFusionScoring() { return true; }
    default void setTritonFusionScoring(boolean enabled) {}

    /**
     * Minimum fusion score required to merge two adjacent sections.
     * @return the minimum score (default: 1.0)
     */
    default float tritonFusionMinScore() { return 1.0f; }
    default void setTritonFusionMinScore(float score) {}

    /**
     * Whether symbolic shape ranges are enabled for DSP segment caching.
     * Dynamic dimensions use rank/dtype-only hashing to avoid recompilation.
     * @return true if enabled (default: true)
     */
    default boolean dspSymbolicShapes() { return true; }
    default void setDspSymbolicShapes(boolean enabled) {}

    /**
     * Number of observation steps before classifying dimensions as static or dynamic.
     * @return warmup step count (default: 2)
     */
    default int dspSymbolicShapeWarmup() { return 2; }
    default void setDspSymbolicShapeWarmup(int steps) {}

    /**
     * Whether capture buffer allocations are routed through CudaMemoryPool for cross-segment reuse.
     * @return true if enabled (default: true)
     */
    default boolean dspCapturePoolEnabled() { return true; }
    default void setDspCapturePoolEnabled(boolean enabled) {}

    /**
     * Maximum bytes for capture buffer pool allocations.
     * @return max bytes (default: 1GB)
     */
    default long dspCapturePoolMaxBytes() { return 1073741824L; }
    default void setDspCapturePoolMaxBytes(long bytes) {}

    // ============== LLM Benchmark Config Presets ==============

    /**
     * Apply the optimal configuration for LLM workloads (inference, training,
     * embedding, and other GPU-accelerated workflows).
     *
     * Sets Triton compilation flags, cuBLAS TF32, DSP batch optimizations,
     * section fusion, graph capture, and consolidated arg table.
     * Matches BenchmarkConfig.optimal() from samediff-llm (~86 tok/s on RTX 4090).
     *
     * This is the recommended default for any LLM execution.
     *
     * @see org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig#optimal()
     */
    default void applyOptimalLLMConfig() {
        // Reset all Triton/DSP flags to defaults
        setTritonGraphCapture(false);
        setTritonSectionFusion(false);
        setTritonConsolidatedArgTable(false);
        setTritonArgDirtyTracking(false);
        setTritonSkipKernels(false);
        setTritonVerifyKernels(false);
        setTritonVerifyFullSnapshot(false);
        setTritonForceRecapture(false);
        setTritonIncludeTypes("");
        setTritonCompileAll(false);
        setTritonExcludeOps("");
        setTritonCooperativeLaunch(false);
        setTritonVerbose(false);
        setTritonDumpSections(false);
        setTritonDumpArgs(false);
        setTritonDumpGraphDot(false);
        setTritonAllowFallbackCapture(false);
        setDspBatchZero(false);
        setDspBatchZeroKernel(false);
        setDspBatchedGemm(false);
        setDspCastSinkMatmul(false);

        // Apply BALANCED profile baseline
        setTritonCacheEnabled(true);
        setTritonAlwaysCompile(false);
        setTritonDisableLineInfo(true);
        setTritonBuildThreads(4);
        setTritonNumWarps(8);
        setTritonNumStages(2);
        setTritonNumCTAs(1);
        setTritonEnableFpFusion(true);

        // Apply optimal overrides (matches BenchmarkConfig.optimal())
        setTritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION");
        setTritonSectionFusion(true);
        setTritonCompileAll(true);
        setTritonGraphCapture(true);
        setTritonAllowFallbackCapture(true);
        setTritonConsolidatedArgTable(true);
        setTritonArgDirtyTracking(true);
        setTritonFusionScoring(false);
        setTritonNumWarps(4);
        setTritonNumStages(1);
        setCublasTf32Enabled(true);
        setDspBatchedGemm(true);
    }

    /**
     * Apply a conservative configuration suitable for LLM inference on
     * systems where Triton/CUDA graph capture may not be available.
     * Enables cuBLAS TF32 and basic DSP optimizations only.
     */
    default void applyBasicLLMConfig() {
        setCublasTf32Enabled(true);
        setDspBatchedGemm(true);
        setDspCastElimination(true);
    }

    /**
     * Apply a debug-friendly LLM configuration with verbose logging
     * and kernel verification enabled.
     */
    default void applyDebugLLMConfig() {
        applyOptimalLLMConfig();
        setTritonVerbose(true);
        setTritonDumpSections(true);
        setTritonVerifyKernels(true);
        setTritonBuildThreads(1);
        setTritonNumWarps(2);
        setTritonNumStages(2);
    }
}
