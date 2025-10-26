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
}
