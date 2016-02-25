/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2015 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

package jcuda.runtime;

import jcuda.*;


/**
 * Java bindings for the NVidia CUDA runtime API.<br />
 * <br />
 * Most comments are extracted from the CUDA online documentation
 */
public class JCuda
{
    /**
     * CUDA runtime version
     */
    public static final int CUDART_VERSION = 7050;

    /**
     * Default page-locked allocation flag
     */
    public static final int cudaHostAllocDefault           = 0x00;

    /**
     * Pinned memory accessible by all CUDA contexts
     */
    public static final int cudaHostAllocPortable          = 0x01;

    /**
     * Map allocation into device space
     */
    public static final int cudaHostAllocMapped            = 0x02;

    /**
     * Write-combined memory
     */
    public static final int cudaHostAllocWriteCombined     = 0x04;


    /**
     * Default host memory registration flag
     */
    public static final int cudaHostRegisterDefault        = 0x00;

    /**
     * Pinned memory accessible by all CUDA contexts
     */
    public static final int cudaHostRegisterPortable       = 0x01;

    /**
     * Map registered memory into device space
     */
    public static final int cudaHostRegisterMapped         = 0x02;

    /** 
     * Memory-mapped I/O space 
     */
    public static final int cudaHostRegisterIoMemory       = 0x04;

    
    /**
     * Default peer addressing enable flag
     */
    public static final int cudaPeerAccessDefault          = 0x00;


    /**
     * Default stream flag
     */
    public static final int cudaStreamDefault              = 0x00;

    /**
     * Stream does not synchronize with stream 0 (the NULL stream)
     */
    public static final int cudaStreamNonBlocking          = 0x01;


    /**
     * Default event flag
     */
    public static final int cudaEventDefault               = 0x00;

    /**
     * Event uses blocking synchronization
     */
    public static final int cudaEventBlockingSync          = 0x01;

    /**
     * Event will not record timing data
     */
    public static final int cudaEventDisableTiming         = 0x02;

    /**
     * Event is suitable for interprocess use. cudaEventDisableTiming must be set
     */
    public static final int cudaEventInterprocess          = 0x04;


    /**
     * Device flag - Automatic scheduling
     */
    public static final int cudaDeviceScheduleAuto         = 0x00;

    /**
     * Device flag - Spin default scheduling
     */
    public static final int cudaDeviceScheduleSpin         = 0x01;

    /**
     * Device flag - Yield default scheduling
     */
    public static final int cudaDeviceScheduleYield        = 0x02;

    /**
     * Device flag - Use blocking synchronization
     */
    public static final int cudaDeviceScheduleBlockingSync = 0x04;

    /**
     * Device flag - Use blocking synchronization
     * @deprecated As of CUDA 4.0 and replaced by cudaDeviceScheduleBlockingSync
     */
    public static final int cudaDeviceBlockingSync         = 0x04;

    /**
     * Device schedule flags mask
     */
    public static final int cudaDeviceScheduleMask         = 0x07;

    /**
     * Device flag - Support mapped pinned allocations
     */
    public static final int cudaDeviceMapHost              = 0x08;

    /**
     * Device flag - Keep local memory allocation after launch
     */
    public static final int cudaDeviceLmemResizeToMax      = 0x10;

    /**
     * Device flags mask
     */
    public static final int cudaDeviceMask                 = 0x1f;


    /**
     * Default CUDA array allocation flag
     */
    public static final int cudaArrayDefault               = 0x00 ;

    /**
     * Must be set in cudaMalloc3DArray to create a layered CUDA array
     */
    public static final int cudaArrayLayered               = 0x01 ;

    /**
     * Must be set in cudaMallocArray or cudaMalloc3DArray in order
     * to bind surfaces to the CUDA array
     */
    public static final int cudaArraySurfaceLoadStore      = 0x02 ;

    /**
     * Must be set in cudaMalloc3DArray to create a cubemap CUDA array
     */
    public static final int cudaArrayCubemap               = 0x04;

    /**
     * Must be set in cudaMallocArray or cudaMalloc3DArray in order to
     * perform texture gather operations on the CUDA array
     */
    public static final int cudaArrayTextureGather         = 0x08;


    /**
     * Automatically enable peer access between remote devices as needed
     */
    public static final int cudaIpcMemLazyEnablePeerAccess = 0x01;

    /**
     * Stream callback flag - stream does not block on callback execution (default)
     *
     * @deprecated This flag was only present in CUDA 5.0.25 (release candidate)
     * and may be removed (or added again) in future releases
     */
    public static final int cudaStreamCallbackNonblocking  = 0x00;

    /**
     * Stream callback flag - stream blocks on callback execution
     *
     * @deprecated This flag was only present in CUDA 5.0.25 (release candidate)
     * and may be removed (or added again) in future releases
     */
    public static final int cudaStreamCallbackBlocking     = 0x01;

    /**
     * cudaSurfaceType1D
     */
    public static final int cudaSurfaceType1D             = 0x01;

    /**
     * cudaSurfaceType2D
     */
    public static final int cudaSurfaceType2D             = 0x02;

    /**
     * cudaSurfaceType3D
     */
    public static final int cudaSurfaceType3D             = 0x03;

    /**
     * cudaSurfaceTypeCubemap
     */
    public static final int cudaSurfaceTypeCubemap        = 0x0C;

    /**
     * cudaSurfaceType1DLayered
     */
    public static final int cudaSurfaceType1DLayered      = 0xF1;

    /**
     * cudaSurfaceType2DLayered
     */
    public static final int cudaSurfaceType2DLayered      = 0xF2;

    /**
     * cudaSurfaceTypeCubemapLayered
     */
    public static final int cudaSurfaceTypeCubemapLayered = 0xFC;





    /**
     * cudaTextureType1D
     */
    public static final int cudaTextureType1D             = 0x01;

    /**
     * cudaTextureType2D
     */
    public static final int cudaTextureType2D             = 0x02;

    /**
     * cudaTextureType3D
     */
    public static final int cudaTextureType3D             = 0x03;

    /**
     * cudaTextureTypeCubemap
     */
    public static final int cudaTextureTypeCubemap        = 0x0C;

    /**
     * cudaTextureType1DLayered
     */
    public static final int cudaTextureType1DLayered      = 0xF1;

    /**
     * cudaTextureType2DLayered
     */
    public static final int cudaTextureType2DLayered      = 0xF2;

    /**
     * cudaTextureTypeCubemapLayered
     */
    public static final int cudaTextureTypeCubemapLayered = 0xFC;

    /**
     * Memory can be accessed by any stream on any device
     */
    public static final int cudaMemAttachGlobal           = 0x01;

    /**
     * Memory cannot be accessed by any stream on any device
     */
    public static final int cudaMemAttachHost             = 0x02;

    /**
     * Memory can only be accessed by a single stream on the associated device
     */
    public static final int cudaMemAttachSingle           = 0x04;


    /**
     * Default behavior
     */
    public static final int cudaOccupancyDefault                = 0x00;

    /**
     * Assume global caching is enabled and cannot be automatically turned off
     */
    public static final int cudaOccupancyDisableCachingOverride = 0x01;


    /**
     * Private inner class for the constant stream values
     */
    private static class ConstantCudaStream_t extends cudaStream_t
    {
        ConstantCudaStream_t(long value)
        {
            super(value);
        }
    }

    /**
     * Stream handle that can be passed as a cudaStream_t to use an implicit stream
     * with legacy synchronization behavior.
     */
    public static cudaStream_t cudaStreamLegacy = new ConstantCudaStream_t(0x1);

    /**
     * Stream handle that can be passed as a cudaStream_t to use an implicit stream
     * with per-thread synchronization behavior.
     */
    public static cudaStream_t cudaStreamPerThread = new ConstantCudaStream_t(0x2);


    /**
     * The flag that indicates whether the native library has been
     * loaded
     */
    private static boolean initialized = false;

    /**
     * Whether a CudaException should be thrown if a method is about
     * to return a result code that is not cudaError.cudaSuccess
     */
    private static boolean exceptionsEnabled = false;


    /* Private constructor to prevent instantiation */
    private JCuda()
    {
    }

    // Initialize the native library.
    static
    {
        initialize();
    }

    /**
     * Initializes the native library. Note that this method
     * does not have to be called explicitly by the user of
     * the library: The library will automatically be
     * initialized when this class is loaded.
     */
    public static void initialize()
    {
        if (!initialized)
        {
            LibUtils.loadLibrary("JCudaRuntime");
            initialized = true;
        }
    }

    /**
     * Set the specified log level for the JCuda runtime library.<br />
     * <br />
     * Currently supported log levels:
     * <br />
     * LOG_QUIET: Never print anything <br />
     * LOG_ERROR: Print error messages <br />
     * LOG_TRACE: Print a trace of all native function calls <br />
     *
     * @param logLevel The log level to use.
     */
    public static void setLogLevel(LogLevel logLevel)
    {
        setLogLevel(logLevel.ordinal());
    }

    private static native void setLogLevel(int logLevel);


    /**
     * Enables or disables exceptions. By default, the methods of this class
     * only return the cudaError error code from the underlying CUDA function.
     * If exceptions are enabled, a CudaException with a detailed error
     * message will be thrown if a method is about to return a result code
     * that is not cudaError.cudaSuccess
     *
     * @param enabled Whether exceptions are enabled
     */
    public static void setExceptionsEnabled(boolean enabled)
    {
        exceptionsEnabled = enabled;
    }

    /**
     * If the given result is different to cudaError.cudaSuccess and
     * exceptions have been enabled, this method will throw a
     * CudaException with an error message that corresponds to the
     * given result code. Otherwise, the given result is simply
     * returned.
     *
     * @param result The result to check
     * @return The result that was given as the parameter
     * @throws CudaException If exceptions have been enabled and
     * the given result code is not cudaError.cudaSuccess
     */
    private static int checkResult(int result)
    {
        if (exceptionsEnabled && result != cudaError.cudaSuccess)
        {
            throw new CudaException(cudaError.stringFor(result));
        }
        return result;
    }




    /**
     * Returns the number of compute-capable devices.
     *
     * <pre>
     * cudaError_t cudaGetDeviceCount (
     *      int* count )
     * </pre>
     * <div>
     *   <p>Returns the number of compute-capable
     *     devices.  Returns in <tt>*count</tt> the number of devices with
     *     compute capability greater or equal to 1.0 that are available for
     *     execution. If there is no such
     *     device then cudaGetDeviceCount() will
     *     return cudaErrorNoDevice. If no driver can be loaded to determine if
     *     any such devices exist then cudaGetDeviceCount() will return
     *     cudaErrorInsufficientDriver.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param count Returns the number of devices with compute capability greater or equal to 1.0
     *
     * @return cudaSuccess, cudaErrorNoDevice, cudaErrorInsufficientDriver
     *
     * @see JCuda#cudaGetDevice
     * @see JCuda#cudaSetDevice
     * @see JCuda#cudaGetDeviceProperties
     * @see JCuda#cudaChooseDevice
     */
    public static int cudaGetDeviceCount(int count[])
    {
        return checkResult(cudaGetDeviceCountNative(count));
    }
    private static native int cudaGetDeviceCountNative(int count[]);


    /**
     * Set device to be used for GPU executions.
     *
     * <pre>
     * cudaError_t cudaSetDevice (
     *      int  device )
     * </pre>
     * <div>
     *   <p>Set device to be used for GPU executions.
     *     Sets <tt>device</tt> as the current device for the calling host
     *     thread.
     *   </p>
     *   <p>Any device memory subsequently allocated
     *     from this host thread using cudaMalloc(), cudaMallocPitch() or
     *     cudaMallocArray() will be physically resident on <tt>device</tt>. Any
     *     host memory allocated from this host thread using cudaMallocHost() or
     *     cudaHostAlloc() or cudaHostRegister() will have its lifetime associated
     *     with <tt>device</tt>. Any streams or events created from this host
     *     thread will be associated with <tt>device</tt>. Any kernels launched
     *     from this host thread using the &lt;&lt;&lt;&gt;&gt;&gt; operator or
     *     cudaLaunch() will be executed on <tt>device</tt>.
     *   </p>
     *   <p>This call may be made from any host
     *     thread, to any device, and at any time. This function will do no
     *     synchronization with
     *     the previous or new device, and should
     *     be considered a very low overhead call.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param device Device on which the active host thread should execute the device code.
     *
     * @return cudaSuccess, cudaErrorInvalidDevice,
     * cudaErrorDeviceAlreadyInUse
     *
     * @see JCuda#cudaGetDeviceCount
     * @see JCuda#cudaGetDevice
     * @see JCuda#cudaGetDeviceProperties
     * @see JCuda#cudaChooseDevice
     */
    public static int cudaSetDevice(int device)
    {
        return checkResult(cudaSetDeviceNative(device));
    }
    private static native int cudaSetDeviceNative(int device);


    /**
     * Sets flags to be used for device executions.
     *
     * <pre>
     * cudaError_t cudaSetDeviceFlags (
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Sets flags to be used for device
     *     executions.  Records <tt>flags</tt> as the flags to use when
     *     initializing the current device. If no device has been made current to
     *     the calling thread then <tt>flags</tt> will be applied to the
     *     initialization of any device initialized by the calling host thread,
     *     unless that device has had its
     *     initialization flags set explicitly by
     *     this or any host thread.
     *   </p>
     *   <p>If the current device has been set and
     *     that device has already been initialized then this call will fail with
     *     the error cudaErrorSetOnActiveProcess. In this case it is necessary to
     *     reset <tt>device</tt> using cudaDeviceReset() before the device's
     *     initialization flags may be set.
     *   </p>
     *   <p>The two LSBs of the <tt>flags</tt>
     *     parameter can be used to control how the CPU thread interacts with the
     *     OS scheduler when waiting for results from the device.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaDeviceScheduleAuto: The
     *         default value if the <tt>flags</tt> parameter is zero, uses a heuristic
     *         based on the number of active CUDA contexts in the process <tt>C</tt>
     *         and the number of logical processors in the system <tt>P</tt>. If <tt>C</tt> &gt; <tt>P</tt>, then CUDA will yield to other OS threads when
     *         waiting for the device, otherwise CUDA will not yield while waiting
     *         for results
     *         and actively spin on the
     *         processor.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDeviceScheduleSpin: Instruct
     *         CUDA to actively spin when waiting for results from the device. This
     *         can decrease latency when waiting for the
     *         device, but may lower the
     *         performance of CPU threads if they are performing work in parallel with
     *         the CUDA thread.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDeviceScheduleYield:
     *         Instruct CUDA to yield its thread when waiting for results from the
     *         device. This can increase latency when waiting for the
     *         device, but can increase the
     *         performance of CPU threads performing work in parallel with the
     *         device.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDeviceScheduleBlockingSync:
     *         Instruct CUDA to block the CPU thread on a synchronization primitive
     *         when waiting for the device to finish work.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDeviceBlockingSync: Instruct
     *         CUDA to block the CPU thread on a synchronization primitive when
     *         waiting for the device to finish work.
     *       </p>
     *       <p>Deprecated: This flag was
     *         deprecated as of CUDA 4.0 and replaced with
     *         cudaDeviceScheduleBlockingSync.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDeviceMapHost: This flag
     *         must be set in order to allocate pinned host memory that is accessible
     *         to the device. If this flag is not set,
     *         cudaHostGetDevicePointer() will
     *         always return a failure code.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDeviceLmemResizeToMax:
     *         Instruct CUDA to not reduce local memory after resizing local memory
     *         for a kernel. This can prevent thrashing by local memory
     *         allocations when launching many
     *         kernels with high local memory usage at the cost of potentially
     *         increased memory usage.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     * </div>
     *
     * @param flags Parameters for device operation
     *
     * @return cudaSuccess, cudaErrorInvalidDevice,
     * cudaErrorSetOnActiveProcess
     *
     * @see JCuda#cudaGetDeviceCount
     * @see JCuda#cudaGetDevice
     * @see JCuda#cudaGetDeviceProperties
     * @see JCuda#cudaSetDevice
     * @see JCuda#cudaSetValidDevices
     * @see JCuda#cudaChooseDevice
     */
    public static int cudaSetDeviceFlags(int flags)
    {
        return checkResult(cudaSetDeviceFlagsNative(flags));
    }
    private static native int cudaSetDeviceFlagsNative(int flags);

    /**
     * Gets the flags for the current device.
     *
     * Returns in flags the flags for the current device. If there is a
     * current device for the calling thread, and the device has been
     * initialized or flags have been set on that device specifically,
     * the flags for the device are returned. If there is no current
     * device, but flags have been set for the thread with
     * cudaSetDeviceFlags, the thread flags are returned. Finally,
     * if there is no current device and no thread flags, the flags
     * for the first device are returned, which may be the default
     * flags. Compare to the behavior of cudaSetDeviceFlags.
     *
     * Typically, the flags returned should match the behavior that will
     * be seen if the calling thread uses a device after this call,
     * without any change to the flags or current device inbetween
     * by this or another thread. Note that if the device is not
     * initialized, it is possible for another thread to change the
     * flags for the current device before it is initialized.
     * Additionally, when using exclusive mode, if this thread has
     * not requested a specific device, it may use a device other
     * than the first device, contrary to the assumption made by this function.
     *
     * If a context has been created via the driver API and is current
     * to the calling thread, the flags for that context are always returned.
     *
     * Flags returned by this function may specifically include
     * cudaDeviceMapHost even though it is not accepted by
     * cudaSetDeviceFlags because it is implicit in runtime API flags.
     * The reason for this is that the current context may have been
     * created via the driver API in which case the flag is not
     * implicit and may be unset.
     *
     * @param flags Pointer to store the device flags
     * @return cudaSuccess, cudaErrorInvalidDevice
     * @see JCuda#cudaGetDevice
     * @see JCuda#cudaGetDeviceProperties
     * @see JCuda#cudaSetDevice
     * @see JCuda#cudaSetDeviceFlags
     */
    public static int cudaGetDeviceFlags(int flags[])
    {
        return checkResult(cudaGetDeviceFlagsNative(flags));
    }
    private static native int cudaGetDeviceFlagsNative(int flags[]);

    /**
     * Set a list of devices that can be used for CUDA.
     *
     * <pre>
     * cudaError_t cudaSetValidDevices (
     *      int* device_arr,
     *      int  len )
     * </pre>
     * <div>
     *   <p>Set a list of devices that can be used
     *     for CUDA.  Sets a list of devices for CUDA execution in priority order
     *     using <tt>device_arr</tt>. The parameter <tt>len</tt> specifies the
     *     number of elements in the list. CUDA will try devices from the list
     *     sequentially until it finds one that works.
     *     If this function is not called, or if it
     *     is called with a <tt>len</tt> of 0, then CUDA will go back to its
     *     default behavior of trying devices sequentially from a default list
     *     containing all of
     *     the available CUDA devices in the system.
     *     If a specified device ID in the list does not exist, this function will
     *     return cudaErrorInvalidDevice. If <tt>len</tt> is not 0 and <tt>device_arr</tt> is NULL or if <tt>len</tt> exceeds the number of
     *     devices in the system, then cudaErrorInvalidValue is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param device_arr List of devices to try
     * @param len Number of devices in specified list
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice
     *
     * @see JCuda#cudaGetDeviceCount
     * @see JCuda#cudaSetDevice
     * @see JCuda#cudaGetDeviceProperties
     * @see JCuda#cudaSetDeviceFlags
     * @see JCuda#cudaChooseDevice
     */
    public static int cudaSetValidDevices (int device_arr[], int len)
    {
        return checkResult(cudaSetValidDevicesNative(device_arr, len));
    }
    private static native int cudaSetValidDevicesNative(int device_arr[], int len);

    /**
     * Returns which device is currently being used.
     *
     * <pre>
     * cudaError_t cudaGetDevice (
     *      int* device )
     * </pre>
     * <div>
     *   <p>Returns which device is currently being
     *     used.  Returns in <tt>*device</tt> the current device for the calling
     *     host thread.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param device Returns the device on which the active host thread executes the device code.
     *
     * @return cudaSuccess
     *
     * @see JCuda#cudaGetDeviceCount
     * @see JCuda#cudaSetDevice
     * @see JCuda#cudaGetDeviceProperties
     * @see JCuda#cudaChooseDevice
     */
    public static int cudaGetDevice(int device[])
    {
        return checkResult(cudaGetDeviceNative(device));
    }
    private static native int cudaGetDeviceNative(int device[]);


    /**
     * Returns information about the compute-device.
     *
     * <pre>
     * cudaError_t cudaGetDeviceProperties (
     *      cudaDeviceProp* prop,
     *      int  device )
     * </pre>
     * <div>
     *   <p>Returns information about the
     *     compute-device.  Returns in <tt>*prop</tt> the properties of device
     *     <tt>dev</tt>. The cudaDeviceProp structure is defined as:
     *   <pre>
     * struct cudaDeviceProp {
     *         char name[256];
     *         size_t totalGlobalMem;
     *         size_t sharedMemPerBlock;
     *         int regsPerBlock;
     *         int warpSize;
     *         size_t memPitch;
     *         int maxThreadsPerBlock;
     *         int maxThreadsDim[3];
     *         int maxGridSize[3];
     *         int clockRate;
     *         size_t totalConstMem;
     *         int major;
     *         int minor;
     *         size_t textureAlignment;
     *         size_t texturePitchAlignment;
     *         int deviceOverlap;
     *         int multiProcessorCount;
     *         int kernelExecTimeoutEnabled;
     *         int integrated;
     *         int canMapHostMemory;
     *         int computeMode;
     *         int maxTexture1D;
     *         int maxTexture1DMipmap;
     *         int maxTexture1DLinear;
     *         int maxTexture2D[2];
     *         int maxTexture2DMipmap[2];
     *         int maxTexture2DLinear[3];
     *         int maxTexture2DGather[2];
     *         int maxTexture3D[3];
     *         int maxTextureCubemap;
     *         int maxTexture1DLayered[2];
     *         int maxTexture2DLayered[3];
     *         int maxTextureCubemapLayered[2];
     *         int maxSurface1D;
     *         int maxSurface2D[2];
     *         int maxSurface3D[3];
     *         int maxSurface1DLayered[2];
     *         int maxSurface2DLayered[3];
     *         int maxSurfaceCubemap;
     *         int maxSurfaceCubemapLayered[2];
     *         size_t surfaceAlignment;
     *         int concurrentKernels;
     *         int ECCEnabled;
     *         int pciBusID;
     *         int pciDeviceID;
     *         int pciDomainID;
     *         int tccDriver;
     *         int asyncEngineCount;
     *         int unifiedAddressing;
     *         int memoryClockRate;
     *         int memoryBusWidth;
     *         int l2CacheSize;
     *         int maxThreadsPerMultiProcessor;
     *     }</pre>
     *   where:
     *   <ul>
     *     <li>
     *       <p>name[256] is an ASCII string
     *         identifying the device;
     *       </p>
     *     </li>
     *     <li>
     *       <p>totalGlobalMem is the total
     *         amount of global memory available on the device in bytes;
     *       </p>
     *     </li>
     *     <li>
     *       <p>sharedMemPerBlock is the
     *         maximum amount of shared memory available to a thread block in bytes;
     *         this amount is shared by all thread blocks simultaneously
     *         resident on a multiprocessor;
     *       </p>
     *     </li>
     *     <li>
     *       <p>regsPerBlock is the maximum
     *         number of 32-bit registers available to a thread block; this number is
     *         shared by all thread blocks simultaneously
     *         resident on a multiprocessor;
     *       </p>
     *     </li>
     *     <li>
     *       <p>warpSize is the warp size in
     *         threads;
     *       </p>
     *     </li>
     *     <li>
     *       <p>memPitch is the maximum pitch
     *         in bytes allowed by the memory copy functions that involve memory
     *         regions allocated through cudaMallocPitch();
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxThreadsPerBlock is the
     *         maximum number of threads per block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxThreadsDim[3] contains the
     *         maximum size of each dimension of a block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxGridSize[3] contains the
     *         maximum size of each dimension of a grid;
     *       </p>
     *     </li>
     *     <li>
     *       <p>clockRate is the clock frequency
     *         in kilohertz;
     *       </p>
     *     </li>
     *     <li>
     *       <p>totalConstMem is the total
     *         amount of constant memory available on the device in bytes;
     *       </p>
     *     </li>
     *     <li>
     *       <p>major, minor are the major and
     *         minor revision numbers defining the device's compute capability;
     *       </p>
     *     </li>
     *     <li>
     *       <p>textureAlignment is the
     *         alignment requirement; texture base addresses that are aligned to
     *         textureAlignment bytes do not need an offset applied to texture
     *         fetches;
     *       </p>
     *     </li>
     *     <li>
     *       <p>texturePitchAlignment is the
     *         pitch alignment requirement for 2D texture references that are bound
     *         to pitched memory;
     *       </p>
     *     </li>
     *     <li>
     *       <p>deviceOverlap is 1 if the
     *         device can concurrently copy memory between host and device while
     *         executing a kernel, or 0 if not. Deprecated,
     *         use instead asyncEngineCount.
     *       </p>
     *     </li>
     *     <li>
     *       <p>multiProcessorCount is the
     *         number of multiprocessors on the device;
     *       </p>
     *     </li>
     *     <li>
     *       <p>kernelExecTimeoutEnabled is 1
     *         if there is a run time limit for kernels executed on the device, or 0
     *         if not.
     *       </p>
     *     </li>
     *     <li>
     *       <p>integrated is 1 if the device
     *         is an integrated (motherboard) GPU and 0 if it is a discrete (card)
     *         component.
     *       </p>
     *     </li>
     *     <li>
     *       <p>canMapHostMemory is 1 if the
     *         device can map host memory into the CUDA address space for use with
     *         cudaHostAlloc()/cudaHostGetDevicePointer(), or 0 if not;
     *       </p>
     *     </li>
     *     <li>
     *       <div>
     *         computeMode is the compute
     *         mode that the device is currently in. Available modes are as follows:
     *         <ul>
     *           <li>
     *             <p>cudaComputeModeDefault:
     *               Default mode - Device is not restricted and multiple threads can use
     *               cudaSetDevice() with this device.
     *             </p>
     *           </li>
     *           <li>
     *             <p>cudaComputeModeExclusive:
     *               Compute-exclusive mode - Only one thread will be able to use
     *               cudaSetDevice() with this device.
     *             </p>
     *           </li>
     *           <li>
     *             <p>cudaComputeModeProhibited:
     *               Compute-prohibited mode - No threads can use cudaSetDevice() with this
     *               device.
     *             </p>
     *           </li>
     *           <li>
     *             <p>cudaComputeModeExclusiveProcess: Compute-exclusive-process mode - Many
     *               threads in one process will be able to use cudaSetDevice() with this
     *               device.
     *             </p>
     *             <p>
     *               If cudaSetDevice() is
     *               called on an already occupied <tt>device</tt> with computeMode
     *               cudaComputeModeExclusive, cudaErrorDeviceAlreadyInUse will be
     *               immediately returned indicating the device cannot be used. When an
     *               occupied exclusive mode device is chosen with
     *               cudaSetDevice, all
     *               subsequent non-device management runtime functions will return
     *               cudaErrorDevicesUnavailable.
     *             </p>
     *           </li>
     *         </ul>
     *       </div>
     *     </li>
     *     <li>
     *       <p>maxTexture1D is the maximum 1D
     *         texture size.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxTexture1DMipmap is the
     *         maximum 1D mipmapped texture texture size.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxTexture1DLinear is the
     *         maximum 1D texture size for textures bound to linear memory.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxTexture2D[2] contains the
     *         maximum 2D texture dimensions.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxTexture2DMipmap[2] contains
     *         the maximum 2D mipmapped texture dimensions.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxTexture2DLinear[3] contains
     *         the maximum 2D texture dimensions for 2D textures bound to pitch linear
     *         memory.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxTexture2DGather[2] contains
     *         the maximum 2D texture dimensions if texture gather operations have to
     *         be performed.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxTexture3D[3] contains the
     *         maximum 3D texture dimensions.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxTextureCubemap is the
     *         maximum cubemap texture width or height.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxTexture1DLayered[2] contains
     *         the maximum 1D layered texture dimensions.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxTexture2DLayered[3] contains
     *         the maximum 2D layered texture dimensions.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxTextureCubemapLayered[2]
     *         contains the maximum cubemap layered texture dimensions.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxSurface1D is the maximum 1D
     *         surface size.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxSurface2D[2] contains the
     *         maximum 2D surface dimensions.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxSurface3D[3] contains the
     *         maximum 3D surface dimensions.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxSurface1DLayered[2] contains
     *         the maximum 1D layered surface dimensions.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxSurface2DLayered[3] contains
     *         the maximum 2D layered surface dimensions.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxSurfaceCubemap is the
     *         maximum cubemap surface width or height.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxSurfaceCubemapLayered[2]
     *         contains the maximum cubemap layered surface dimensions.
     *       </p>
     *     </li>
     *     <li>
     *       <p>surfaceAlignment specifies the
     *         alignment requirements for surfaces.
     *       </p>
     *     </li>
     *     <li>
     *       <p>concurrentKernels is 1 if the
     *         device supports executing multiple kernels within the same context
     *         simultaneously, or 0 if not. It is not guaranteed
     *         that multiple kernels will be
     *         resident on the device concurrently so this feature should not be
     *         relied upon for correctness;
     *       </p>
     *     </li>
     *     <li>
     *       <p>ECCEnabled is 1 if the device
     *         has ECC support turned on, or 0 if not.
     *       </p>
     *     </li>
     *     <li>
     *       <p>pciBusID is the PCI bus
     *         identifier of the device.
     *       </p>
     *     </li>
     *     <li>
     *       <p>pciDeviceID is the PCI device
     *         (sometimes called slot) identifier of the device.
     *       </p>
     *     </li>
     *     <li>
     *       <p>pciDomainID is the PCI domain
     *         identifier of the device.
     *       </p>
     *     </li>
     *     <li>
     *       <p>tccDriver is 1 if the device
     *         is using a TCC driver or 0 if not.
     *       </p>
     *     </li>
     *     <li>
     *       <p>asyncEngineCount is 1 when the
     *         device can concurrently copy memory between host and device while
     *         executing a kernel. It is 2 when the device
     *         can concurrently copy memory
     *         between host and device in both directions and execute a kernel at the
     *         same time. It is 0 if
     *         neither of these is supported.
     *       </p>
     *     </li>
     *     <li>
     *       <p>unifiedAddressing is 1 if the
     *         device shares a unified address space with the host and 0 otherwise.
     *       </p>
     *     </li>
     *     <li>
     *       <p>memoryClockRate is the peak
     *         memory clock frequency in kilohertz.
     *       </p>
     *     </li>
     *     <li>
     *       <p>memoryBusWidth is the memory
     *         bus width in bits.
     *       </p>
     *     </li>
     *     <li>
     *       <p>l2CacheSize is L2 cache size
     *         in bytes.
     *       </p>
     *     </li>
     *     <li>
     *       <p>maxThreadsPerMultiProcessor is
     *         the number of maximum resident threads per multiprocessor.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     * </div>
     *
     * @param prop Properties for the specified device
     * @param device Device number to get properties for
     *
     * @return cudaSuccess, cudaErrorInvalidDevice
     *
     * @see JCuda#cudaGetDeviceCount
     * @see JCuda#cudaGetDevice
     * @see JCuda#cudaSetDevice
     * @see JCuda#cudaChooseDevice
     * @see JCuda#cudaDeviceGetAttribute
     */
    public static int cudaGetDeviceProperties(cudaDeviceProp prop, int device)
    {
        return checkResult(cudaGetDevicePropertiesNative(prop, device));
    }
    private static native int cudaGetDevicePropertiesNative(cudaDeviceProp prop, int device);


    /**
     * Returns information about the device.
     *
     * <pre>
     * cudaError_t cudaDeviceGetAttribute (
     *      int* value,
     *      cudaDeviceAttr attr,
     *      int  device )
     * </pre>
     * <div>
     *   <p>Returns information about the device.
     *     Returns in <tt>*value</tt> the integer value of the attribute <tt>attr</tt> on device <tt>device</tt>. The supported attributes are:
     *   <ul>
     *     <li>
     *       <p>cudaDevAttrMaxThreadsPerBlock:
     *         Maximum number of threads per block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxBlockDimX:
     *         Maximum x-dimension of a block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxBlockDimY:
     *         Maximum y-dimension of a block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxBlockDimZ:
     *         Maximum z-dimension of a block;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxGridDimX: Maximum
     *         x-dimension of a grid;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxGridDimY: Maximum
     *         y-dimension of a grid;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxGridDimZ: Maximum
     *         z-dimension of a grid;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSharedMemoryPerBlock:
     *         Maximum amount of shared memory available to a thread block in bytes;
     *         this amount is shared by all thread blocks simultaneously
     *         resident on a multiprocessor;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrTotalConstantMemory:
     *         Memory available on device for __constant__ variables in a CUDA C
     *         kernel in bytes;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrWarpSize: Warp size
     *         in threads;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxPitch: Maximum
     *         pitch in bytes allowed by the memory copy functions that involve memory
     *         regions allocated through cudaMallocPitch();
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture1DWidth:
     *         Maximum 1D texture width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture1DLinearWidth:
     *         Maximum width for a 1D texture bound to linear memory;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture1DMipmappedWidth:
     *         Maximum mipmapped 1D texture width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture2DWidth:
     *         Maximum 2D texture width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture2DHeight:
     *         Maximum 2D texture height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture2DLinearWidth:
     *         Maximum width for a 2D texture bound to linear memory;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture2DLinearHeight:
     *         Maximum height for a 2D texture bound to linear memory;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture2DLinearPitch:
     *         Maximum pitch in bytes for a 2D texture bound to linear memory;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture2DMipmappedWidth:
     *         Maximum mipmapped 2D texture width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture2DMipmappedHeight:
     *         Maximum mipmapped 2D texture height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture3DWidth:
     *         Maximum 3D texture width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture3DHeight:
     *         Maximum 3D texture height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture3DDepth:
     *         Maximum 3D texture depth;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture3DWidthAlt:
     *         Alternate maximum 3D texture width, 0 if no alternate maximum 3D
     *         texture size is supported;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture3DHeightAlt:
     *         Alternate maximum 3D texture height, 0 if no alternate maximum 3D
     *         texture size is supported;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture3DDepthAlt:
     *         Alternate maximum 3D texture depth, 0 if no alternate maximum 3D
     *         texture size is supported;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTextureCubemapWidth:
     *         Maximum cubemap texture width or height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture1DLayeredWidth:
     *         Maximum 1D layered texture width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture1DLayeredLayers:
     *         Maximum layers in a 1D layered texture;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture2DLayeredWidth:
     *         Maximum 2D layered texture width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture2DLayeredHeight:
     *         Maximum 2D layered texture height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTexture2DLayeredLayers:
     *         Maximum layers in a 2D layered texture;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTextureCubemapLayeredWidth: Maximum cubemap layered
     *         texture width or height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxTextureCubemapLayeredLayers: Maximum layers in a cubemap
     *         layered texture;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurface1DWidth:
     *         Maximum 1D surface width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurface2DWidth:
     *         Maximum 2D surface width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurface2DHeight:
     *         Maximum 2D surface height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurface3DWidth:
     *         Maximum 3D surface width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurface3DHeight:
     *         Maximum 3D surface height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurface3DDepth:
     *         Maximum 3D surface depth;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurface1DLayeredWidth:
     *         Maximum 1D layered surface width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurface1DLayeredLayers:
     *         Maximum layers in a 1D layered surface;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurface2DLayeredWidth:
     *         Maximum 2D layered surface width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurface2DLayeredHeight:
     *         Maximum 2D layered surface height;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurface2DLayeredLayers:
     *         Maximum layers in a 2D layered surface;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurfaceCubemapWidth:
     *         Maximum cubemap surface width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurfaceCubemapLayeredWidth: Maximum cubemap layered
     *         surface width;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxSurfaceCubemapLayeredLayers: Maximum layers in a cubemap
     *         layered surface;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxRegistersPerBlock:
     *         Maximum number of 32-bit registers available to a thread block; this
     *         number is shared by all thread blocks simultaneously
     *         resident on a multiprocessor;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrClockRate: Peak
     *         clock frequency in kilohertz;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrTextureAlignment:
     *         Alignment requirement; texture base addresses aligned to textureAlign
     *         bytes do not need an offset applied to texture fetches;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrTexturePitchAlignment:
     *         Pitch alignment requirement for 2D texture references bound to pitched
     *         memory;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrGpuOverlap: 1 if
     *         the device can concurrently copy memory between host and device while
     *         executing a kernel, or 0 if not;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMultiProcessorCount:
     *         Number of multiprocessors on the device;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrKernelExecTimeout:
     *         1 if there is a run time limit for kernels executed on the device, or
     *         0 if not;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrIntegrated: 1 if
     *         the device is integrated with the memory subsystem, or 0 if not;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrCanMapHostMemory: 1
     *         if the device can map host memory into the CUDA address space, or 0 if
     *         not;
     *       </p>
     *     </li>
     *     <li>
     *       <div>
     *         cudaDevAttrComputeMode:
     *         Compute mode is the compute mode that the device is currently in.
     *         Available modes are as follows:
     *         <ul>
     *           <li>
     *             <p>cudaComputeModeDefault:
     *               Default mode - Device is not restricted and multiple threads can use
     *               cudaSetDevice() with this device.
     *             </p>
     *           </li>
     *           <li>
     *             <p>cudaComputeModeExclusive:
     *               Compute-exclusive mode - Only one thread will be able to use
     *               cudaSetDevice() with this device.
     *             </p>
     *           </li>
     *           <li>
     *             <p>cudaComputeModeProhibited:
     *               Compute-prohibited mode - No threads can use cudaSetDevice() with this
     *               device.
     *             </p>
     *           </li>
     *           <li>
     *             <p>cudaComputeModeExclusiveProcess: Compute-exclusive-process mode - Many
     *               threads in one process will be able to use cudaSetDevice() with this
     *               device.
     *             </p>
     *           </li>
     *         </ul>
     *       </div>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrConcurrentKernels:
     *         1 if the device supports executing multiple kernels within the same
     *         context simultaneously, or 0 if not. It is not guaranteed
     *         that multiple kernels will be
     *         resident on the device concurrently so this feature should not be
     *         relied upon for correctness;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrEccEnabled: 1 if
     *         error correction is enabled on the device, 0 if error correction is
     *         disabled or not supported by the device;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrPciBusId: PCI bus
     *         identifier of the device;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrPciDeviceId: PCI
     *         device (also known as slot) identifier of the device;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrTccDriver: 1 if the
     *         device is using a TCC driver. TCC is only available on Tesla hardware
     *         running Windows Vista or later;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMemoryClockRate:
     *         Peak memory clock frequency in kilohertz;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrGlobalMemoryBusWidth:
     *         Global memory bus width in bits;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrL2CacheSize: Size
     *         of L2 cache in bytes. 0 if the device doesn't have L2 cache;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrMaxThreadsPerMultiProcessor:
     *         Maximum resident threads per multiprocessor;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrUnifiedAddressing:
     *         1 if the device shares a unified address space with the host, or 0 if
     *         not;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrComputeCapabilityMajor:
     *         Major compute capability version number;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaDevAttrComputeCapabilityMinor:
     *         Minor compute capability version number;
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param value Returned device attribute value
     * @param attr Device attribute to query
     * @param device Device number to query
     *
     * @return cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue
     *
     * @see JCuda#cudaGetDeviceCount
     * @see JCuda#cudaGetDevice
     * @see JCuda#cudaSetDevice
     * @see JCuda#cudaChooseDevice
     * @see JCuda#cudaGetDeviceProperties
     */
    public static int cudaDeviceGetAttribute(int value[], int cudaDeviceAttr_attr, int device)
    {
        return checkResult(cudaDeviceGetAttributeNative(value, cudaDeviceAttr_attr, device));
    }
    private static native int cudaDeviceGetAttributeNative(int value[], int cudaDeviceAttr_attr, int device);


    /**
     * Select compute-device which best matches criteria.
     *
     * <pre>
     * cudaError_t cudaChooseDevice (
     *      int* device,
     *      const cudaDeviceProp* prop )
     * </pre>
     * <div>
     *   <p>Select compute-device which best matches
     *     criteria.  Returns in <tt>*device</tt> the device which has properties
     *     that best match <tt>*prop</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param device Device with best match
     * @param prop Desired device properties
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaGetDeviceCount
     * @see JCuda#cudaGetDevice
     * @see JCuda#cudaSetDevice
     * @see JCuda#cudaGetDeviceProperties
     */
    public static int cudaChooseDevice(int device[], cudaDeviceProp prop)
    {
        return checkResult(cudaChooseDeviceNative(device, prop));
    }
    private static native int cudaChooseDeviceNative(int device[], cudaDeviceProp prop);



    /**
     * Allocates logical 1D, 2D, or 3D memory objects on the device.
     *
     * <pre>
     * cudaError_t cudaMalloc3D (
     *      cudaPitchedPtr* pitchedDevPtr,
     *      cudaExtent extent )
     * </pre>
     * <div>
     *   <p>Allocates logical 1D, 2D, or 3D memory
     *     objects on the device.  Allocates at least <tt>width</tt> * <tt>height</tt> * <tt>depth</tt> bytes of linear memory on the device
     *     and returns a cudaPitchedPtr in which <tt>ptr</tt> is a pointer to
     *     the allocated memory. The function may pad the allocation to ensure
     *     hardware alignment requirements are met.
     *     The pitch returned in the <tt>pitch</tt>
     *     field of <tt>pitchedDevPtr</tt> is the width in bytes of the
     *     allocation.
     *   </p>
     *   <p>The returned cudaPitchedPtr contains
     *     additional fields <tt>xsize</tt> and <tt>ysize</tt>, the logical
     *     width and height of the allocation, which are equivalent to the <tt>width</tt> and <tt>height</tt><tt>extent</tt> parameters provided
     *     by the programmer during allocation.
     *   </p>
     *   <p>For allocations of 2D and 3D objects,
     *     it is highly recommended that programmers perform allocations using
     *     cudaMalloc3D() or cudaMallocPitch(). Due to alignment restrictions in
     *     the hardware, this is especially true if the application will be
     *     performing memory copies
     *     involving 2D or 3D objects (whether
     *     linear memory or CUDA arrays).
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pitchedDevPtr Pointer to allocated pitched device memory
     * @param extent Requested allocation size (width field in bytes)
     *
     * @return cudaSuccess, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaFree
     * @see JCuda#cudaMemcpy3D
     * @see JCuda#cudaMemset3D
     * @see JCuda#cudaMalloc3DArray
     * @see JCuda#cudaMallocArray
     * @see JCuda#cudaFreeArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaHostAlloc
     * @see cudaPitchedPtr
     * @see cudaExtent
     */
    public static int cudaMalloc3D(cudaPitchedPtr pitchDevPtr, cudaExtent extent)
    {
        return checkResult(cudaMalloc3DNative(pitchDevPtr, extent));
    }
    private static native int cudaMalloc3DNative(cudaPitchedPtr pitchDevPtr, cudaExtent extent);

    /**
     * Allocate an array on the device.
     *
     * <pre>
     * cudaError_t cudaMalloc3DArray (
     *      cudaArray_t* array,
     *      const cudaChannelFormatDesc* desc,
     *      cudaExtent extent,
     *      unsigned int  flags = 0 )
     * </pre>
     * <div>
     *   <p>Allocate an array on the device.
     *     Allocates a CUDA array according to the cudaChannelFormatDesc structure
     *     <tt>desc</tt> and returns a handle to the new CUDA array in <tt>*array</tt>.
     *   </p>
     *   <p>The cudaChannelFormatDesc is defined
     *     as:
     *   <pre>    struct cudaChannelFormatDesc {
     *         int x, y, z, w;
     *         enum cudaChannelFormatKind
     *                   f;
     *     };</pre>
     *   where cudaChannelFormatKind is one of
     *   cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, or
     *   cudaChannelFormatKindFloat.
     *   </p>
     *   <p>cudaMalloc3DArray() can allocate the
     *     following:
     *   </p>
     *   <ul>
     *     <li>
     *       <p>A 1D array is allocated if the
     *         height and depth extents are both zero.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 2D array is allocated if only
     *         the depth extent is zero.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 3D array is allocated if all
     *         three extents are non-zero.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 1D layered CUDA array is
     *         allocated if only the height extent is zero and the cudaArrayLayered
     *         flag is set. Each layer is
     *         a 1D array. The number of layers
     *         is determined by the depth extent.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 2D layered CUDA array is
     *         allocated if all three extents are non-zero and the cudaArrayLayered
     *         flag is set. Each layer is
     *         a 2D array. The number of layers
     *         is determined by the depth extent.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A cubemap CUDA array is
     *         allocated if all three extents are non-zero and the cudaArrayCubemap
     *         flag is set. Width must be equal
     *         to height, and depth must be
     *         six. A cubemap is a special type of 2D layered CUDA array, where the
     *         six layers represent the
     *         six faces of a cube. The order
     *         of the six layers in memory is the same as that listed in
     *         cudaGraphicsCubeFace.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A cubemap layered CUDA array
     *         is allocated if all three extents are non-zero, and both, cudaArrayCubemap
     *         and cudaArrayLayered
     *         flags are set. Width must be
     *         equal to height, and depth must be a multiple of six. A cubemap layered
     *         CUDA array is a special
     *         type of 2D layered CUDA array
     *         that consists of a collection of cubemaps. The first six layers
     *         represent the first cubemap,
     *         the next six layers form the
     *         second cubemap, and so on.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The <tt>flags</tt> parameter enables
     *     different options to be specified that affect the allocation, as
     *     follows.
     *   <ul>
     *     <li>
     *       <p>cudaArrayDefault: This flag's
     *         value is defined to be 0 and provides default array allocation
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArrayLayered: Allocates a
     *         layered CUDA array, with the depth extent indicating the number of
     *         layers
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArrayCubemap: Allocates a
     *         cubemap CUDA array. Width must be equal to height, and depth must be
     *         six. If the cudaArrayLayered flag is also
     *         set, depth must be a multiple
     *         of six.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArraySurfaceLoadStore:
     *         Allocates a CUDA array that could be read from or written to using a
     *         surface reference.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArrayTextureGather: This
     *         flag indicates that texture gather operations will be performed on the
     *         CUDA array. Texture gather can only be performed
     *         on 2D CUDA arrays.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The width, height and depth extents must
     *     meet certain size requirements as listed in the following table. All
     *     values are specified
     *     in elements.
     *   </p>
     *   <p>Note that 2D CUDA arrays have different
     *     size requirements if the cudaArrayTextureGather flag is set. In that
     *     case, the valid range for (width, height, depth) is
     *     ((1,maxTexture2DGather[0]), (1,maxTexture2DGather[1]),
     *     0).
     *   </p>
     *   <div>
     *     <table cellpadding="4" cellspacing="0" summary="" frame="border" border="1" rules="all">
     *       <tbody>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>CUDA array
     *               type</strong>
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>Valid extents
     *               that must always be met
     *               {(width range in
     *               elements), (height range), (depth range)}</strong>
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>Valid extents
     *               with cudaArraySurfaceLoadStore set
     *               {(width range in
     *               elements), (height range), (depth range)}</strong>
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>1D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxTexture1D),
     *               0, 0 }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxSurface1D),
     *               0, 0 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>2D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxTexture2D[0]),
     *               (1,maxTexture2D[1]), 0 }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxSurface2D[0]),
     *               (1,maxSurface2D[1]), 0 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>3D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxTexture3D[0]),
     *               (1,maxTexture3D[1]), (1,maxTexture3D[2]) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxSurface3D[0]),
     *               (1,maxSurface3D[1]), (1,maxSurface3D[2]) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>1D Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxTexture1DLayered[0]), 0, (1,maxTexture1DLayered[1]) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxSurface1DLayered[0]), 0, (1,maxSurface1DLayered[1]) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>2D Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxTexture2DLayered[0]), (1,maxTexture2DLayered[1]),
     *               (1,maxTexture2DLayered[2]) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxSurface2DLayered[0]), (1,maxSurface2DLayered[1]),
     *               (1,maxSurface2DLayered[2]) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>Cubemap </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxTextureCubemap),
     *               (1,maxTextureCubemap), 6 }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxSurfaceCubemap),
     *               (1,maxSurfaceCubemap), 6 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>Cubemap Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxTextureCubemapLayered[0]), (1,maxTextureCubemapLayered[0]),
     *               (1,maxTextureCubemapLayered[1]) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxSurfaceCubemapLayered[0]), (1,maxSurfaceCubemapLayered[0]),
     *               (1,maxSurfaceCubemapLayered[1]) }
     *             </p>
     *           </td>
     *         </tr>
     *       </tbody>
     *     </table>
     *   </div>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param array Pointer to allocated array in device memory
     * @param desc Requested channel format
     * @param extent Requested allocation size (width field in elements)
     * @param flags Flags for extensions
     *
     * @return cudaSuccess, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaFree
     * @see JCuda#cudaFreeArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaHostAlloc
     * @see cudaExtent
     */
    public static int cudaMalloc3DArray(cudaArray arrayPtr, cudaChannelFormatDesc desc, cudaExtent extent)    {
        return cudaMalloc3DArray(arrayPtr, desc, extent, 0);
    }

    /**
     * Allocate an array on the device.
     *
     * <pre>
     * cudaError_t cudaMalloc3DArray (
     *      cudaArray_t* array,
     *      const cudaChannelFormatDesc* desc,
     *      cudaExtent extent,
     *      unsigned int  flags = 0 )
     * </pre>
     * <div>
     *   <p>Allocate an array on the device.
     *     Allocates a CUDA array according to the cudaChannelFormatDesc structure
     *     <tt>desc</tt> and returns a handle to the new CUDA array in <tt>*array</tt>.
     *   </p>
     *   <p>The cudaChannelFormatDesc is defined
     *     as:
     *   <pre>    struct cudaChannelFormatDesc {
     *         int x, y, z, w;
     *         enum cudaChannelFormatKind
     *                   f;
     *     };</pre>
     *   where cudaChannelFormatKind is one of
     *   cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, or
     *   cudaChannelFormatKindFloat.
     *   </p>
     *   <p>cudaMalloc3DArray() can allocate the
     *     following:
     *   </p>
     *   <ul>
     *     <li>
     *       <p>A 1D array is allocated if the
     *         height and depth extents are both zero.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 2D array is allocated if only
     *         the depth extent is zero.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 3D array is allocated if all
     *         three extents are non-zero.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 1D layered CUDA array is
     *         allocated if only the height extent is zero and the cudaArrayLayered
     *         flag is set. Each layer is
     *         a 1D array. The number of layers
     *         is determined by the depth extent.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 2D layered CUDA array is
     *         allocated if all three extents are non-zero and the cudaArrayLayered
     *         flag is set. Each layer is
     *         a 2D array. The number of layers
     *         is determined by the depth extent.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A cubemap CUDA array is
     *         allocated if all three extents are non-zero and the cudaArrayCubemap
     *         flag is set. Width must be equal
     *         to height, and depth must be
     *         six. A cubemap is a special type of 2D layered CUDA array, where the
     *         six layers represent the
     *         six faces of a cube. The order
     *         of the six layers in memory is the same as that listed in
     *         cudaGraphicsCubeFace.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A cubemap layered CUDA array
     *         is allocated if all three extents are non-zero, and both, cudaArrayCubemap
     *         and cudaArrayLayered
     *         flags are set. Width must be
     *         equal to height, and depth must be a multiple of six. A cubemap layered
     *         CUDA array is a special
     *         type of 2D layered CUDA array
     *         that consists of a collection of cubemaps. The first six layers
     *         represent the first cubemap,
     *         the next six layers form the
     *         second cubemap, and so on.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The <tt>flags</tt> parameter enables
     *     different options to be specified that affect the allocation, as
     *     follows.
     *   <ul>
     *     <li>
     *       <p>cudaArrayDefault: This flag's
     *         value is defined to be 0 and provides default array allocation
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArrayLayered: Allocates a
     *         layered CUDA array, with the depth extent indicating the number of
     *         layers
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArrayCubemap: Allocates a
     *         cubemap CUDA array. Width must be equal to height, and depth must be
     *         six. If the cudaArrayLayered flag is also
     *         set, depth must be a multiple
     *         of six.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArraySurfaceLoadStore:
     *         Allocates a CUDA array that could be read from or written to using a
     *         surface reference.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArrayTextureGather: This
     *         flag indicates that texture gather operations will be performed on the
     *         CUDA array. Texture gather can only be performed
     *         on 2D CUDA arrays.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The width, height and depth extents must
     *     meet certain size requirements as listed in the following table. All
     *     values are specified
     *     in elements.
     *   </p>
     *   <p>Note that 2D CUDA arrays have different
     *     size requirements if the cudaArrayTextureGather flag is set. In that
     *     case, the valid range for (width, height, depth) is
     *     ((1,maxTexture2DGather[0]), (1,maxTexture2DGather[1]),
     *     0).
     *   </p>
     *   <div>
     *     <table cellpadding="4" cellspacing="0" summary="" frame="border" border="1" rules="all">
     *       <tbody>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>CUDA array
     *               type</strong>
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>Valid extents
     *               that must always be met
     *               {(width range in
     *               elements), (height range), (depth range)}</strong>
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>Valid extents
     *               with cudaArraySurfaceLoadStore set
     *               {(width range in
     *               elements), (height range), (depth range)}</strong>
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>1D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxTexture1D),
     *               0, 0 }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxSurface1D),
     *               0, 0 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>2D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxTexture2D[0]),
     *               (1,maxTexture2D[1]), 0 }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxSurface2D[0]),
     *               (1,maxSurface2D[1]), 0 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>3D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxTexture3D[0]),
     *               (1,maxTexture3D[1]), (1,maxTexture3D[2]) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxSurface3D[0]),
     *               (1,maxSurface3D[1]), (1,maxSurface3D[2]) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>1D Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxTexture1DLayered[0]), 0, (1,maxTexture1DLayered[1]) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxSurface1DLayered[0]), 0, (1,maxSurface1DLayered[1]) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>2D Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxTexture2DLayered[0]), (1,maxTexture2DLayered[1]),
     *               (1,maxTexture2DLayered[2]) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxSurface2DLayered[0]), (1,maxSurface2DLayered[1]),
     *               (1,maxSurface2DLayered[2]) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>Cubemap </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxTextureCubemap),
     *               (1,maxTextureCubemap), 6 }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxSurfaceCubemap),
     *               (1,maxSurfaceCubemap), 6 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>Cubemap Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxTextureCubemapLayered[0]), (1,maxTextureCubemapLayered[0]),
     *               (1,maxTextureCubemapLayered[1]) }
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxSurfaceCubemapLayered[0]), (1,maxSurfaceCubemapLayered[0]),
     *               (1,maxSurfaceCubemapLayered[1]) }
     *             </p>
     *           </td>
     *         </tr>
     *       </tbody>
     *     </table>
     *   </div>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param array Pointer to allocated array in device memory
     * @param desc Requested channel format
     * @param extent Requested allocation size (width field in elements)
     * @param flags Flags for extensions
     *
     * @return cudaSuccess, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaFree
     * @see JCuda#cudaFreeArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaHostAlloc
     * @see cudaExtent
     */
    public static int cudaMalloc3DArray(cudaArray arrayPtr, cudaChannelFormatDesc desc, cudaExtent extent, int flags)
    {
        return checkResult(cudaMalloc3DArrayNative(arrayPtr, desc, extent, flags));
    }
    private static native int cudaMalloc3DArrayNative(cudaArray arrayPtr, cudaChannelFormatDesc desc, cudaExtent extent, int flags);


    /**
     * Allocate a mipmapped array on the device.
     *
     * <pre>
     * cudaError_t cudaMallocMipmappedArray (
     *      cudaMipmappedArray_t* mipmappedArray,
     *      const cudaChannelFormatDesc* desc,
     *      cudaExtent extent,
     *      unsigned int  numLevels,
     *      unsigned int  flags = 0 )
     * </pre>
     * <div>
     *   <p>Allocate a mipmapped array on the device.
     *     Allocates a CUDA mipmapped array according to the cudaChannelFormatDesc
     *     structure <tt>desc</tt> and returns a handle to the new CUDA mipmapped
     *     array in <tt>*mipmappedArray</tt>. <tt>numLevels</tt> specifies the
     *     number of mipmap levels to be allocated. This value is clamped to the
     *     range [1, 1 + floor(log2(max(width, height,
     *     depth)))].
     *   </p>
     *   <p>The cudaChannelFormatDesc is defined
     *     as:
     *   <pre>    struct cudaChannelFormatDesc {
     *         int x, y, z, w;
     *         enum cudaChannelFormatKind
     *                   f;
     *     };</pre>
     *   where cudaChannelFormatKind is one of
     *   cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, or
     *   cudaChannelFormatKindFloat.
     *   </p>
     *   <p>cudaMallocMipmappedArray() can allocate
     *     the following:
     *   </p>
     *   <ul>
     *     <li>
     *       <p>A 1D mipmapped array is
     *         allocated if the height and depth extents are both zero.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 2D mipmapped array is
     *         allocated if only the depth extent is zero.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 3D mipmapped array is
     *         allocated if all three extents are non-zero.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 1D layered CUDA mipmapped
     *         array is allocated if only the height extent is zero and the
     *         cudaArrayLayered flag is set. Each
     *         layer is a 1D mipmapped array.
     *         The number of layers is determined by the depth extent.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A 2D layered CUDA mipmapped
     *         array is allocated if all three extents are non-zero and the
     *         cudaArrayLayered flag is set. Each
     *         layer is a 2D mipmapped array.
     *         The number of layers is determined by the depth extent.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A cubemap CUDA mipmapped array
     *         is allocated if all three extents are non-zero and the cudaArrayCubemap
     *         flag is set. Width
     *         must be equal to height, and
     *         depth must be six. The order of the six layers in memory is the same
     *         as that listed in cudaGraphicsCubeFace.
     *       </p>
     *     </li>
     *     <li>
     *       <p>A cubemap layered CUDA mipmapped
     *         array is allocated if all three extents are non-zero, and both,
     *         cudaArrayCubemap and cudaArrayLayered
     *         flags are set. Width must be
     *         equal to height, and depth must be a multiple of six. A cubemap layered
     *         CUDA mipmapped array
     *         is a special type of 2D layered
     *         CUDA mipmapped array that consists of a collection of cubemap mipmapped
     *         arrays. The first
     *         six layers represent the first
     *         cubemap mipmapped array, the next six layers form the second cubemap
     *         mipmapped array, and so
     *         on.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The <tt>flags</tt> parameter enables
     *     different options to be specified that affect the allocation, as
     *     follows.
     *   <ul>
     *     <li>
     *       <p>cudaArrayDefault: This flag's
     *         value is defined to be 0 and provides default mipmapped array
     *         allocation
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArrayLayered: Allocates a
     *         layered CUDA mipmapped array, with the depth extent indicating the
     *         number of layers
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArrayCubemap: Allocates a
     *         cubemap CUDA mipmapped array. Width must be equal to height, and depth
     *         must be six. If the cudaArrayLayered
     *         flag is also set, depth must be
     *         a multiple of six.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArraySurfaceLoadStore: This
     *         flag indicates that individual mipmap levels of the CUDA mipmapped
     *         array will be read from or written to using a surface
     *         reference.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArrayTextureGather: This
     *         flag indicates that texture gather operations will be performed on the
     *         CUDA array. Texture gather can only be performed
     *         on 2D CUDA mipmapped arrays,
     *         and the gather operations are performed only on the most detailed
     *         mipmap level.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The width, height and depth extents must
     *     meet certain size requirements as listed in the following table. All
     *     values are specified
     *     in elements.
     *   </p>
     *   <div>
     *     <table cellpadding="4" cellspacing="0" summary="" frame="border" border="1" rules="all">
     *       <tbody>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>CUDA array
     *               type</strong>
     *             </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p><strong>Valid
     *               extents
     *               {(width range in
     *               elements), (height range), (depth range)}</strong>
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>1D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxTexture1DMipmap),
     *               0, 0 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>2D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxTexture2DMipmap[0]), (1,maxTexture2DMipmap[1]), 0 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>3D </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxTexture3D[0]),
     *               (1,maxTexture3D[1]), (1,maxTexture3D[2]) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>1D Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxTexture1DLayered[0]), 0, (1,maxTexture1DLayered[1]) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>2D Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxTexture2DLayered[0]), (1,maxTexture2DLayered[1]),
     *               (1,maxTexture2DLayered[2]) }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>Cubemap </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{ (1,maxTextureCubemap),
     *               (1,maxTextureCubemap), 6 }
     *             </p>
     *           </td>
     *         </tr>
     *         <tr>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>Cubemap Layered </p>
     *           </td>
     *           <td valign="top" rowspan="1" colspan="1">
     *             <p>{
     *               (1,maxTextureCubemapLayered[0]), (1,maxTextureCubemapLayered[0]),
     *               (1,maxTextureCubemapLayered[1]) }
     *             </p>
     *           </td>
     *         </tr>
     *       </tbody>
     *     </table>
     *   </div>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param mipmappedArray Pointer to allocated mipmapped array in device memory
     * @param desc Requested channel format
     * @param extent Requested allocation size (width field in elements)
     * @param numLevels Number of mipmap levels to allocate
     * @param flags Flags for extensions
     *
     * @return cudaSuccess, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaFree
     * @see JCuda#cudaFreeArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaHostAlloc
     * @see cudaExtent
     */
    public static int cudaMallocMipmappedArray(cudaMipmappedArray mipmappedArray, cudaChannelFormatDesc desc, cudaExtent extent, int numLevels, int flags)
    {
        return checkResult(cudaMallocMipmappedArrayNative(mipmappedArray, desc, extent, numLevels, flags));
    }
    private static native int cudaMallocMipmappedArrayNative(cudaMipmappedArray mipmappedArray, cudaChannelFormatDesc desc, cudaExtent extent, int numLevels, int flags);


    /**
     * Gets a mipmap level of a CUDA mipmapped array.
     *
     * <pre>
     * cudaError_t cudaGetMipmappedArrayLevel (
     *      cudaArray_t* levelArray,
     *      cudaMipmappedArray_const_t mipmappedArray,
     *      unsigned int  level )
     * </pre>
     * <div>
     *   <p>Gets a mipmap level of a CUDA mipmapped
     *     array.  Returns in <tt>*levelArray</tt> a CUDA array that represents
     *     a single mipmap level of the CUDA mipmapped array <tt>mipmappedArray</tt>.
     *   </p>
     *   <p>If <tt>level</tt> is greater than the
     *     maximum number of levels in this mipmapped array, cudaErrorInvalidValue
     *     is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param levelArray Returned mipmap level CUDA array
     * @param mipmappedArray CUDA mipmapped array
     * @param level Mipmap level
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaFree
     * @see JCuda#cudaFreeArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaHostAlloc
     * @see cudaExtent
     */
    public static int cudaGetMipmappedArrayLevel(cudaArray levelArray, cudaMipmappedArray mipmappedArray, int level)
    {
        return checkResult(cudaGetMipmappedArrayLevelNative(levelArray, mipmappedArray, level));
    }
    private static native int cudaGetMipmappedArrayLevelNative(cudaArray levelArray, cudaMipmappedArray mipmappedArray, int level);

    /**
     * Initializes or sets device memory to a value.
     *
     * <pre>
     * cudaError_t cudaMemset3D (
     *      cudaPitchedPtr pitchedDevPtr,
     *      int  value,
     *      cudaExtent extent )
     * </pre>
     * <div>
     *   <p>Initializes or sets device memory to a
     *     value.  Initializes each element of a 3D array to the specified value
     *     <tt>value</tt>. The object to initialize is defined by <tt>pitchedDevPtr</tt>. The <tt>pitch</tt> field of <tt>pitchedDevPtr</tt>
     *     is the width in memory in bytes of the 3D array pointed to by <tt>pitchedDevPtr</tt>, including any padding added to the end of each
     *     row. The <tt>xsize</tt> field specifies the logical width of each row
     *     in bytes, while the <tt>ysize</tt> field specifies the height of each
     *     2D slice in rows.
     *   </p>
     *   <p>The extents of the initialized region
     *     are specified as a <tt>width</tt> in bytes, a <tt>height</tt> in
     *     rows, and a <tt>depth</tt> in slices.
     *   </p>
     *   <p>Extents with <tt>width</tt> greater
     *     than or equal to the <tt>xsize</tt> of <tt>pitchedDevPtr</tt> may
     *     perform significantly faster than extents narrower than the <tt>xsize</tt>. Secondarily, extents with <tt>height</tt> equal to the
     *     <tt>ysize</tt> of <tt>pitchedDevPtr</tt> will perform faster than
     *     when the <tt>height</tt> is shorter than the <tt>ysize</tt>.
     *   </p>
     *   <p>This function performs fastest when the
     *     <tt>pitchedDevPtr</tt> has been allocated by cudaMalloc3D().
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host unless <tt>pitchedDevPtr</tt> refers to
     *     pinned host memory.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pitchedDevPtr Pointer to pitched device memory
     * @param value Value to set for each byte of specified memory
     * @param extent Size parameters for where to set device memory (width field in bytes)
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInvalidDevicePointer
     *
     * @see JCuda#cudaMemset
     * @see JCuda#cudaMemset2D
     * @see JCuda#cudaMemsetAsync
     * @see JCuda#cudaMemset2DAsync
     * @see JCuda#cudaMemset3DAsync
     * @see JCuda#cudaMalloc3D
     * @see cudaPitchedPtr
     * @see cudaExtent
     */
    public static int cudaMemset3D(cudaPitchedPtr pitchDevPtr, int value, cudaExtent extent)
    {
        return checkResult(cudaMemset3DNative(pitchDevPtr, value, extent));
    }
    private static native int cudaMemset3DNative(cudaPitchedPtr pitchDevPtr, int value, cudaExtent extent);


    /**
     * Initializes or sets device memory to a value.
     *
     * <pre>
     * cudaError_t cudaMemsetAsync (
     *      void* devPtr,
     *      int  value,
     *      size_t count,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Initializes or sets device memory to a
     *     value.  Fills the first <tt>count</tt> bytes of the memory area
     *     pointed to by <tt>devPtr</tt> with the constant byte value <tt>value</tt>.
     *   </p>
     *   <p>cudaMemsetAsync() is asynchronous with
     *     respect to the host, so the call may return before the memset is
     *     complete. The operation can optionally
     *     be associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>stream</tt> is non-zero,
     *     the operation may overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param devPtr Pointer to device memory
     * @param value Value to set for each byte of specified memory
     * @param count Size in bytes to set
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInvalidDevicePointer
     *
     * @see JCuda#cudaMemset
     * @see JCuda#cudaMemset2D
     * @see JCuda#cudaMemset3D
     * @see JCuda#cudaMemset2DAsync
     * @see JCuda#cudaMemset3DAsync
     */
    public static int cudaMemsetAsync(Pointer devPtr, int value, long count, cudaStream_t stream)
    {
        return checkResult(cudaMemsetAsyncNative(devPtr, value, count, stream));
    }


    private static native int cudaMemsetAsyncNative(Pointer devPtr, int value, long count, cudaStream_t stream);


    /**
     * Initializes or sets device memory to a value.
     *
     * <pre>
     * cudaError_t cudaMemset2DAsync (
     *      void* devPtr,
     *      size_t pitch,
     *      int  value,
     *      size_t width,
     *      size_t height,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Initializes or sets device memory to a
     *     value.  Sets to the specified value <tt>value</tt> a matrix (<tt>height</tt> rows of <tt>width</tt> bytes each) pointed to by <tt>dstPtr</tt>. <tt>pitch</tt> is the width in bytes of the 2D array
     *     pointed to by <tt>dstPtr</tt>, including any padding added to the end
     *     of each row. This function performs fastest when the pitch is one that
     *     has been passed
     *     back by cudaMallocPitch().
     *   </p>
     *   <p>cudaMemset2DAsync() is asynchronous with
     *     respect to the host, so the call may return before the memset is
     *     complete. The operation can optionally
     *     be associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>stream</tt> is non-zero,
     *     the operation may overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param devPtr Pointer to 2D device memory
     * @param pitch Pitch in bytes of 2D device memory
     * @param value Value to set for each byte of specified memory
     * @param width Width of matrix set (columns in bytes)
     * @param height Height of matrix set (rows)
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInvalidDevicePointer
     *
     * @see JCuda#cudaMemset
     * @see JCuda#cudaMemset2D
     * @see JCuda#cudaMemset3D
     * @see JCuda#cudaMemsetAsync
     * @see JCuda#cudaMemset3DAsync
     */
    public static int cudaMemset2DAsync(Pointer devPtr, long pitch, int value, long width, long height, cudaStream_t stream)
    {
        return checkResult(cudaMemset2DAsyncNative(devPtr, pitch, value, width, height, stream));
    }
    private static native int cudaMemset2DAsyncNative(Pointer devPtr, long pitch, int value, long width, long height, cudaStream_t stream);


    /**
     * Initializes or sets device memory to a value.
     *
     * <pre>
     * cudaError_t cudaMemset3DAsync (
     *      cudaPitchedPtr pitchedDevPtr,
     *      int  value,
     *      cudaExtent extent,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Initializes or sets device memory to a
     *     value.  Initializes each element of a 3D array to the specified value
     *     <tt>value</tt>. The object to initialize is defined by <tt>pitchedDevPtr</tt>. The <tt>pitch</tt> field of <tt>pitchedDevPtr</tt>
     *     is the width in memory in bytes of the 3D array pointed to by <tt>pitchedDevPtr</tt>, including any padding added to the end of each
     *     row. The <tt>xsize</tt> field specifies the logical width of each row
     *     in bytes, while the <tt>ysize</tt> field specifies the height of each
     *     2D slice in rows.
     *   </p>
     *   <p>The extents of the initialized region
     *     are specified as a <tt>width</tt> in bytes, a <tt>height</tt> in
     *     rows, and a <tt>depth</tt> in slices.
     *   </p>
     *   <p>Extents with <tt>width</tt> greater
     *     than or equal to the <tt>xsize</tt> of <tt>pitchedDevPtr</tt> may
     *     perform significantly faster than extents narrower than the <tt>xsize</tt>. Secondarily, extents with <tt>height</tt> equal to the
     *     <tt>ysize</tt> of <tt>pitchedDevPtr</tt> will perform faster than
     *     when the <tt>height</tt> is shorter than the <tt>ysize</tt>.
     *   </p>
     *   <p>This function performs fastest when the
     *     <tt>pitchedDevPtr</tt> has been allocated by cudaMalloc3D().
     *   </p>
     *   <p>cudaMemset3DAsync() is asynchronous with
     *     respect to the host, so the call may return before the memset is
     *     complete. The operation can optionally
     *     be associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>stream</tt> is non-zero,
     *     the operation may overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pitchedDevPtr Pointer to pitched device memory
     * @param value Value to set for each byte of specified memory
     * @param extent Size parameters for where to set device memory (width field in bytes)
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInvalidDevicePointer
     *
     * @see JCuda#cudaMemset
     * @see JCuda#cudaMemset2D
     * @see JCuda#cudaMemset3D
     * @see JCuda#cudaMemsetAsync
     * @see JCuda#cudaMemset2DAsync
     * @see JCuda#cudaMalloc3D
     * @see cudaPitchedPtr
     * @see cudaExtent
     */
    public static int cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream)
    {
        return checkResult(cudaMemset3DAsyncNative(pitchedDevPtr, value, extent, stream));
    }

    private static native int cudaMemset3DAsyncNative(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream);

    /**
     * Copies data between 3D objects.
     *
     * <pre>
     * cudaError_t cudaMemcpy3D (
     *      const cudaMemcpy3DParms* p )
     * </pre>
     * <div>
     *   <p>Copies data between 3D objects.
     *   <pre>struct cudaExtent {
     *   size_t width;
     *   size_t height;
     *   size_t depth;
     * };
     * struct cudaExtent
     *                   make_cudaExtent(size_t w, size_t h, size_t d);
     *
     * struct cudaPos {
     *   size_t x;
     *   size_t y;
     *   size_t z;
     * };
     * struct cudaPos
     *                   make_cudaPos(size_t x, size_t y, size_t z);
     *
     * struct cudaMemcpy3DParms {
     *   cudaArray_t
     *                   srcArray;
     *   struct cudaPos
     *                   srcPos;
     *   struct cudaPitchedPtr
     *                   srcPtr;
     *   cudaArray_t
     *                   dstArray;
     *   struct cudaPos
     *                   dstPos;
     *   struct cudaPitchedPtr
     *                   dstPtr;
     *   struct cudaExtent
     *                   extent;
     *   enum cudaMemcpyKind
     *                   kind;
     * };</pre>
     *   </p>
     *   <p>cudaMemcpy3D() copies data betwen two
     *     3D objects. The source and destination objects may be in either host
     *     memory, device memory, or a CUDA
     *     array. The source, destination, extent,
     *     and kind of copy performed is specified by the cudaMemcpy3DParms struct
     *     which should be initialized to zero before use:
     *   <pre>cudaMemcpy3DParms
     * myParms = {0};</pre>
     *   </p>
     *   <p>The struct passed to cudaMemcpy3D() must
     *     specify one of <tt>srcArray</tt> or <tt>srcPtr</tt> and one of <tt>dstArray</tt> or <tt>dstPtr</tt>. Passing more than one non-zero
     *     source or destination will cause cudaMemcpy3D() to return an error.
     *   </p>
     *   <p>The <tt>srcPos</tt> and <tt>dstPos</tt>
     *     fields are optional offsets into the source and destination objects
     *     and are defined in units of each object's elements. The
     *     element for a host or device pointer is
     *     assumed to be <strong>unsigned char</strong>. For CUDA arrays,
     *     positions must be in the range [0, 2048) for any dimension.
     *   </p>
     *   <p>The <tt>extent</tt> field defines the
     *     dimensions of the transferred area in elements. If a CUDA array is
     *     participating in the copy, the extent
     *     is defined in terms of that array's
     *     elements. If no CUDA array is participating in the copy then the
     *     extents are defined in
     *     elements of <strong>unsigned
     *     char</strong>.
     *   </p>
     *   <p>The <tt>kind</tt> field defines the
     *     direction of the copy. It must be one of cudaMemcpyHostToHost,
     *     cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or
     *     cudaMemcpyDeviceToDevice.
     *   </p>
     *   <p>If the source and destination are both
     *     arrays, cudaMemcpy3D() will return an error if they do not have the
     *     same element size.
     *   </p>
     *   <p>The source and destination object may
     *     not overlap. If overlapping source and destination objects are
     *     specified, undefined
     *     behavior will result.
     *   </p>
     *   <p>The source object must lie entirely
     *     within the region defined by <tt>srcPos</tt> and <tt>extent</tt>.
     *     The destination object must lie entirely within the region defined by
     *     <tt>dstPos</tt> and <tt>extent</tt>.
     *   </p>
     *   <p>cudaMemcpy3D() returns an error if the
     *     pitch of <tt>srcPtr</tt> or <tt>dstPtr</tt> exceeds the maximum
     *     allowed. The pitch of a cudaPitchedPtr allocated with cudaMalloc3D()
     *     will always be valid.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           synchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param p 3D memory copy parameters
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc3DArray
     * @see JCuda#cudaMemset3D
     * @see JCuda#cudaMemcpy3DAsync
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     * @see cudaExtent
     * @see cudaPos
     */
    public static int cudaMemcpy3D(cudaMemcpy3DParms p)
    {
        return checkResult(cudaMemcpy3DNative(p));
    }
    private static native int cudaMemcpy3DNative(cudaMemcpy3DParms p);


    /**
     * Copies memory between devices.
     *
     * <pre>
     * cudaError_t cudaMemcpy3DPeer (
     *      const cudaMemcpy3DPeerParms* p )
     * </pre>
     * <div>
     *   <p>Copies memory between devices.  Perform
     *     a 3D memory copy according to the parameters specified in <tt>p</tt>.
     *     See the definition of the cudaMemcpy3DPeerParms structure for
     *     documentation of its parameters.
     *   </p>
     *   <p>Note that this function is synchronous
     *     with respect to the host only if the source or destination of the
     *     transfer is host
     *     memory. Note also that this copy is
     *     serialized with respect to all pending and future asynchronous work in
     *     to the current
     *     device, the copy's source device, and
     *     the copy's destination device (use cudaMemcpy3DPeerAsync to avoid this
     *     synchronization).
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           synchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param p Parameters for the memory copy
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpyPeer
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpyPeerAsync
     * @see JCuda#cudaMemcpy3DPeerAsync
     */
    public static int cudaMemcpy3DPeer(cudaMemcpy3DPeerParms p)
    {
        return checkResult(cudaMemcpy3DPeerNative(p));
    }
    private static native int cudaMemcpy3DPeerNative(cudaMemcpy3DPeerParms p);


    /**
     * Copies data between 3D objects.
     *
     * <pre>
     * cudaError_t cudaMemcpy3DAsync (
     *      const cudaMemcpy3DParms* p,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Copies data between 3D objects.
     *   <pre>struct cudaExtent {
     *   size_t width;
     *   size_t height;
     *   size_t depth;
     * };
     * struct cudaExtent
     *                   make_cudaExtent(size_t w, size_t h, size_t d);
     *
     * struct cudaPos {
     *   size_t x;
     *   size_t y;
     *   size_t z;
     * };
     * struct cudaPos
     *                   make_cudaPos(size_t x, size_t y, size_t z);
     *
     * struct cudaMemcpy3DParms {
     *   cudaArray_t
     *                   srcArray;
     *   struct cudaPos
     *                   srcPos;
     *   struct cudaPitchedPtr
     *                   srcPtr;
     *   cudaArray_t
     *                   dstArray;
     *   struct cudaPos
     *                   dstPos;
     *   struct cudaPitchedPtr
     *                   dstPtr;
     *   struct cudaExtent
     *                   extent;
     *   enum cudaMemcpyKind
     *                   kind;
     * };</pre>
     *   </p>
     *   <p>cudaMemcpy3DAsync() copies data betwen
     *     two 3D objects. The source and destination objects may be in either
     *     host memory, device memory, or a CUDA
     *     array. The source, destination, extent,
     *     and kind of copy performed is specified by the cudaMemcpy3DParms struct
     *     which should be initialized to zero before use:
     *   <pre>cudaMemcpy3DParms
     * myParms = {0};</pre>
     *   </p>
     *   <p>The struct passed to cudaMemcpy3DAsync()
     *     must specify one of <tt>srcArray</tt> or <tt>srcPtr</tt> and one of
     *     <tt>dstArray</tt> or <tt>dstPtr</tt>. Passing more than one non-zero
     *     source or destination will cause cudaMemcpy3DAsync() to return an
     *     error.
     *   </p>
     *   <p>The <tt>srcPos</tt> and <tt>dstPos</tt>
     *     fields are optional offsets into the source and destination objects
     *     and are defined in units of each object's elements. The
     *     element for a host or device pointer is
     *     assumed to be <strong>unsigned char</strong>. For CUDA arrays,
     *     positions must be in the range [0, 2048) for any dimension.
     *   </p>
     *   <p>The <tt>extent</tt> field defines the
     *     dimensions of the transferred area in elements. If a CUDA array is
     *     participating in the copy, the extent
     *     is defined in terms of that array's
     *     elements. If no CUDA array is participating in the copy then the
     *     extents are defined in
     *     elements of <strong>unsigned
     *     char</strong>.
     *   </p>
     *   <p>The <tt>kind</tt> field defines the
     *     direction of the copy. It must be one of cudaMemcpyHostToHost,
     *     cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or
     *     cudaMemcpyDeviceToDevice.
     *   </p>
     *   <p>If the source and destination are both
     *     arrays, cudaMemcpy3DAsync() will return an error if they do not have
     *     the same element size.
     *   </p>
     *   <p>The source and destination object may
     *     not overlap. If overlapping source and destination objects are
     *     specified, undefined
     *     behavior will result.
     *   </p>
     *   <p>The source object must lie entirely
     *     within the region defined by <tt>srcPos</tt> and <tt>extent</tt>.
     *     The destination object must lie entirely within the region defined by
     *     <tt>dstPos</tt> and <tt>extent</tt>.
     *   </p>
     *   <p>cudaMemcpy3DAsync() returns an error if
     *     the pitch of <tt>srcPtr</tt> or <tt>dstPtr</tt> exceeds the maximum
     *     allowed. The pitch of a cudaPitchedPtr allocated with cudaMalloc3D()
     *     will always be valid.
     *   </p>
     *   <p>cudaMemcpy3DAsync() is asynchronous with
     *     respect to the host, so the call may return before the copy is complete.
     *     The copy can optionally be
     *     associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>kind</tt> is
     *     cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and <tt>stream</tt>
     *     is non-zero, the copy may overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           asynchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param p 3D memory copy parameters
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc3DArray
     * @see JCuda#cudaMemset3D
     * @see JCuda#cudaMemcpy3D
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     * @see cudaExtent
     * @see cudaPos
     */
    public static int cudaMemcpy3DAsync(cudaMemcpy3DParms p, cudaStream_t stream)
    {
        return checkResult(cudaMemcpy3DAsyncNative(p, stream));
    }
    private static native int cudaMemcpy3DAsyncNative(cudaMemcpy3DParms p, cudaStream_t stream);


    /**
     * Copies memory between devices asynchronously.
     *
     * <pre>
     * cudaError_t cudaMemcpy3DPeerAsync (
     *      const cudaMemcpy3DPeerParms* p,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Copies memory between devices
     *     asynchronously.  Perform a 3D memory copy according to the parameters
     *     specified in <tt>p</tt>. See the definition of the cudaMemcpy3DPeerParms
     *     structure for documentation of its parameters.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           asynchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param p Parameters for the memory copy
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpyPeer
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpyPeerAsync
     * @see JCuda#cudaMemcpy3DPeerAsync
     */
    public static int cudaMemcpy3DPeerAsync(cudaMemcpy3DPeerParms p, cudaStream_t stream)
    {
        return checkResult(cudaMemcpy3DPeerAsyncNative(p, stream));
    }
    private static native int cudaMemcpy3DPeerAsyncNative(cudaMemcpy3DPeerParms p, cudaStream_t stream);



    /**
     * Gets free and total device memory.
     *
     * <pre>
     * cudaError_t cudaMemGetInfo (
     *      size_t* free,
     *      size_t* total )
     * </pre>
     * <div>
     *   <p>Gets free and total device memory.
     *     Returns in <tt>*free</tt> and <tt>*total</tt> respectively, the free
     *     and total amount of memory available for allocation by the device in
     *     bytes.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param free Returned free memory in bytes
     * @param total Returned total memory in bytes
     *
     * @return cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue,
     * cudaErrorLaunchFailure
     *
     */
    public static int cudaMemGetInfo(long free[], long total[])
    {
        return checkResult(cudaMemGetInfoNative(free, total));
    }
    private static native int cudaMemGetInfoNative(long free[], long total[]);

    /**
     * Gets info about the specified cudaArray.
     *
     * <pre>
     * cudaError_t cudaArrayGetInfo (
     *      cudaChannelFormatDesc* desc,
     *      cudaExtent* extent,
     *      unsigned int* flags,
     *      cudaArray_t array )
     * </pre>
     * <div>
     *   <p>Gets info about the specified cudaArray.
     *     Returns in <tt>*desc</tt>, <tt>*extent</tt> and <tt>*flags</tt>
     *     respectively, the type, shape and flags of <tt>array</tt>.
     *   </p>
     *   <p>Any of <tt>*desc</tt>, <tt>*extent</tt>
     *     and <tt>*flags</tt> may be specified as NULL.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param desc Returned array type
     * @param extent Returned array shape. 2D arrays will have depth of zero
     * @param flags Returned array flags
     * @param array The cudaArray to get info for
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     */
    public static int cudaArrayGetInfo(cudaChannelFormatDesc desc, cudaExtent extent, int flags[], cudaArray array)
    {
        return checkResult(cudaArrayGetInfoNative(desc, extent, flags, array));
    }
    private static native int cudaArrayGetInfoNative(cudaChannelFormatDesc desc, cudaExtent extent, int flags[], cudaArray array);


    /**
     * Allocates page-locked memory on the host.
     *
     * <pre>
     * cudaError_t cudaHostAlloc (
     *      void** pHost,
     *      size_t size,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Allocates page-locked memory on the host.
     *     Allocates <tt>size</tt> bytes of host memory that is page-locked and
     *     accessible to the device. The driver tracks the virtual memory ranges
     *     allocated
     *     with this function and automatically
     *     accelerates calls to functions such as cudaMemcpy(). Since the memory
     *     can be accessed directly by the device, it can be read or written with
     *     much higher bandwidth than pageable
     *     memory obtained with functions such as
     *     malloc(). Allocating excessive amounts of pinned memory may degrade
     *     system performance,
     *     since it reduces the amount of memory
     *     available to the system for paging. As a result, this function is best
     *     used sparingly
     *     to allocate staging areas for data
     *     exchange between host and device.
     *   </p>
     *   <p>The <tt>flags</tt> parameter enables
     *     different options to be specified that affect the allocation, as
     *     follows.
     *   <ul>
     *     <li>
     *       <p>cudaHostAllocDefault: This
     *         flag's value is defined to be 0 and causes cudaHostAlloc() to emulate
     *         cudaMallocHost().
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaHostAllocPortable: The
     *         memory returned by this call will be considered as pinned memory by
     *         all CUDA contexts, not just the one that performed
     *         the allocation.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaHostAllocMapped: Maps the
     *         allocation into the CUDA address space. The device pointer to the
     *         memory may be obtained by calling cudaHostGetDevicePointer().
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaHostAllocWriteCombined:
     *         Allocates the memory as write-combined (WC). WC memory can be
     *         transferred across the PCI Express bus more quickly on some
     *         system configurations, but
     *         cannot be read efficiently by most CPUs. WC memory is a good option
     *         for buffers that will be written
     *         by the CPU and read by the
     *         device via mapped pinned memory or host-&gt;device transfers.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>All of these flags are orthogonal to
     *     one another: a developer may allocate memory that is portable, mapped
     *     and/or write-combined
     *     with no restrictions.
     *   </p>
     *   <p>cudaSetDeviceFlags() must have been
     *     called with the cudaDeviceMapHost flag in order for the
     *     cudaHostAllocMapped flag to have any effect.
     *   </p>
     *   <p>The cudaHostAllocMapped flag may be
     *     specified on CUDA contexts for devices that do not support mapped
     *     pinned memory. The failure is deferred to cudaHostGetDevicePointer()
     *     because the memory may be mapped into other CUDA contexts via the
     *     cudaHostAllocPortable flag.
     *   </p>
     *   <p>Memory allocated by this function must
     *     be freed with cudaFreeHost().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pHost Device pointer to allocated memory
     * @param size Requested allocation size in bytes
     * @param flags Requested properties of allocated memory
     *
     * @return cudaSuccess, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaSetDeviceFlags
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     */
    public static int cudaHostAlloc(Pointer ptr, long size, int flags)
    {
        return checkResult(cudaHostAllocNative(ptr, size, flags));
    }
    private static native int cudaHostAllocNative(Pointer ptr, long size, int flags);



    /**
     * Registers an existing host memory range for use by CUDA.
     *
     * <pre>
     * cudaError_t cudaHostRegister (
     *      void* ptr,
     *      size_t size,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Registers an existing host memory range
     *     for use by CUDA.  Page-locks the memory range specified by <tt>ptr</tt>
     *     and <tt>size</tt> and maps it for the device(s) as specified by <tt>flags</tt>. This memory range also is added to the same tracking
     *     mechanism as cudaHostAlloc() to automatically accelerate calls to
     *     functions such as cudaMemcpy(). Since the memory can be accessed
     *     directly by the device, it can be read or written with much higher
     *     bandwidth than pageable
     *     memory that has not been registered.
     *     Page-locking excessive amounts of memory may degrade system performance,
     *     since it reduces
     *     the amount of memory available to the
     *     system for paging. As a result, this function is best used sparingly
     *     to register staging
     *     areas for data exchange between host and
     *     device.
     *   </p>
     *   <p>The <tt>flags</tt> parameter enables
     *     different options to be specified that affect the allocation, as
     *     follows.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaHostRegisterPortable: The
     *         memory returned by this call will be considered as pinned memory by
     *         all CUDA contexts, not just the one that performed
     *         the allocation.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaHostRegisterMapped: Maps
     *         the allocation into the CUDA address space. The device pointer to the
     *         memory may be obtained by calling cudaHostGetDevicePointer(). This
     *         feature is available only on GPUs with compute capability greater than
     *         or equal to 1.1.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaHostRegisterIoMemory: 
     *        The passed memory pointer is treated as
     *        pointing to some memory-mapped I/O space, e.g. belonging to a
     *        third-party PCIe device, and it will marked as non cache-coherent and
     *        contiguous.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>All of these flags are orthogonal to
     *     one another: a developer may page-lock memory that is portable or
     *     mapped with no restrictions.
     *   </p>
     *   <p>The CUDA context must have been created
     *     with the cudaMapHost flag in order for the cudaHostRegisterMapped flag
     *     to have any effect.
     *   </p>
     *   <p>The cudaHostRegisterMapped flag may be
     *     specified on CUDA contexts for devices that do not support mapped
     *     pinned memory. The failure is deferred to cudaHostGetDevicePointer()
     *     because the memory may be mapped into other CUDA contexts via the
     *     cudaHostRegisterPortable flag.
     *   </p>
     *   <p>The memory page-locked by this function
     *     must be unregistered with cudaHostUnregister().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param ptr Host pointer to memory to page-lock
     * @param size Size in bytes of the address range to page-lock in bytes
     * @param flags Flags for allocation request
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaHostUnregister
     * @see JCuda#cudaHostGetFlags
     * @see JCuda#cudaHostGetDevicePointer
     */
    public static int cudaHostRegister(Pointer ptr, long size, int flags)
    {
        return checkResult(cudaHostRegisterNative(ptr, size, flags));
    }
    private static native int cudaHostRegisterNative(Pointer ptr, long size, int flags);

    /**
     * Unregisters a memory range that was registered with cudaHostRegister.
     *
     * <pre>
     * cudaError_t cudaHostUnregister (
     *      void* ptr )
     * </pre>
     * <div>
     *   <p>Unregisters a memory range that was
     *     registered with cudaHostRegister.  Unmaps the memory range whose base
     *     address is specified
     *     by <tt>ptr</tt>, and makes it pageable
     *     again.
     *   </p>
     *   <p>The base address must be the same one
     *     specified to cudaHostRegister().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param ptr Host pointer to memory to unregister
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaHostUnregister
     */
    public static int cudaHostUnregister(Pointer ptr)
    {
        return checkResult(cudaHostUnregisterNative(ptr));
    }
    private static native int cudaHostUnregisterNative(Pointer ptr);


    /**
     * Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.
     *
     * <pre>
     * cudaError_t cudaHostGetDevicePointer (
     *      void** pDevice,
     *      void* pHost,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Passes back device pointer of mapped host
     *     memory allocated by cudaHostAlloc or registered by cudaHostRegister.
     *     Passes back
     *     the device pointer corresponding to the
     *     mapped, pinned host buffer allocated by cudaHostAlloc() or registered
     *     by cudaHostRegister().
     *   </p>
     *   <p>cudaHostGetDevicePointer() will fail if
     *     the cudaDeviceMapHost flag was not specified before deferred context
     *     creation occurred, or if called on a device that does not support
     *     mapped,
     *     pinned memory.
     *   </p>
     *   <p><tt>flags</tt> provides for future
     *     releases. For now, it must be set to 0.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pDevice Returned device pointer for mapped memory
     * @param pHost Requested host pointer mapping
     * @param flags Flags for extensions (must be 0 for now)
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaSetDeviceFlags
     * @see JCuda#cudaHostAlloc
     */
    public static int cudaHostGetDevicePointer(Pointer pDevice, Pointer pHost, int flags)
    {
        return checkResult(cudaHostGetDevicePointerNative(pDevice, pHost, flags));
    }
    private static native int cudaHostGetDevicePointerNative(Pointer pDevice, Pointer pHost, int flags);



    public static int cudaMallocManaged(Pointer devPtr, long size, int flags)
    {
      return checkResult(cudaMallocManagedNative(devPtr, size, flags));
    }
    private static native int cudaMallocManagedNative(Pointer devPtr, long size, int flags);


    /**
     * Allocate memory on the device.
     *
     * <pre>
     * cudaError_t cudaMalloc (
     *      void** devPtr,
     *      size_t size )
     * </pre>
     * <div>
     *   <p>Allocate memory on the device.  Allocates
     *     <tt>size</tt> bytes of linear memory on the device and returns in <tt>*devPtr</tt> a pointer to the allocated memory. The allocated memory
     *     is suitably aligned for any kind of variable. The memory is not
     *     cleared.
     *     cudaMalloc() returns cudaErrorMemoryAllocation
     *     in case of failure.
     *   </p>
     * </div>
     *
     * @param devPtr Pointer to allocated device memory
     * @param size Requested allocation size in bytes
     *
     * @return cudaSuccess, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaFree
     * @see JCuda#cudaMallocArray
     * @see JCuda#cudaFreeArray
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc3DArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaHostAlloc
     */
    public static int cudaMalloc(Pointer devPtr, long size)
    {
        return checkResult(cudaMallocNative(devPtr, size));
    }
    private static native int cudaMallocNative(Pointer devPtr, long size);


    /**
     * [C++ API] Allocates page-locked memory on the host
     *
     * <pre>
     * cudaError_t cudaMallocHost (
     *      void** ptr,
     *      size_t size,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>[C++ API] Allocates page-locked memory
     *     on the host  Allocates <tt>size</tt> bytes of host memory that is
     *     page-locked and accessible to the device. The driver tracks the virtual
     *     memory ranges allocated
     *     with this function and automatically
     *     accelerates calls to functions such as cudaMemcpy(). Since the memory
     *     can be accessed directly by the device, it can be read or written with
     *     much higher bandwidth than pageable
     *     memory obtained with functions such as
     *     malloc(). Allocating excessive amounts of pinned memory may degrade
     *     system performance,
     *     since it reduces the amount of memory
     *     available to the system for paging. As a result, this function is best
     *     used sparingly
     *     to allocate staging areas for data
     *     exchange between host and device.
     *   </p>
     *   <p>The <tt>flags</tt> parameter enables
     *     different options to be specified that affect the allocation, as
     *     follows.
     *   <ul>
     *     <li>
     *       <p>cudaHostAllocDefault: This
     *         flag's value is defined to be 0.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaHostAllocPortable: The
     *         memory returned by this call will be considered as pinned memory by
     *         all CUDA contexts, not just the one that performed
     *         the allocation.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaHostAllocMapped: Maps the
     *         allocation into the CUDA address space. The device pointer to the
     *         memory may be obtained by calling cudaHostGetDevicePointer().
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaHostAllocWriteCombined:
     *         Allocates the memory as write-combined (WC). WC memory can be
     *         transferred across the PCI Express bus more quickly on some
     *         system configurations, but
     *         cannot be read efficiently by most CPUs. WC memory is a good option
     *         for buffers that will be written
     *         by the CPU and read by the
     *         device via mapped pinned memory or host-&gt;device transfers.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>All of these flags are orthogonal to
     *     one another: a developer may allocate memory that is portable, mapped
     *     and/or write-combined
     *     with no restrictions.
     *   </p>
     *   <p>cudaSetDeviceFlags() must have been
     *     called with the cudaDeviceMapHost flag in order for the
     *     cudaHostAllocMapped flag to have any effect.
     *   </p>
     *   <p>The cudaHostAllocMapped flag may be
     *     specified on CUDA contexts for devices that do not support mapped
     *     pinned memory. The failure is deferred to cudaHostGetDevicePointer()
     *     because the memory may be mapped into other CUDA contexts via the
     *     cudaHostAllocPortable flag.
     *   </p>
     *   <p>Memory allocated by this function must
     *     be freed with cudaFreeHost().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param ptr Pointer to allocated host memory
     * @param size Requested allocation size in bytes
     * @param ptr Device pointer to allocated memory
     * @param size Requested allocation size in bytes
     * @param flags Requested properties of allocated memory
     *
     * @return cudaSuccess, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaSetDeviceFlags
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaHostAlloc
     */
    public static int cudaMallocHost(Pointer ptr, long size)
    {
        return checkResult(cudaMallocHostNative(ptr, size));
    }
    private static native int cudaMallocHostNative(Pointer ptr, long size);


    /**
     * Allocates pitched memory on the device.
     *
     * <pre>
     * cudaError_t cudaMallocPitch (
     *      void** devPtr,
     *      size_t* pitch,
     *      size_t width,
     *      size_t height )
     * </pre>
     * <div>
     *   <p>Allocates pitched memory on the device.
     *     Allocates at least <tt>width</tt> (in bytes) * <tt>height</tt> bytes
     *     of linear memory on the device and returns in <tt>*devPtr</tt> a
     *     pointer to the allocated memory. The function may pad the allocation
     *     to ensure that corresponding pointers in any given
     *     row will continue to meet the alignment
     *     requirements for coalescing as the address is updated from row to row.
     *     The pitch returned
     *     in <tt>*pitch</tt> by cudaMallocPitch()
     *     is the width in bytes of the allocation. The intended usage of <tt>pitch</tt> is as a separate parameter of the allocation, used to
     *     compute addresses within the 2D array. Given the row and column of
     *     an array element of type <tt>T</tt>,
     *     the address is computed as:
     *   <pre>    T* pElement = (T*)((char*)BaseAddress
     * + Row * pitch) + Column;</pre>
     *   </p>
     *   <p>For allocations of 2D arrays, it is
     *     recommended that programmers consider performing pitch allocations
     *     using cudaMallocPitch(). Due to pitch alignment restrictions in the
     *     hardware, this is especially true if the application will be performing
     *     2D memory
     *     copies between different regions of
     *     device memory (whether linear memory or CUDA arrays).
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param devPtr Pointer to allocated pitched device memory
     * @param pitch Pitch for allocation
     * @param width Requested pitched allocation width (in bytes)
     * @param height Requested pitched allocation height
     *
     * @return cudaSuccess, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaFree
     * @see JCuda#cudaMallocArray
     * @see JCuda#cudaFreeArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc3DArray
     * @see JCuda#cudaHostAlloc
     */
    public static int cudaMallocPitch(Pointer devPtr, long pitch[], long width, long height)
    {
        return checkResult(cudaMallocPitchNative(devPtr, pitch, width, height));
    }
    private static native int cudaMallocPitchNative(Pointer devPtr, long pitch[], long width, long height);


    /**
     * Allocate an array on the device.
     *
     * <pre>
     * cudaError_t cudaMallocArray (
     *      cudaArray_t* array,
     *      const cudaChannelFormatDesc* desc,
     *      size_t width,
     *      size_t height = 0,
     *      unsigned int  flags = 0 )
     * </pre>
     * <div>
     *   <p>Allocate an array on the device.
     *     Allocates a CUDA array according to the cudaChannelFormatDesc structure
     *     <tt>desc</tt> and returns a handle to the new CUDA array in <tt>*array</tt>.
     *   </p>
     *   <p>The cudaChannelFormatDesc is defined
     *     as:
     *   <pre>    struct cudaChannelFormatDesc {
     *         int x, y, z, w;
     *     enum cudaChannelFormatKind
     *                   f;
     *     };</pre>
     *   where cudaChannelFormatKind is one of
     *   cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, or
     *   cudaChannelFormatKindFloat.
     *   </p>
     *   <p>The <tt>flags</tt> parameter enables
     *     different options to be specified that affect the allocation, as
     *     follows.
     *   <ul>
     *     <li>
     *       <p>cudaArrayDefault: This flag's
     *         value is defined to be 0 and provides default array allocation
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArraySurfaceLoadStore:
     *         Allocates an array that can be read from or written to using a surface
     *         reference
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArrayTextureGather: This
     *         flag indicates that texture gather operations will be performed on the
     *         array.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p><tt>width</tt> and <tt>height</tt>
     *     must meet certain size requirements. See cudaMalloc3DArray() for more
     *     details.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param array Pointer to allocated array in device memory
     * @param desc Requested channel format
     * @param width Requested array allocation width
     * @param height Requested array allocation height
     * @param flags Requested properties of allocated array
     *
     * @return cudaSuccess, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaFree
     * @see JCuda#cudaFreeArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc3DArray
     * @see JCuda#cudaHostAlloc
     */
    public static int cudaMallocArray(cudaArray array, cudaChannelFormatDesc desc, long width, long height)
    {
        return cudaMallocArray(array, desc, width, height, 0);
    }

    /**
     * Allocate an array on the device.
     *
     * <pre>
     * cudaError_t cudaMallocArray (
     *      cudaArray_t* array,
     *      const cudaChannelFormatDesc* desc,
     *      size_t width,
     *      size_t height = 0,
     *      unsigned int  flags = 0 )
     * </pre>
     * <div>
     *   <p>Allocate an array on the device.
     *     Allocates a CUDA array according to the cudaChannelFormatDesc structure
     *     <tt>desc</tt> and returns a handle to the new CUDA array in <tt>*array</tt>.
     *   </p>
     *   <p>The cudaChannelFormatDesc is defined
     *     as:
     *   <pre>    struct cudaChannelFormatDesc {
     *         int x, y, z, w;
     *     enum cudaChannelFormatKind
     *                   f;
     *     };</pre>
     *   where cudaChannelFormatKind is one of
     *   cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, or
     *   cudaChannelFormatKindFloat.
     *   </p>
     *   <p>The <tt>flags</tt> parameter enables
     *     different options to be specified that affect the allocation, as
     *     follows.
     *   <ul>
     *     <li>
     *       <p>cudaArrayDefault: This flag's
     *         value is defined to be 0 and provides default array allocation
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArraySurfaceLoadStore:
     *         Allocates an array that can be read from or written to using a surface
     *         reference
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaArrayTextureGather: This
     *         flag indicates that texture gather operations will be performed on the
     *         array.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p><tt>width</tt> and <tt>height</tt>
     *     must meet certain size requirements. See cudaMalloc3DArray() for more
     *     details.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param array Pointer to allocated array in device memory
     * @param desc Requested channel format
     * @param width Requested array allocation width
     * @param height Requested array allocation height
     * @param flags Requested properties of allocated array
     *
     * @return cudaSuccess, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaFree
     * @see JCuda#cudaFreeArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc3DArray
     * @see JCuda#cudaHostAlloc
     */
    public static int cudaMallocArray(cudaArray array, cudaChannelFormatDesc desc, long width, long height, int flags)
    {
        return checkResult(cudaMallocArrayNative(array, desc, width, height, flags));
    }
    private static native int cudaMallocArrayNative(cudaArray array, cudaChannelFormatDesc desc, long width, long height, int flags);


    /**
     * Frees memory on the device.
     *
     * <pre>
     * cudaError_t cudaFree (
     *      void* devPtr )
     * </pre>
     * <div>
     *   <p>Frees memory on the device.  Frees the
     *     memory space pointed to by <tt>devPtr</tt>, which must have been
     *     returned by a previous call to cudaMalloc() or cudaMallocPitch().
     *     Otherwise, or if cudaFree(<tt>devPtr</tt>) has already been called
     *     before, an error is returned. If <tt>devPtr</tt> is 0, no operation
     *     is performed. cudaFree() returns cudaErrorInvalidDevicePointer in case
     *     of failure.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param devPtr Device pointer to memory to free
     *
     * @return cudaSuccess, cudaErrorInvalidDevicePointer,
     * cudaErrorInitializationError
     *
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaMallocArray
     * @see JCuda#cudaFreeArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc3DArray
     * @see JCuda#cudaHostAlloc
     */
    public static int cudaFree(Pointer devPtr)
    {
        return checkResult(cudaFreeNative(devPtr));
    }
    private static native int cudaFreeNative(Pointer devPtr);


    /**
     * Frees page-locked memory.
     *
     * <pre>
     * cudaError_t cudaFreeHost (
     *      void* ptr )
     * </pre>
     * <div>
     *   <p>Frees page-locked memory.  Frees the
     *     memory space pointed to by <tt>hostPtr</tt>, which must have been
     *     returned by a previous call to cudaMallocHost() or cudaHostAlloc().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param ptr Pointer to memory to free
     *
     * @return cudaSuccess, cudaErrorInitializationError
     *
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaFree
     * @see JCuda#cudaMallocArray
     * @see JCuda#cudaFreeArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaMalloc3D
     * @see JCuda#cudaMalloc3DArray
     * @see JCuda#cudaHostAlloc
     */
    public static int cudaFreeHost(Pointer ptr)
    {
        return checkResult(cudaFreeHostNative(ptr));
    }
    private static native int cudaFreeHostNative(Pointer ptr);


    /**
     * Frees an array on the device.
     *
     * <pre>
     * cudaError_t cudaFreeArray (
     *      cudaArray_t array )
     * </pre>
     * <div>
     *   <p>Frees an array on the device.  Frees the
     *     CUDA array <tt>array</tt>, which must have been * returned by a
     *     previous call to cudaMallocArray(). If cudaFreeArray(<tt>array</tt>)
     *     has already been called before, cudaErrorInvalidValue is returned. If
     *     <tt>devPtr</tt> is 0, no operation is performed.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param array Pointer to array to free
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInitializationError
     *
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaFree
     * @see JCuda#cudaMallocArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaHostAlloc
     */
    public static int cudaFreeArray(cudaArray array)
    {
        return checkResult(cudaFreeArrayNative(array));
    }
    private static native int cudaFreeArrayNative(cudaArray array);


    /**
     * Frees a mipmapped array on the device.
     *
     * <pre>
     * cudaError_t cudaFreeMipmappedArray (
     *      cudaMipmappedArray_t mipmappedArray )
     * </pre>
     * <div>
     *   <p>Frees a mipmapped array on the device.
     *     Frees the CUDA mipmapped array <tt>mipmappedArray</tt>, which must
     *     have been returned by a previous call to cudaMallocMipmappedArray().
     *     If cudaFreeMipmappedArray(<tt>mipmappedArray</tt>) has already been
     *     called before, cudaErrorInvalidValue is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param mipmappedArray Pointer to mipmapped array to free
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInitializationError
     *
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaMallocPitch
     * @see JCuda#cudaFree
     * @see JCuda#cudaMallocArray
     * @see JCuda#cudaMallocHost
     * @see JCuda#cudaFreeHost
     * @see JCuda#cudaHostAlloc
     */
    public static int cudaFreeMipmappedArray(cudaMipmappedArray mipmappedArray)
    {
        return checkResult(cudaFreeMipmappedArrayNative(mipmappedArray));
    }
    private static native int cudaFreeMipmappedArrayNative(cudaMipmappedArray mipmappedArray);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpy (
     *      void* dst,
     *      const void* src,
     *      size_t count,
     *      cudaMemcpyKind kind )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies <tt>count</tt> bytes from the memory area pointed to by <tt>src</tt> to the memory area pointed to by <tt>dst</tt>, where <tt>kind</tt> is one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
     *     cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice, and specifies the
     *     direction of the copy. The memory areas may not overlap. Calling
     *     cudaMemcpy() with <tt>dst</tt> and <tt>src</tt> pointers that do not
     *     match the direction of the copy results in an undefined behavior.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           synchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param src Source memory address
     * @param count Size in bytes to copy
     * @param kind Type of transfer
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpy(Pointer dst, Pointer src, long count, int cudaMemcpyKind_kind)
    {
        return checkResult(cudaMemcpyNative(dst, src, count, cudaMemcpyKind_kind));
    }
    private static native int cudaMemcpyNative(Pointer dst, Pointer src, long count, int cudaMemcpyKind_kind);


    /**
     * Copies memory between two devices.
     *
     * <pre>
     * cudaError_t cudaMemcpyPeer (
     *      void* dst,
     *      int  dstDevice,
     *      const void* src,
     *      int  srcDevice,
     *      size_t count )
     * </pre>
     * <div>
     *   <p>Copies memory between two devices.
     *     Copies memory from one device to memory on another device. <tt>dst</tt>
     *     is the base device pointer of the destination memory and <tt>dstDevice</tt> is the destination device. <tt>src</tt> is the base
     *     device pointer of the source memory and <tt>srcDevice</tt> is the
     *     source device. <tt>count</tt> specifies the number of bytes to copy.
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host, but serialized with respect all pending and
     *     future asynchronous
     *     work in to the current device, <tt>srcDevice</tt>, and <tt>dstDevice</tt> (use cudaMemcpyPeerAsync to
     *     avoid this synchronization).
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           synchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination device pointer
     * @param dstDevice Destination device
     * @param src Source device pointer
     * @param srcDevice Source device
     * @param count Size of memory copy in bytes
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpyPeerAsync
     * @see JCuda#cudaMemcpy3DPeerAsync
     */
    public static int cudaMemcpyPeer(Pointer dst, int dstDevice, Pointer src, int srcDevice, long count)
    {
        return checkResult(cudaMemcpyPeerNative(dst, dstDevice, src, srcDevice, count));
    }
    private static native int cudaMemcpyPeerNative(Pointer dst, int dstDevice, Pointer src, int srcDevice, long count);



    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpyToArray (
     *      cudaArray_t dst,
     *      size_t wOffset,
     *      size_t hOffset,
     *      const void* src,
     *      size_t count,
     *      cudaMemcpyKind kind )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies <tt>count</tt> bytes from the memory area pointed to by <tt>src</tt> to the CUDA array <tt>dst</tt> starting at the upper left
     *     corner (<tt>wOffset</tt>, <tt>hOffset</tt>), where <tt>kind</tt> is
     *     one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
     *     cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice, and specifies the
     *     direction of the copy.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           synchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param wOffset Destination starting X offset
     * @param hOffset Destination starting Y offset
     * @param src Source memory address
     * @param count Size in bytes to copy
     * @param kind Type of transfer
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpyToArray(cudaArray dst, long wOffset, long hOffset, Pointer src, long count, int cudaMemcpyKind_kind)
    {
        return checkResult(cudaMemcpyToArrayNative(dst, wOffset, hOffset, src, count, cudaMemcpyKind_kind));
    }
    private static native int cudaMemcpyToArrayNative(cudaArray dst, long wOffset, long hOffset, Pointer src, long count, int cudaMemcpyKind_kind);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpyFromArray (
     *      void* dst,
     *      cudaArray_const_t src,
     *      size_t wOffset,
     *      size_t hOffset,
     *      size_t count,
     *      cudaMemcpyKind kind )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies <tt>count</tt> bytes from the CUDA array <tt>src</tt> starting
     *     at the upper left corner (<tt>wOffset</tt>, hOffset) to the memory
     *     area pointed to by <tt>dst</tt>, where <tt>kind</tt> is one of
     *     cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
     *     or cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           synchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param src Source memory address
     * @param wOffset Source starting X offset
     * @param hOffset Source starting Y offset
     * @param count Size in bytes to copy
     * @param kind Type of transfer
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpyFromArray(Pointer dst, cudaArray src, long wOffset, long hOffset, long count, int cudaMemcpyKind_kind)
    {
        return checkResult(cudaMemcpyFromArrayNative(dst, src, wOffset, hOffset, count, cudaMemcpyKind_kind));
    }
    private static native int cudaMemcpyFromArrayNative(Pointer dst, cudaArray src, long wOffset, long hOffset, long count, int cudaMemcpyKind_kind);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpyArrayToArray (
     *      cudaArray_t dst,
     *      size_t wOffsetDst,
     *      size_t hOffsetDst,
     *      cudaArray_const_t src,
     *      size_t wOffsetSrc,
     *      size_t hOffsetSrc,
     *      size_t count,
     *      cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies <tt>count</tt> bytes from the CUDA array <tt>src</tt> starting
     *     at the upper left corner (<tt>wOffsetSrc</tt>, <tt>hOffsetSrc</tt>)
     *     to the CUDA array <tt>dst</tt> starting at the upper left corner (<tt>wOffsetDst</tt>, <tt>hOffsetDst</tt>) where <tt>kind</tt> is one of
     *     cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
     *     or cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param wOffsetDst Destination starting X offset
     * @param hOffsetDst Destination starting Y offset
     * @param src Source memory address
     * @param wOffsetSrc Source starting X offset
     * @param hOffsetSrc Source starting Y offset
     * @param count Size in bytes to copy
     * @param kind Type of transfer
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpyArrayToArray(cudaArray dst, long wOffsetDst, long hOffsetDst, cudaArray src, long wOffsetSrc, long hOffsetSrc, long count, int cudaMemcpyKind_kind)
    {
        return checkResult(cudaMemcpyArrayToArrayNative(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, cudaMemcpyKind_kind));
    }
    private static native int cudaMemcpyArrayToArrayNative(cudaArray dst, long wOffsetDst, long hOffsetDst, cudaArray src, long wOffsetSrc, long hOffsetSrc, long count, int cudaMemcpyKind_kind);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpy2D (
     *      void* dst,
     *      size_t dpitch,
     *      const void* src,
     *      size_t spitch,
     *      size_t width,
     *      size_t height,
     *      cudaMemcpyKind kind )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies a matrix (<tt>height</tt> rows of <tt>width</tt> bytes each)
     *     from the memory area pointed to by <tt>src</tt> to the memory area
     *     pointed to by <tt>dst</tt>, where <tt>kind</tt> is one of
     *     cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
     *     or cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
     *     <tt>dpitch</tt> and <tt>spitch</tt> are the widths in memory in bytes
     *     of the 2D arrays pointed to by <tt>dst</tt> and <tt>src</tt>,
     *     including any padding added to the end of each row. The memory areas
     *     may not overlap. <tt>width</tt> must not exceed either <tt>dpitch</tt>
     *     or <tt>spitch</tt>. Calling cudaMemcpy2D() with <tt>dst</tt> and <tt>src</tt> pointers that do not match the direction of the copy results
     *     in an undefined behavior. cudaMemcpy2D() returns an error if <tt>dpitch</tt> or <tt>spitch</tt> exceeds the maximum allowed.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param dpitch Pitch of destination memory
     * @param src Source memory address
     * @param spitch Pitch of source memory
     * @param width Width of matrix transfer (columns in bytes)
     * @param height Height of matrix transfer (rows)
     * @param kind Type of transfer
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue,
     * cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpy2D(Pointer dst, long dpitch, Pointer src, long spitch, long width, long height, int cudaMemcpyKind_kind)
    {
        return checkResult(cudaMemcpy2DNative(dst, dpitch, src, spitch, width, height, cudaMemcpyKind_kind));
    }
    private static native int cudaMemcpy2DNative(Pointer dst, long dpitch, Pointer src, long spitch, long width, long height, int cudaMemcpyKind_kind);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpy2DToArray (
     *      cudaArray_t dst,
     *      size_t wOffset,
     *      size_t hOffset,
     *      const void* src,
     *      size_t spitch,
     *      size_t width,
     *      size_t height,
     *      cudaMemcpyKind kind )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies a matrix (<tt>height</tt> rows of <tt>width</tt> bytes each)
     *     from the memory area pointed to by <tt>src</tt> to the CUDA array <tt>dst</tt> starting at the upper left corner (<tt>wOffset</tt>, <tt>hOffset</tt>) where <tt>kind</tt> is one of cudaMemcpyHostToHost,
     *     cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice,
     *     and specifies the direction of the copy. <tt>spitch</tt> is the width
     *     in memory in bytes of the 2D array pointed to by <tt>src</tt>,
     *     including any padding added to the end of each row. <tt>wOffset</tt>
     *     + <tt>width</tt> must not exceed the width of the CUDA array <tt>dst</tt>. <tt>width</tt> must not exceed <tt>spitch</tt>.
     *     cudaMemcpy2DToArray() returns an error if <tt>spitch</tt> exceeds the
     *     maximum allowed.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           synchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param wOffset Destination starting X offset
     * @param hOffset Destination starting Y offset
     * @param src Source memory address
     * @param spitch Pitch of source memory
     * @param width Width of matrix transfer (columns in bytes)
     * @param height Height of matrix transfer (rows)
     * @param kind Type of transfer
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpy2DToArray(cudaArray dst, long wOffset, long hOffset, Pointer src, long spitch, long width, long height, int cudaMemcpyKind_kind)
    {
        return checkResult(cudaMemcpy2DToArrayNative(dst, wOffset, hOffset, src, spitch, width, height, cudaMemcpyKind_kind));
    }
    private static native int cudaMemcpy2DToArrayNative(cudaArray dst, long wOffset, long hOffset, Pointer src, long spitch, long width, long height, int cudaMemcpyKind_kind);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpy2DFromArray (
     *      void* dst,
     *      size_t dpitch,
     *      cudaArray_const_t src,
     *      size_t wOffset,
     *      size_t hOffset,
     *      size_t width,
     *      size_t height,
     *      cudaMemcpyKind kind )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies a matrix (<tt>height</tt> rows of <tt>width</tt> bytes each)
     *     from the CUDA array <tt>srcArray</tt> starting at the upper left
     *     corner (<tt>wOffset</tt>, <tt>hOffset</tt>) to the memory area
     *     pointed to by <tt>dst</tt>, where <tt>kind</tt> is one of
     *     cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
     *     or cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
     *     <tt>dpitch</tt> is the width in memory in bytes of the 2D array
     *     pointed to by <tt>dst</tt>, including any padding added to the end of
     *     each row. <tt>wOffset</tt> + <tt>width</tt> must not exceed the width
     *     of the CUDA array <tt>src</tt>. <tt>width</tt> must not exceed <tt>dpitch</tt>. cudaMemcpy2DFromArray() returns an error if <tt>dpitch</tt> exceeds the maximum allowed.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           synchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param dpitch Pitch of destination memory
     * @param src Source memory address
     * @param wOffset Source starting X offset
     * @param hOffset Source starting Y offset
     * @param width Width of matrix transfer (columns in bytes)
     * @param height Height of matrix transfer (rows)
     * @param kind Type of transfer
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpy2DFromArray(Pointer dst, long dpitch, cudaArray src, long wOffset, long hOffset, long width, long height, int cudaMemcpyKind_kind)
    {
        return checkResult(cudaMemcpy2DFromArrayNative(dst, dpitch, src, wOffset, hOffset, width, height, cudaMemcpyKind_kind));
    }
    private static native int cudaMemcpy2DFromArrayNative(Pointer dst, long dpitch, cudaArray src, long wOffset, long hOffset, long width, long height, int cudaMemcpyKind_kind);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpy2DArrayToArray (
     *      cudaArray_t dst,
     *      size_t wOffsetDst,
     *      size_t hOffsetDst,
     *      cudaArray_const_t src,
     *      size_t wOffsetSrc,
     *      size_t hOffsetSrc,
     *      size_t width,
     *      size_t height,
     *      cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies a matrix (<tt>height</tt> rows of <tt>width</tt> bytes each)
     *     from the CUDA array <tt>srcArray</tt> starting at the upper left
     *     corner (<tt>wOffsetSrc</tt>, <tt>hOffsetSrc</tt>) to the CUDA array
     *     <tt>dst</tt> starting at the upper left corner (<tt>wOffsetDst</tt>,
     *     <tt>hOffsetDst</tt>), where <tt>kind</tt> is one of cudaMemcpyHostToHost,
     *     cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice,
     *     and specifies the direction of the copy. <tt>wOffsetDst</tt> + <tt>width</tt> must not exceed the width of the CUDA array <tt>dst</tt>.
     *     <tt>wOffsetSrc</tt> + <tt>width</tt> must not exceed the width of
     *     the CUDA array <tt>src</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           synchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param wOffsetDst Destination starting X offset
     * @param hOffsetDst Destination starting Y offset
     * @param src Source memory address
     * @param wOffsetSrc Source starting X offset
     * @param hOffsetSrc Source starting Y offset
     * @param width Width of matrix transfer (columns in bytes)
     * @param height Height of matrix transfer (rows)
     * @param kind Type of transfer
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpy2DArrayToArray(cudaArray dst, long wOffsetDst, long hOffsetDst, cudaArray src, long wOffsetSrc, long hOffsetSrc, long width, long height, int cudaMemcpyKind_kind)
    {
        return checkResult(cudaMemcpy2DArrayToArrayNative(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, cudaMemcpyKind_kind));
    }
    private static native int cudaMemcpy2DArrayToArrayNative(cudaArray dst, long wOffsetDst, long hOffsetDst, cudaArray src, long wOffsetSrc, long hOffsetSrc, long width, long height, int cudaMemcpyKind_kind);


    /**
     * [C++ API] Copies data to the given symbol on the device
     *
     * <pre>
     * template < class T > cudaError_t cudaMemcpyToSymbol (
     *      const T& symbol,
     *      const void* src,
     *      size_t count,
     *      size_t offset = 0,
     *      cudaMemcpyKind kind = cudaMemcpyHostToDevice ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Copies data to the given symbol
     *     on the device  Copies <tt>count</tt> bytes from the memory area
     *     pointed to by <tt>src</tt> to the memory area <tt>offset</tt> bytes
     *     from the start of symbol <tt>symbol</tt>. The memory areas may not
     *     overlap. <tt>symbol</tt> is a variable that resides in global or
     *     constant memory space. <tt>kind</tt> can be either cudaMemcpyHostToDevice
     *     or cudaMemcpyDeviceToDevice.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           synchronous behavior for most use cases.
     *         </p>
     *       </li>
     *       <li>
     *         <p>Use of a string naming a
     *           variable as the <tt>symbol</tt> paramater was deprecated in CUDA 4.1
     *           and removed in CUDA 5.0.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param symbol Device symbol address
     * @param src Source memory address
     * @param count Size in bytes to copy
     * @param offset Offset from start of symbol in bytes
     * @param kind Type of transfer
     * @param symbol Device symbol reference
     * @param src Source memory address
     * @param count Size in bytes to copy
     * @param offset Offset from start of symbol in bytes
     * @param kind Type of transfer
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidSymbol,
     * cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpyToSymbol(String symbol, Pointer src, long count, long offset, int cudaMemcpyKind_kind)
    {
        if (true)
        {
            throw new UnsupportedOperationException(
                "This function is no longer supported as of CUDA 5.0");
        }
        return checkResult(cudaMemcpyToSymbolNative(symbol, src, count, offset, cudaMemcpyKind_kind));
    }
    private static native int cudaMemcpyToSymbolNative(String symbol, Pointer src, long count, long offset, int cudaMemcpyKind_kind);

    /**
     * [C++ API] Copies data from the given symbol on the device
     *
     * <pre>
     * template < class T > cudaError_t cudaMemcpyFromSymbol (
     *      void* dst,
     *      const T& symbol,
     *      size_t count,
     *      size_t offset = 0,
     *      cudaMemcpyKind kind = cudaMemcpyDeviceToHost ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Copies data from the given
     *     symbol on the device  Copies <tt>count</tt> bytes from the memory area
     *     <tt>offset</tt> bytes from the start of symbol <tt>symbol</tt> to
     *     the memory area pointed to by <tt>dst</tt>. The memory areas may not
     *     overlap. <tt>symbol</tt> is a variable that resides in global or
     *     constant memory space. <tt>kind</tt> can be either cudaMemcpyDeviceToHost
     *     or cudaMemcpyDeviceToDevice.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           synchronous behavior for most use cases.
     *         </p>
     *       </li>
     *       <li>
     *         <p>Use of a string naming a
     *           variable as the <tt>symbol</tt> paramater was deprecated in CUDA 4.1
     *           and removed in CUDA 5.0.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param symbol Device symbol address
     * @param count Size in bytes to copy
     * @param offset Offset from start of symbol in bytes
     * @param kind Type of transfer
     * @param dst Destination memory address
     * @param symbol Device symbol reference
     * @param count Size in bytes to copy
     * @param offset Offset from start of symbol in bytes
     * @param kind Type of transfer
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidSymbol,
     * cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpyFromSymbol(Pointer dst, String symbol, long count, long offset, int cudaMemcpyKind_kind)
    {
        if (true)
        {
            throw new UnsupportedOperationException(
                "This function is no longer supported as of CUDA 5.0");
        }
        return checkResult(cudaMemcpyFromSymbolNative(dst, symbol, count, offset, cudaMemcpyKind_kind));
    }
    private static native int cudaMemcpyFromSymbolNative(Pointer dst, String symbol, long count, long offset, int cudaMemcpyKind_kind);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpyAsync (
     *      void* dst,
     *      const void* src,
     *      size_t count,
     *      cudaMemcpyKind kind,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies <tt>count</tt> bytes from the memory area pointed to by <tt>src</tt> to the memory area pointed to by <tt>dst</tt>, where <tt>kind</tt> is one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
     *     cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice, and specifies the
     *     direction of the copy. The memory areas may not overlap. Calling
     *     cudaMemcpyAsync() with <tt>dst</tt> and <tt>src</tt> pointers that
     *     do not match the direction of the copy results in an undefined
     *     behavior.
     *   </p>
     *   <p>cudaMemcpyAsync() is asynchronous with
     *     respect to the host, so the call may return before the copy is complete.
     *     The copy can optionally be
     *     associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>kind</tt> is
     *     cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and the <tt>stream</tt>
     *     is non-zero, the copy may overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           asynchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param src Source memory address
     * @param count Size in bytes to copy
     * @param kind Type of transfer
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpyAsync(Pointer dst, Pointer src, long count, int cudaMemcpyKind_kind, cudaStream_t stream)
    {
        return checkResult(cudaMemcpyAsyncNative(dst, src, count, cudaMemcpyKind_kind, stream));
    }
    private static native int cudaMemcpyAsyncNative(Pointer dst, Pointer src, long count, int cudaMemcpyKind_kind, cudaStream_t stream);

    /**
     * Copies memory between two devices asynchronously.
     *
     * <pre>
     * cudaError_t cudaMemcpyPeerAsync (
     *      void* dst,
     *      int  dstDevice,
     *      const void* src,
     *      int  srcDevice,
     *      size_t count,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Copies memory between two devices
     *     asynchronously.  Copies memory from one device to memory on another
     *     device. <tt>dst</tt> is the base device pointer of the destination
     *     memory and <tt>dstDevice</tt> is the destination device. <tt>src</tt>
     *     is the base device pointer of the source memory and <tt>srcDevice</tt>
     *     is the source device. <tt>count</tt> specifies the number of bytes to
     *     copy.
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host and all work in other streams and other
     *     devices.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           asynchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination device pointer
     * @param dstDevice Destination device
     * @param src Source device pointer
     * @param srcDevice Source device
     * @param count Size of memory copy in bytes
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpyPeer
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy3DPeerAsync
     */
    public static int cudaMemcpyPeerAsync(Pointer dst, int dstDevice, Pointer src, int srcDevice, long count, cudaStream_t stream)
    {
        return checkResult(cudaMemcpyPeerAsyncNative(dst, dstDevice, src, srcDevice, count, stream));
    }
    private static native int cudaMemcpyPeerAsyncNative(Pointer dst, int dstDevice, Pointer src, int srcDevice, long count, cudaStream_t stream);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpyToArrayAsync (
     *      cudaArray_t dst,
     *      size_t wOffset,
     *      size_t hOffset,
     *      const void* src,
     *      size_t count,
     *      cudaMemcpyKind kind,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies <tt>count</tt> bytes from the memory area pointed to by <tt>src</tt> to the CUDA array <tt>dst</tt> starting at the upper left
     *     corner (<tt>wOffset</tt>, <tt>hOffset</tt>), where <tt>kind</tt> is
     *     one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
     *     cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice, and specifies the
     *     direction of the copy.
     *   </p>
     *   <p>cudaMemcpyToArrayAsync() is asynchronous
     *     with respect to the host, so the call may return before the copy is
     *     complete. The copy can optionally be
     *     associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>kind</tt> is
     *     cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and <tt>stream</tt>
     *     is non-zero, the copy may overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           asynchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param wOffset Destination starting X offset
     * @param hOffset Destination starting Y offset
     * @param src Source memory address
     * @param count Size in bytes to copy
     * @param kind Type of transfer
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpyToArrayAsync(cudaArray dst, long wOffset, long hOffset, Pointer src, long count, int cudaMemcpyKind_kind, cudaStream_t stream)
    {
        return checkResult(cudaMemcpyToArrayAsyncNative(dst, wOffset, hOffset, src, count, cudaMemcpyKind_kind, stream));
    }
    private static native int cudaMemcpyToArrayAsyncNative(cudaArray dst, long wOffset, long hOffset, Pointer src, long count, int cudaMemcpyKind_kind, cudaStream_t stream);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpyFromArrayAsync (
     *      void* dst,
     *      cudaArray_const_t src,
     *      size_t wOffset,
     *      size_t hOffset,
     *      size_t count,
     *      cudaMemcpyKind kind,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies <tt>count</tt> bytes from the CUDA array <tt>src</tt> starting
     *     at the upper left corner (<tt>wOffset</tt>, hOffset) to the memory
     *     area pointed to by <tt>dst</tt>, where <tt>kind</tt> is one of
     *     cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
     *     or cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
     *   </p>
     *   <p>cudaMemcpyFromArrayAsync() is asynchronous
     *     with respect to the host, so the call may return before the copy is
     *     complete. The copy can optionally be
     *     associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>kind</tt> is
     *     cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and <tt>stream</tt>
     *     is non-zero, the copy may overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           asynchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param src Source memory address
     * @param wOffset Source starting X offset
     * @param hOffset Source starting Y offset
     * @param count Size in bytes to copy
     * @param kind Type of transfer
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpyFromArrayAsync(Pointer dst, cudaArray src, long wOffset, long hOffset, long count, int cudaMemcpyKind_kind, cudaStream_t stream)
    {
        return checkResult(cudaMemcpyFromArrayAsyncNative(dst, src, wOffset, hOffset, count, cudaMemcpyKind_kind, stream));
    }
    private static native int cudaMemcpyFromArrayAsyncNative(Pointer dst, cudaArray src, long wOffset, long hOffset, long count, int cudaMemcpyKind_kind, cudaStream_t stream);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpy2DAsync (
     *      void* dst,
     *      size_t dpitch,
     *      const void* src,
     *      size_t spitch,
     *      size_t width,
     *      size_t height,
     *      cudaMemcpyKind kind,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies a matrix (<tt>height</tt> rows of <tt>width</tt> bytes each)
     *     from the memory area pointed to by <tt>src</tt> to the memory area
     *     pointed to by <tt>dst</tt>, where <tt>kind</tt> is one of
     *     cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
     *     or cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
     *     <tt>dpitch</tt> and <tt>spitch</tt> are the widths in memory in bytes
     *     of the 2D arrays pointed to by <tt>dst</tt> and <tt>src</tt>,
     *     including any padding added to the end of each row. The memory areas
     *     may not overlap. <tt>width</tt> must not exceed either <tt>dpitch</tt>
     *     or <tt>spitch</tt>. Calling cudaMemcpy2DAsync() with <tt>dst</tt>
     *     and <tt>src</tt> pointers that do not match the direction of the copy
     *     results in an undefined behavior. cudaMemcpy2DAsync() returns an error
     *     if <tt>dpitch</tt> or <tt>spitch</tt> is greater than the maximum
     *     allowed.
     *   </p>
     *   <p>cudaMemcpy2DAsync() is asynchronous with
     *     respect to the host, so the call may return before the copy is complete.
     *     The copy can optionally be
     *     associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>kind</tt> is
     *     cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and <tt>stream</tt>
     *     is non-zero, the copy may overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           asynchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param dpitch Pitch of destination memory
     * @param src Source memory address
     * @param spitch Pitch of source memory
     * @param width Width of matrix transfer (columns in bytes)
     * @param height Height of matrix transfer (rows)
     * @param kind Type of transfer
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue,
     * cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpy2DAsync(Pointer dst, long dpitch, Pointer src, long spitch, long width, long height, int cudaMemcpyKind_kind, cudaStream_t stream)
    {
        return checkResult(cudaMemcpy2DAsyncNative(dst, dpitch, src, spitch, width, height, cudaMemcpyKind_kind, stream));
    }
    private static native int cudaMemcpy2DAsyncNative(Pointer dst, long dpitch, Pointer src, long spitch, long width, long height, int cudaMemcpyKind_kind, cudaStream_t stream);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpy2DToArrayAsync (
     *      cudaArray_t dst,
     *      size_t wOffset,
     *      size_t hOffset,
     *      const void* src,
     *      size_t spitch,
     *      size_t width,
     *      size_t height,
     *      cudaMemcpyKind kind,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies a matrix (<tt>height</tt> rows of <tt>width</tt> bytes each)
     *     from the memory area pointed to by <tt>src</tt> to the CUDA array <tt>dst</tt> starting at the upper left corner (<tt>wOffset</tt>, <tt>hOffset</tt>) where <tt>kind</tt> is one of cudaMemcpyHostToHost,
     *     cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice,
     *     and specifies the direction of the copy. <tt>spitch</tt> is the width
     *     in memory in bytes of the 2D array pointed to by <tt>src</tt>,
     *     including any padding added to the end of each row. <tt>wOffset</tt>
     *     + <tt>width</tt> must not exceed the width of the CUDA array <tt>dst</tt>. <tt>width</tt> must not exceed <tt>spitch</tt>.
     *     cudaMemcpy2DToArrayAsync() returns an error if <tt>spitch</tt> exceeds
     *     the maximum allowed.
     *   </p>
     *   <p>cudaMemcpy2DToArrayAsync() is asynchronous
     *     with respect to the host, so the call may return before the copy is
     *     complete. The copy can optionally be
     *     associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>kind</tt> is
     *     cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and <tt>stream</tt>
     *     is non-zero, the copy may overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           asynchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param wOffset Destination starting X offset
     * @param hOffset Destination starting Y offset
     * @param src Source memory address
     * @param spitch Pitch of source memory
     * @param width Width of matrix transfer (columns in bytes)
     * @param height Height of matrix transfer (rows)
     * @param kind Type of transfer
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpy2DToArrayAsync(cudaArray dst, long wOffset, long hOffset, Pointer src, long spitch, long width, long height, int cudaMemcpyKind_kind, cudaStream_t stream)
    {
        return checkResult(cudaMemcpy2DToArrayAsyncNative(dst, wOffset, hOffset, src, spitch, width, height, cudaMemcpyKind_kind, stream));
    }
    private static native int cudaMemcpy2DToArrayAsyncNative(cudaArray dst, long wOffset, long hOffset, Pointer src, long spitch, long width, long height, int cudaMemcpyKind_kind, cudaStream_t stream);


    /**
     * Copies data between host and device.
     *
     * <pre>
     * cudaError_t cudaMemcpy2DFromArrayAsync (
     *      void* dst,
     *      size_t dpitch,
     *      cudaArray_const_t src,
     *      size_t wOffset,
     *      size_t hOffset,
     *      size_t width,
     *      size_t height,
     *      cudaMemcpyKind kind,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Copies data between host and device.
     *     Copies a matrix (<tt>height</tt> rows of <tt>width</tt> bytes each)
     *     from the CUDA array <tt>srcArray</tt> starting at the upper left
     *     corner (<tt>wOffset</tt>, <tt>hOffset</tt>) to the memory area
     *     pointed to by <tt>dst</tt>, where <tt>kind</tt> is one of
     *     cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
     *     or cudaMemcpyDeviceToDevice, and specifies the direction of the copy.
     *     <tt>dpitch</tt> is the width in memory in bytes of the 2D array
     *     pointed to by <tt>dst</tt>, including any padding added to the end of
     *     each row. <tt>wOffset</tt> + <tt>width</tt> must not exceed the width
     *     of the CUDA array <tt>src</tt>. <tt>width</tt> must not exceed <tt>dpitch</tt>. cudaMemcpy2DFromArrayAsync() returns an error if <tt>dpitch</tt> exceeds the maximum allowed.
     *   </p>
     *   <p>cudaMemcpy2DFromArrayAsync() is
     *     asynchronous with respect to the host, so the call may return before
     *     the copy is complete. The copy can optionally be
     *     associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>kind</tt> is
     *     cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost and <tt>stream</tt>
     *     is non-zero, the copy may overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           asynchronous behavior for most use cases.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param dpitch Pitch of destination memory
     * @param src Source memory address
     * @param wOffset Source starting X offset
     * @param hOffset Source starting Y offset
     * @param width Width of matrix transfer (columns in bytes)
     * @param height Height of matrix transfer (rows)
     * @param kind Type of transfer
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpy2DFromArrayAsync(Pointer dst, long dpitch, cudaArray src, long wOffset, long hOffset, long width, long height, int cudaMemcpyKind_kind, cudaStream_t stream)
    {
        return checkResult(cudaMemcpy2DFromArrayAsyncNative(dst, dpitch, src, wOffset, hOffset, width, height, cudaMemcpyKind_kind, stream));
    }
    private static native int cudaMemcpy2DFromArrayAsyncNative(Pointer dst, long dpitch, cudaArray src, long wOffset, long hOffset, long width, long height, int cudaMemcpyKind_kind, cudaStream_t stream);


    /**
     * [C++ API] Copies data to the given symbol on the device
     *
     * <pre>
     * template < class T > cudaError_t cudaMemcpyToSymbolAsync (
     *      const T& symbol,
     *      const void* src,
     *      size_t count,
     *      size_t offset = 0,
     *      cudaMemcpyKind kind = cudaMemcpyHostToDevice,
     *      cudaStream_t stream = 0 ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Copies data to the given symbol
     *     on the device  Copies <tt>count</tt> bytes from the memory area
     *     pointed to by <tt>src</tt> to the memory area <tt>offset</tt> bytes
     *     from the start of symbol <tt>symbol</tt>. The memory areas may not
     *     overlap. <tt>symbol</tt> is a variable that resides in global or
     *     constant memory space. <tt>kind</tt> can be either cudaMemcpyHostToDevice
     *     or cudaMemcpyDeviceToDevice.
     *   </p>
     *   <p>cudaMemcpyToSymbolAsync() is asynchronous
     *     with respect to the host, so the call may return before the copy is
     *     complete. The copy can optionally be
     *     associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>kind</tt> is
     *     cudaMemcpyHostToDevice and <tt>stream</tt> is non-zero, the copy may
     *     overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           asynchronous behavior for most use cases.
     *         </p>
     *       </li>
     *       <li>
     *         <p>Use of a string naming a
     *           variable as the <tt>symbol</tt> paramater was deprecated in CUDA 4.1
     *           and removed in CUDA 5.0.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param symbol Device symbol address
     * @param src Source memory address
     * @param count Size in bytes to copy
     * @param offset Offset from start of symbol in bytes
     * @param kind Type of transfer
     * @param stream Stream identifier
     * @param symbol Device symbol reference
     * @param src Source memory address
     * @param count Size in bytes to copy
     * @param offset Offset from start of symbol in bytes
     * @param kind Type of transfer
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidSymbol,
     * cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyFromSymbolAsync
     */
    public static int cudaMemcpyToSymbolAsync(String symbol, Pointer src, long count, long offset, int cudaMemcpyKind_kind, cudaStream_t stream)
    {
        if (true)
        {
            throw new UnsupportedOperationException(
                "This function is no longer supported as of CUDA 5.0");
        }
        return checkResult(cudaMemcpyToSymbolAsyncNative(symbol, src, count, offset, cudaMemcpyKind_kind, stream));
    }
    private static native int cudaMemcpyToSymbolAsyncNative(String symbol, Pointer src, long count, long offset, int cudaMemcpyKind_kind, cudaStream_t stream);

    /**
     * [C++ API] Copies data from the given symbol on the device
     *
     * <pre>
     * template < class T > cudaError_t cudaMemcpyFromSymbolAsync (
     *      void* dst,
     *      const T& symbol,
     *      size_t count,
     *      size_t offset = 0,
     *      cudaMemcpyKind kind = cudaMemcpyDeviceToHost,
     *      cudaStream_t stream = 0 ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Copies data from the given
     *     symbol on the device  Copies <tt>count</tt> bytes from the memory area
     *     <tt>offset</tt> bytes from the start of symbol <tt>symbol</tt> to
     *     the memory area pointed to by <tt>dst</tt>. The memory areas may not
     *     overlap. <tt>symbol</tt> is a variable that resides in global or
     *     constant memory space. <tt>kind</tt> can be either cudaMemcpyDeviceToHost
     *     or cudaMemcpyDeviceToDevice.
     *   </p>
     *   <p>cudaMemcpyFromSymbolAsync() is
     *     asynchronous with respect to the host, so the call may return before
     *     the copy is complete. The copy can optionally be
     *     associated to a stream by passing a
     *     non-zero <tt>stream</tt> argument. If <tt>kind</tt> is
     *     cudaMemcpyDeviceToHost and <tt>stream</tt> is non-zero, the copy may
     *     overlap with operations in other streams.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>This function exhibits
     *           asynchronous behavior for most use cases.
     *         </p>
     *       </li>
     *       <li>
     *         <p>Use of a string naming a
     *           variable as the <tt>symbol</tt> paramater was deprecated in CUDA 4.1
     *           and removed in CUDA 5.0.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param dst Destination memory address
     * @param symbol Device symbol address
     * @param count Size in bytes to copy
     * @param offset Offset from start of symbol in bytes
     * @param kind Type of transfer
     * @param stream Stream identifier
     * @param dst Destination memory address
     * @param symbol Device symbol reference
     * @param count Size in bytes to copy
     * @param offset Offset from start of symbol in bytes
     * @param kind Type of transfer
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidSymbol,
     * cudaErrorInvalidDevicePointer, cudaErrorInvalidMemcpyDirection
     *
     * @see JCuda#cudaMemcpy
     * @see JCuda#cudaMemcpy2D
     * @see JCuda#cudaMemcpyToArray
     * @see JCuda#cudaMemcpy2DToArray
     * @see JCuda#cudaMemcpyFromArray
     * @see JCuda#cudaMemcpy2DFromArray
     * @see JCuda#cudaMemcpyArrayToArray
     * @see JCuda#cudaMemcpy2DArrayToArray
     * @see JCuda#cudaMemcpyToSymbol
     * @see JCuda#cudaMemcpyFromSymbol
     * @see JCuda#cudaMemcpyAsync
     * @see JCuda#cudaMemcpy2DAsync
     * @see JCuda#cudaMemcpyToArrayAsync
     * @see JCuda#cudaMemcpy2DToArrayAsync
     * @see JCuda#cudaMemcpyFromArrayAsync
     * @see JCuda#cudaMemcpy2DFromArrayAsync
     * @see JCuda#cudaMemcpyToSymbolAsync
     */
    public static int cudaMemcpyFromSymbolAsync(Pointer dst, String symbol, long count, long offset, int cudaMemcpyKind_kind, cudaStream_t stream)
    {
        if (true)
        {
            throw new UnsupportedOperationException(
                "This function is no longer supported as of CUDA 5.0");
        }
        return checkResult(cudaMemcpyFromSymbolAsyncNative(dst, symbol, count, offset, cudaMemcpyKind_kind, stream));
    }
    private static native int cudaMemcpyFromSymbolAsyncNative(Pointer dst, String symbol, long count, long offset, int cudaMemcpyKind_kind, cudaStream_t stream);



    /**
     * Initializes or sets device memory to a value.
     *
     * <pre>
     * cudaError_t cudaMemset (
     *      void* devPtr,
     *      int  value,
     *      size_t count )
     * </pre>
     * <div>
     *   <p>Initializes or sets device memory to a
     *     value.  Fills the first <tt>count</tt> bytes of the memory area
     *     pointed to by <tt>devPtr</tt> with the constant byte value <tt>value</tt>.
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host unless <tt>devPtr</tt> refers to pinned host
     *     memory.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param devPtr Pointer to device memory
     * @param value Value to set for each byte of specified memory
     * @param count Size in bytes to set
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInvalidDevicePointer
     *
     * @see JCuda#cudaMemset2D
     * @see JCuda#cudaMemset3D
     * @see JCuda#cudaMemsetAsync
     * @see JCuda#cudaMemset2DAsync
     * @see JCuda#cudaMemset3DAsync
     */
    public static int cudaMemset(Pointer mem, int c, long count)
    {
        return checkResult(cudaMemsetNative(mem, c, count));
    }
    private static native int cudaMemsetNative(Pointer mem, int c, long count);


    /**
     * Initializes or sets device memory to a value.
     *
     * <pre>
     * cudaError_t cudaMemset2D (
     *      void* devPtr,
     *      size_t pitch,
     *      int  value,
     *      size_t width,
     *      size_t height )
     * </pre>
     * <div>
     *   <p>Initializes or sets device memory to a
     *     value.  Sets to the specified value <tt>value</tt> a matrix (<tt>height</tt> rows of <tt>width</tt> bytes each) pointed to by <tt>dstPtr</tt>. <tt>pitch</tt> is the width in bytes of the 2D array
     *     pointed to by <tt>dstPtr</tt>, including any padding added to the end
     *     of each row. This function performs fastest when the pitch is one that
     *     has been passed
     *     back by cudaMallocPitch().
     *   </p>
     *   <p>Note that this function is asynchronous
     *     with respect to the host unless <tt>devPtr</tt> refers to pinned host
     *     memory.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param devPtr Pointer to 2D device memory
     * @param pitch Pitch in bytes of 2D device memory
     * @param value Value to set for each byte of specified memory
     * @param width Width of matrix set (columns in bytes)
     * @param height Height of matrix set (rows)
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInvalidDevicePointer
     *
     * @see JCuda#cudaMemset
     * @see JCuda#cudaMemset3D
     * @see JCuda#cudaMemsetAsync
     * @see JCuda#cudaMemset2DAsync
     * @see JCuda#cudaMemset3DAsync
     */
    public static int cudaMemset2D(Pointer mem, long pitch, int c, long width, long height)
    {
        return checkResult(cudaMemset2DNative(mem, pitch, c, width, height));
    }
    private static native int cudaMemset2DNative(Pointer mem, long pitch, int c, long width, long height);



    /**
     * Get the channel descriptor of an array.
     *
     * <pre>
     * cudaError_t cudaGetChannelDesc (
     *      cudaChannelFormatDesc* desc,
     *      cudaArray_const_t array )
     * </pre>
     * <div>
     *   <p>Get the channel descriptor of an array.
     *     Returns in <tt>*desc</tt> the channel descriptor of the CUDA array
     *     <tt>array</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param desc Channel format
     * @param array Memory array on device
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaCreateChannelDesc
     * @see JCuda#cudaGetTextureReference
     * @see JCuda#cudaBindTexture
     * @see JCuda#cudaBindTexture2D
     * @see JCuda#cudaBindTextureToArray
     * @see JCuda#cudaUnbindTexture
     * @see JCuda#cudaGetTextureAlignmentOffset
     */
    public static int cudaGetChannelDesc(cudaChannelFormatDesc desc, cudaArray array)
    {
        return checkResult(cudaGetChannelDescNative(desc, array));
    }
    private static native int cudaGetChannelDescNative(cudaChannelFormatDesc desc, cudaArray array);


    /**
     * [C++ API] Returns a channel descriptor using the specified format
     *
     * <pre>
     * template < class T > cudaChannelFormatDesc cudaCreateChannelDesc (
     *      void ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Returns a channel descriptor
     *     using the specified format  Returns a channel descriptor with format
     *     <tt>f</tt> and number of bits of each component <tt>x</tt>, <tt>y</tt>, <tt>z</tt>, and <tt>w</tt>. The cudaChannelFormatDesc is
     *     defined as:
     *   <pre>  struct cudaChannelFormatDesc {
     *     int x, y, z, w;
     *     enum cudaChannelFormatKind
     *                   f;
     *   };</pre>
     *   </p>
     *   <p>where cudaChannelFormatKind is one of
     *     cudaChannelFormatKindSigned, cudaChannelFormatKindUnsigned, or
     *     cudaChannelFormatKindFloat.
     *   </p>
     * </div>
     *
     * @param x X component
     * @param y Y component
     * @param z Z component
     * @param w W component
     * @param f Channel format
     *
     * @return Channel descriptor with format f
     *
     * @see JCuda#cudaCreateChannelDesc
     * @see JCuda#cudaGetChannelDesc
     * @see JCuda#cudaGetTextureReference
     * @see JCuda#cudaBindTexture
     * @see JCuda#cudaBindTexture
     *
     * @see JCuda#cudaBindTexture2D
     * @see JCuda#cudaBindTextureToArray
     * @see JCuda#cudaBindTextureToArray
     *
     * @see JCuda#cudaUnbindTexture
     * @see JCuda#cudaGetTextureAlignmentOffset
     */
    public static cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, int cudaChannelFormatKind_f)
    {
        return cudaCreateChannelDescNative(x,y,z,w,cudaChannelFormatKind_f);
    }
    private static native cudaChannelFormatDesc cudaCreateChannelDescNative(int x, int y, int z, int w, int cudaChannelFormatKind_f);

    /**
     * Returns the last error from a runtime call.
     *
     * <pre>
     * cudaError_t cudaGetLastError (
     *      void )
     * </pre>
     * <div>
     *   <p>Returns the last error from a runtime
     *     call.  Returns the last error that has been produced by any of the
     *     runtime calls in
     *     the same host thread and resets it to
     *     cudaSuccess.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return cudaSuccess, cudaErrorMissingConfiguration,
     * cudaErrorMemoryAllocation, cudaErrorInitializationError,
     * cudaErrorLaunchFailure, cudaErrorLaunchTimeout,
     * cudaErrorLaunchOutOfResources, cudaErrorInvalidDeviceFunction,
     * cudaErrorInvalidConfiguration, cudaErrorInvalidDevice,
     * cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidSymbol,
     * cudaErrorUnmapBufferObjectFailed, cudaErrorInvalidHostPointer,
     * cudaErrorInvalidDevicePointer, cudaErrorInvalidTexture,
     * cudaErrorInvalidTextureBinding, cudaErrorInvalidChannelDescriptor,
     * cudaErrorInvalidMemcpyDirection, cudaErrorInvalidFilterSetting,
     * cudaErrorInvalidNormSetting, cudaErrorUnknown,
     * cudaErrorInvalidResourceHandle, cudaErrorInsufficientDriver,
     * cudaErrorSetOnActiveProcess, cudaErrorStartupFailure,
     *
     * @see JCuda#cudaPeekAtLastError
     * @see JCuda#cudaGetErrorString
     *
     */
    public static int cudaGetLastError()
    {
        return checkResult(cudaGetLastErrorNative());
    }
    private static native int cudaGetLastErrorNative();


    /**
     * Returns the last error from a runtime call.
     *
     * <pre>
     * cudaError_t cudaPeekAtLastError (
     *      void )
     * </pre>
     * <div>
     *   <p>Returns the last error from a runtime
     *     call.  Returns the last error that has been produced by any of the
     *     runtime calls in
     *     the same host thread. Note that this call
     *     does not reset the error to cudaSuccess like cudaGetLastError().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return cudaSuccess, cudaErrorMissingConfiguration,
     * cudaErrorMemoryAllocation, cudaErrorInitializationError,
     * cudaErrorLaunchFailure, cudaErrorLaunchTimeout,
     * cudaErrorLaunchOutOfResources, cudaErrorInvalidDeviceFunction,
     * cudaErrorInvalidConfiguration, cudaErrorInvalidDevice,
     * cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidSymbol,
     * cudaErrorUnmapBufferObjectFailed, cudaErrorInvalidHostPointer,
     * cudaErrorInvalidDevicePointer, cudaErrorInvalidTexture,
     * cudaErrorInvalidTextureBinding, cudaErrorInvalidChannelDescriptor,
     * cudaErrorInvalidMemcpyDirection, cudaErrorInvalidFilterSetting,
     * cudaErrorInvalidNormSetting, cudaErrorUnknown,
     * cudaErrorInvalidResourceHandle, cudaErrorInsufficientDriver,
     * cudaErrorSetOnActiveProcess, cudaErrorStartupFailure,
     *
     * @see JCuda#cudaGetLastError
     * @see JCuda#cudaGetErrorString
     *
     */
    public static int cudaPeekAtLastError()
    {
        return checkResult(cudaPeekAtLastErrorNative());
    }
    private static native int cudaPeekAtLastErrorNative();

    /**
     * <code><pre>
     * \brief Returns the string representation of an error code enum name
     *
     * Returns a string containing the name of an error code in the enum, or NULL
     * if the error code is not valid.
     *
     * \param error - Error code to convert to string
     *
     * \return
     * \p char* pointer to a NULL-terminated string, or NULL if the error code is not valid.
     *
     * \sa ::cudaGetErrorString, ::cudaGetLastError, ::cudaPeekAtLastError, ::cudaError
     * </pre></code>
     */
    public static String cudaGetErrorName(int error)
    {
        return cudaGetErrorNameNative(error);
    }
    private static native String cudaGetErrorNameNative(int error);

    /**
     * Returns the message string from an error code.
     *
     * <div>
     *   <div>
     *     <table>
     *       <tr>
     *         <td>const char* cudaGetErrorString           </td>
     *         <td>(</td>
     *         <td>cudaError_t&nbsp;</td>
     *         <td> <em>error</em>          </td>
     *         <td>&nbsp;)&nbsp;</td>
     *         <td></td>
     *       </tr>
     *     </table>
     *   </div>
     *   <div>
     *     <p>
     *       Returns the message string from an error code.
     *     <p>
     *   </div>
     * </div>
     *
     * @return <code>char*</code> pointer to a NULL-terminated string
     *
     * @see JCuda#cudaGetLastError
     * @see JCuda#cudaPeekAtLastError
     * @see cudaError
     */
    public static String cudaGetErrorString(int error)
    {
        return cudaGetErrorStringNative(error);
    }
    private static native String cudaGetErrorStringNative(int error);


    /**
     * Create an asynchronous stream.
     *
     * <pre>
     * cudaError_t cudaStreamCreate (
     *      cudaStream_t* pStream )
     * </pre>
     * <div>
     *   <p>Create an asynchronous stream.  Creates
     *     a new asynchronous stream.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pStream Pointer to new stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaStreamCreate
     * @see JCuda#cudaStreamCreateWithFlags
     * @see JCuda#cudaStreamQuery
     * @see JCuda#cudaStreamSynchronize
     * @see JCuda#cudaStreamWaitEvent
     * @see JCuda#cudaStreamAddCallback
     * @see JCuda#cudaStreamDestroy
     */
    public static int cudaStreamCreate(cudaStream_t stream)
    {
        return checkResult(cudaStreamCreateNative(stream));
    }
    private static native int cudaStreamCreateNative(cudaStream_t stream);


    /**
     * Create an asynchronous stream.
     *
     * <pre>
     * cudaError_t cudaStreamCreateWithFlags (
     *      cudaStream_t* pStream,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Create an asynchronous stream.  Creates
     *     a new asynchronous stream. The <tt>flags</tt> argument determines the
     *     behaviors of the stream. Valid values for <tt>flags</tt> are
     *   <ul>
     *     <li>
     *       <p>cudaStreamDefault: Default
     *         stream creation flag.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaStreamNonBlocking: Specifies
     *         that work running in the created stream may run concurrently with work
     *         in stream 0 (the NULL stream), and that
     *         the created stream should
     *         perform no implicit synchronization with stream 0.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pStream Pointer to new stream identifier
     * @param flags Parameters for stream creation
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaStreamCreate
     * @see JCuda#cudaStreamQuery
     * @see JCuda#cudaStreamSynchronize
     * @see JCuda#cudaStreamWaitEvent
     * @see JCuda#cudaStreamAddCallback
     * @see JCuda#cudaStreamDestroy
     */
    public static int cudaStreamCreateWithFlags(cudaStream_t pStream, int flags)
    {
        return checkResult(cudaStreamCreateWithFlagsNative(pStream, flags));
    }
    private static native int cudaStreamCreateWithFlagsNative(cudaStream_t pStream, int flags);

    public static int cudaStreamCreateWithPriority(cudaStream_t pStream, int flags, int priority)
    {
        return checkResult(cudaStreamCreateWithPriorityNative(pStream, flags, priority));
    }
    private static native int cudaStreamCreateWithPriorityNative(cudaStream_t pStream, int flags, int priority);

    public static int cudaStreamGetPriority(cudaStream_t hStream, int priority[])
    {
        return checkResult(cudaStreamGetPriorityNative(hStream, priority));
    }
    private static native int cudaStreamGetPriorityNative(cudaStream_t hStream, int priority[]);

    public static int cudaStreamGetFlags(cudaStream_t hStream, int flags[])
    {
        return checkResult(cudaStreamGetFlagsNative(hStream, flags));
    }
    private static native int cudaStreamGetFlagsNative(cudaStream_t hStream, int flags[]);


    /**
     * Destroys and cleans up an asynchronous stream.
     *
     * <pre>
     * cudaError_t cudaStreamDestroy (
     *      cudaStream_t stream )
     * </pre>
     * <div>
     *   <p>Destroys and cleans up an asynchronous
     *     stream.  Destroys and cleans up the asynchronous stream specified by
     *     <tt>stream</tt>.
     *   </p>
     *   <p>In case the device is still doing work
     *     in the stream <tt>stream</tt> when cudaStreamDestroy() is called, the
     *     function will return immediately and the resources associated with <tt>stream</tt> will be released automatically once the device has
     *     completed all work in <tt>stream</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidResourceHandle
     *
     * @see JCuda#cudaStreamCreate
     * @see JCuda#cudaStreamCreateWithFlags
     * @see JCuda#cudaStreamQuery
     * @see JCuda#cudaStreamWaitEvent
     * @see JCuda#cudaStreamSynchronize
     * @see JCuda#cudaStreamAddCallback
     */
    public static int cudaStreamDestroy(cudaStream_t stream)
    {
        return checkResult(cudaStreamDestroyNative(stream));
    }
    private static native int cudaStreamDestroyNative(cudaStream_t stream);


    /**
     * Make a compute stream wait on an event.
     *
     * <pre>
     * cudaError_t cudaStreamWaitEvent (
     *      cudaStream_t stream,
     *      cudaEvent_t event,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Make a compute stream wait on an event.
     *     Makes all future work submitted to <tt>stream</tt> wait until <tt>event</tt> reports completion before beginning execution. This
     *     synchronization will be performed efficiently on the device. The event
     *     <tt>event</tt> may be from a different
     *     context than <tt>stream</tt>, in which case this function will perform
     *     cross-device synchronization.
     *   </p>
     *   <p>The stream <tt>stream</tt> will wait
     *     only for the completion of the most recent host call to cudaEventRecord()
     *     on <tt>event</tt>. Once this call has returned, any functions
     *     (including cudaEventRecord() and cudaEventDestroy()) may be called on
     *     <tt>event</tt> again, and the subsequent calls will not have any
     *     effect on <tt>stream</tt>.
     *   </p>
     *   <p>If <tt>stream</tt> is NULL, any future
     *     work submitted in any stream will wait for <tt>event</tt> to complete
     *     before beginning execution. This effectively creates a barrier for all
     *     future work submitted to the device on
     *     this thread.
     *   </p>
     *   <p>If cudaEventRecord() has not been called
     *     on <tt>event</tt>, this call acts as if the record has already
     *     completed, and so is a functional no-op.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param stream Stream to wait
     * @param event Event to wait on
     * @param flags Parameters for the operation (must be 0)
     *
     * @return cudaSuccess, cudaErrorInvalidResourceHandle
     *
     * @see JCuda#cudaStreamCreate
     * @see JCuda#cudaStreamCreateWithFlags
     * @see JCuda#cudaStreamQuery
     * @see JCuda#cudaStreamSynchronize
     * @see JCuda#cudaStreamAddCallback
     * @see JCuda#cudaStreamDestroy
     */
    public static int cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, int flags)
    {
        return checkResult(cudaStreamWaitEventNative(stream, event, flags));
    }
    private static native int cudaStreamWaitEventNative(cudaStream_t stream, cudaEvent_t event, int flags);


    /**
     * Add a callback to a compute stream.
     *
     * <pre>
     * cudaError_t cudaStreamAddCallback (
     *      cudaStream_t stream,
     *      cudaStreamCallback_t callback,
     *      void* userData,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Add a callback to a compute stream.  Adds
     *     a callback to be called on the host after all currently enqueued items
     *     in the stream
     *     have completed. For each
     *     cudaStreamAddCallback call, a callback will be executed exactly once.
     *     The callback will block later
     *     work in the stream until it is finished.
     *   </p>
     *   <p>The callback may be passed cudaSuccess
     *     or an error code. In the event of a device error, all subsequently
     *     executed callbacks will receive an appropriate cudaError_t.
     *   </p>
     *   <p>Callbacks must not make any CUDA API
     *     calls. Attempting to use CUDA APIs will result in cudaErrorNotPermitted.
     *     Callbacks must not perform any synchronization that may depend on
     *     outstanding device work or other callbacks that are not
     *     mandated to run earlier. Callbacks
     *     without a mandated order (in independent streams) execute in undefined
     *     order and may be
     *     serialized.
     *   </p>
     *   <p>This API requires compute capability
     *     1.1 or greater. See cudaDeviceGetAttribute or cudaGetDeviceProperties
     *     to query compute capability. Calling this API with an earlier compute
     *     version will return cudaErrorNotSupported.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param stream Stream to add callback to
     * @param callback The function to call once preceding stream operations are complete
     * @param userData User specified data to be passed to the callback function
     * @param flags Reserved for future use, must be 0
     *
     * @return cudaSuccess, cudaErrorInvalidResourceHandle,
     * cudaErrorNotSupported
     *
     * @see JCuda#cudaStreamCreate
     * @see JCuda#cudaStreamCreateWithFlags
     * @see JCuda#cudaStreamQuery
     * @see JCuda#cudaStreamSynchronize
     * @see JCuda#cudaStreamWaitEvent
     * @see JCuda#cudaStreamDestroy
     */
    public static int cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback callback, Object userData, int flags)
    {
        return checkResult(cudaStreamAddCallbackNative(stream, callback, userData, flags));
    }
    private static native int cudaStreamAddCallbackNative(cudaStream_t stream, cudaStreamCallback callback, Object userData, int flags);

    /**
     * Waits for stream tasks to complete.
     *
     * <pre>
     * cudaError_t cudaStreamSynchronize (
     *      cudaStream_t stream )
     * </pre>
     * <div>
     *   <p>Waits for stream tasks to complete.
     *     Blocks until <tt>stream</tt> has completed all operations. If the
     *     cudaDeviceScheduleBlockingSync flag was set for this device, the host
     *     thread will block until the stream is finished with all of its tasks.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidResourceHandle
     *
     * @see JCuda#cudaStreamCreate
     * @see JCuda#cudaStreamCreateWithFlags
     * @see JCuda#cudaStreamQuery
     * @see JCuda#cudaStreamWaitEvent
     * @see JCuda#cudaStreamAddCallback
     * @see JCuda#cudaStreamDestroy
     */
    public static int cudaStreamSynchronize(cudaStream_t stream)
    {
        return checkResult(cudaStreamSynchronizeNative(stream));
    }
    private static native int cudaStreamSynchronizeNative(cudaStream_t stream);


    /**
     * Queries an asynchronous stream for completion status.
     *
     * <pre>
     * cudaError_t cudaStreamQuery (
     *      cudaStream_t stream )
     * </pre>
     * <div>
     *   <p>Queries an asynchronous stream for
     *     completion status.  Returns cudaSuccess if all operations in <tt>stream</tt> have completed, or cudaErrorNotReady if not.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorNotReady, cudaErrorInvalidResourceHandle
     *
     * @see JCuda#cudaStreamCreate
     * @see JCuda#cudaStreamCreateWithFlags
     * @see JCuda#cudaStreamWaitEvent
     * @see JCuda#cudaStreamSynchronize
     * @see JCuda#cudaStreamAddCallback
     * @see JCuda#cudaStreamDestroy
     */
    public static int cudaStreamQuery(cudaStream_t stream)
    {
        return checkResult(cudaStreamQueryNative(stream));
    }
    private static native int cudaStreamQueryNative(cudaStream_t stream);


    public static int cudaStreamAttachMemAsync(cudaStream_t stream, Pointer devPtr, long length, int flags)
    {
      return checkResult(cudaStreamAttachMemAsyncNative(stream, devPtr, length, flags));
    }
    private static native int cudaStreamAttachMemAsyncNative(cudaStream_t stream, Pointer devPtr, long length, int flags);


    /**
     * [C++ API] Creates an event object with the specified flags
     *
     * <pre>
     * cudaError_t cudaEventCreate (
     *      cudaEvent_t* event,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>[C++ API] Creates an event object with
     *     the specified flags  Creates an event object with the specified flags.
     *     Valid flags
     *     include:
     *   <ul>
     *     <li>
     *       <p>cudaEventDefault: Default event
     *         creation flag.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaEventBlockingSync: Specifies
     *         that event should use blocking synchronization. A host thread that uses
     *         cudaEventSynchronize() to wait on an event created with this flag will
     *         block until the event actually completes.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaEventDisableTiming:
     *         Specifies that the created event does not need to record timing data.
     *         Events created with this flag specified and the cudaEventBlockingSync
     *         flag not specified will provide the best performance when used with
     *         cudaStreamWaitEvent() and cudaEventQuery().
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param event Newly created event
     * @param event Newly created event
     * @param flags Flags for new event
     *
     * @return cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue,
     * cudaErrorLaunchFailure, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaEventCreate
     * @see JCuda#cudaEventCreateWithFlags
     * @see JCuda#cudaEventRecord
     * @see JCuda#cudaEventQuery
     * @see JCuda#cudaEventSynchronize
     * @see JCuda#cudaEventDestroy
     * @see JCuda#cudaEventElapsedTime
     * @see JCuda#cudaStreamWaitEvent
     */
    public static int cudaEventCreate(cudaEvent_t event)
    {
        return checkResult(cudaEventCreateNative(event));
    }
    private static native int cudaEventCreateNative(cudaEvent_t event);


    /**
     * Creates an event object with the specified flags.
     *
     * <pre>
     * cudaError_t cudaEventCreateWithFlags (
     *      cudaEvent_t* event,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Creates an event object with the specified
     *     flags.  Creates an event object with the specified flags. Valid flags
     *     include:
     *   <ul>
     *     <li>
     *       <p>cudaEventDefault: Default event
     *         creation flag.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaEventBlockingSync: Specifies
     *         that event should use blocking synchronization. A host thread that uses
     *         cudaEventSynchronize() to wait on an event created with this flag will
     *         block until the event actually completes.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaEventDisableTiming:
     *         Specifies that the created event does not need to record timing data.
     *         Events created with this flag specified and the cudaEventBlockingSync
     *         flag not specified will provide the best performance when used with
     *         cudaStreamWaitEvent() and cudaEventQuery().
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaEventInterprocess: Specifies
     *         that the created event may be used as an interprocess event by
     *         cudaIpcGetEventHandle(). cudaEventInterprocess must be specified along
     *         with cudaEventDisableTiming.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param event Newly created event
     * @param flags Flags for new event
     *
     * @return cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue,
     * cudaErrorLaunchFailure, cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaEventCreate
     * @see JCuda#cudaEventSynchronize
     * @see JCuda#cudaEventDestroy
     * @see JCuda#cudaEventElapsedTime
     * @see JCuda#cudaStreamWaitEvent
     */
    public static int cudaEventCreateWithFlags (cudaEvent_t event, int flags)
    {
        return checkResult(cudaEventCreateWithFlagsNative(event, flags));
    }
    private static native int cudaEventCreateWithFlagsNative(cudaEvent_t event, int flags);


    /**
     * Records an event.
     *
     * <pre>
     * cudaError_t cudaEventRecord (
     *      cudaEvent_t event,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Records an event.  Records an event. If
     *     <tt>stream</tt> is non-zero, the event is recorded after all preceding
     *     operations in <tt>stream</tt> have been completed; otherwise, it is
     *     recorded after all preceding operations in the CUDA context have been
     *     completed. Since
     *     operation is asynchronous, cudaEventQuery()
     *     and/or cudaEventSynchronize() must be used to determine when the event
     *     has actually been recorded.
     *   </p>
     *   <p>If cudaEventRecord() has previously been
     *     called on <tt>event</tt>, then this call will overwrite any existing
     *     state in <tt>event</tt>. Any subsequent calls which examine the status
     *     of <tt>event</tt> will only examine the completion of this most recent
     *     call to cudaEventRecord().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param event Event to record
     * @param stream Stream in which to record event
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInitializationError,
     * cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure
     *
     * @see JCuda#cudaEventCreate
     * @see JCuda#cudaEventCreateWithFlags
     * @see JCuda#cudaEventQuery
     * @see JCuda#cudaEventSynchronize
     * @see JCuda#cudaEventDestroy
     * @see JCuda#cudaEventElapsedTime
     * @see JCuda#cudaStreamWaitEvent
     */
    public static int cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
    {
        return checkResult(cudaEventRecordNative(event, stream));
    }
    private static native int cudaEventRecordNative(cudaEvent_t event, cudaStream_t stream);


    /**
     * Queries an event's status.
     *
     * <pre>
     * cudaError_t cudaEventQuery (
     *      cudaEvent_t event )
     * </pre>
     * <div>
     *   <p>Queries an event's status.  Query the
     *     status of all device work preceding the most recent call to
     *     cudaEventRecord() (in the appropriate compute streams, as specified by
     *     the arguments to cudaEventRecord()).
     *   </p>
     *   <p>If this work has successfully been
     *     completed by the device, or if cudaEventRecord() has not been called
     *     on <tt>event</tt>, then cudaSuccess is returned. If this work has not
     *     yet been completed by the device then cudaErrorNotReady is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param event Event to query
     *
     * @return cudaSuccess, cudaErrorNotReady, cudaErrorInitializationError,
     * cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
     * cudaErrorLaunchFailure
     *
     * @see JCuda#cudaEventCreate
     * @see JCuda#cudaEventCreateWithFlags
     * @see JCuda#cudaEventRecord
     * @see JCuda#cudaEventSynchronize
     * @see JCuda#cudaEventDestroy
     * @see JCuda#cudaEventElapsedTime
     */
    public static int cudaEventQuery(cudaEvent_t event)
    {
        return checkResult(cudaEventQueryNative(event));
    }
    private static native int cudaEventQueryNative(cudaEvent_t event);


    /**
     * Waits for an event to complete.
     *
     * <pre>
     * cudaError_t cudaEventSynchronize (
     *      cudaEvent_t event )
     * </pre>
     * <div>
     *   <p>Waits for an event to complete.  Wait
     *     until the completion of all device work preceding the most recent call
     *     to cudaEventRecord() (in the appropriate compute streams, as specified
     *     by the arguments to cudaEventRecord()).
     *   </p>
     *   <p>If cudaEventRecord() has not been called
     *     on <tt>event</tt>, cudaSuccess is returned immediately.
     *   </p>
     *   <p>Waiting for an event that was created
     *     with the cudaEventBlockingSync flag will cause the calling CPU thread
     *     to block until the event has been completed by the device. If the
     *     cudaEventBlockingSync flag has not been set, then the CPU thread will
     *     busy-wait until the event has been completed by the device.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param event Event to wait for
     *
     * @return cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue,
     * cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure
     *
     * @see JCuda#cudaEventCreate
     * @see JCuda#cudaEventCreateWithFlags
     * @see JCuda#cudaEventRecord
     * @see JCuda#cudaEventQuery
     * @see JCuda#cudaEventDestroy
     * @see JCuda#cudaEventElapsedTime
     */
    public static int cudaEventSynchronize(cudaEvent_t event)
    {
        return checkResult(cudaEventSynchronizeNative(event));
    }
    private static native int cudaEventSynchronizeNative(cudaEvent_t event);


    /**
     * Destroys an event object.
     *
     * <pre>
     * cudaError_t cudaEventDestroy (
     *      cudaEvent_t event )
     * </pre>
     * <div>
     *   <p>Destroys an event object.  Destroys the
     *     event specified by <tt>event</tt>.
     *   </p>
     *   <p>In case <tt>event</tt> has been recorded
     *     but has not yet been completed when cudaEventDestroy() is called, the
     *     function will return immediately and the resources associated with <tt>event</tt> will be released automatically once the device has completed
     *     <tt>event</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param event Event to destroy
     *
     * @return cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue,
     * cudaErrorLaunchFailure
     *
     * @see JCuda#cudaEventCreate
     * @see JCuda#cudaEventCreateWithFlags
     * @see JCuda#cudaEventQuery
     * @see JCuda#cudaEventSynchronize
     * @see JCuda#cudaEventRecord
     * @see JCuda#cudaEventElapsedTime
     */
    public static int cudaEventDestroy(cudaEvent_t event)
    {
        return checkResult(cudaEventDestroyNative(event));
    }
    private static native int cudaEventDestroyNative(cudaEvent_t event);


    /**
     * Computes the elapsed time between events.
     *
     * <pre>
     * cudaError_t cudaEventElapsedTime (
     *      float* ms,
     *      cudaEvent_t start,
     *      cudaEvent_t end )
     * </pre>
     * <div>
     *   <p>Computes the elapsed time between events.
     *     Computes the elapsed time between two events (in milliseconds with a
     *     resolution
     *     of around 0.5 microseconds).
     *   </p>
     *   <p>If either event was last recorded in a
     *     non-NULL stream, the resulting time may be greater than expected (even
     *     if both used
     *     the same stream handle). This happens
     *     because the cudaEventRecord() operation takes place asynchronously and
     *     there is no guarantee that the measured latency is actually just
     *     between the two
     *     events. Any number of other different
     *     stream operations could execute in between the two measured events,
     *     thus altering the
     *     timing in a significant way.
     *   </p>
     *   <p>If cudaEventRecord() has not been called
     *     on either event, then cudaErrorInvalidResourceHandle is returned. If
     *     cudaEventRecord() has been called on both events but one or both of
     *     them has not yet been completed (that is, cudaEventQuery() would return
     *     cudaErrorNotReady on at least one of the events), cudaErrorNotReady is
     *     returned. If either event was created with the cudaEventDisableTiming
     *     flag, then this function will return cudaErrorInvalidResourceHandle.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param ms Time between start and end in ms
     * @param start Starting event
     * @param end Ending event
     *
     * @return cudaSuccess, cudaErrorNotReady, cudaErrorInvalidValue,
     * cudaErrorInitializationError, cudaErrorInvalidResourceHandle,
     * cudaErrorLaunchFailure
     *
     * @see JCuda#cudaEventCreate
     * @see JCuda#cudaEventCreateWithFlags
     * @see JCuda#cudaEventQuery
     * @see JCuda#cudaEventSynchronize
     * @see JCuda#cudaEventDestroy
     * @see JCuda#cudaEventRecord
     */
    public static int cudaEventElapsedTime(float ms[], cudaEvent_t start, cudaEvent_t end)
    {
        return checkResult(cudaEventElapsedTimeNative(ms, start, end));
    }
    private static native int cudaEventElapsedTimeNative(float ms[], cudaEvent_t start, cudaEvent_t end);



    /**
     * Destroy all allocations and reset all state on the current device in the current process.
     *
     * <pre>
     * cudaError_t cudaDeviceReset (
     *      void )
     * </pre>
     * <div>
     *   <p>Destroy all allocations and reset all
     *     state on the current device in the current process.  Explicitly destroys
     *     and cleans
     *     up all resources associated with the
     *     current device in the current process. Any subsequent API call to this
     *     device will reinitialize
     *     the device.
     *   </p>
     *   <p>Note that this function will reset the
     *     device immediately. It is the caller's responsibility to ensure that
     *     the device is
     *     not being accessed by any other host
     *     threads from the process when this function is called.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return cudaSuccess
     *
     * @see JCuda#cudaDeviceSynchronize
     */
    public static int cudaDeviceReset()
    {
        return checkResult(cudaDeviceResetNative());
    }
    private static native int cudaDeviceResetNative();

    /**
     * Wait for compute device to finish.
     *
     * <pre>
     * cudaError_t cudaDeviceSynchronize (
     *      void )
     * </pre>
     * <div>
     *   <p>Wait for compute device to finish.
     *     Blocks until the device has completed all preceding requested tasks.
     *     cudaDeviceSynchronize() returns an error if one of the preceding tasks
     *     has failed. If the cudaDeviceScheduleBlockingSync flag was set for this
     *     device, the host thread will block until the device has finished its
     *     work.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return cudaSuccess
     *
     * @see JCuda#cudaDeviceReset
     */
    public static int cudaDeviceSynchronize()
    {
        return checkResult(cudaDeviceSynchronizeNative());
    }
    private static native int cudaDeviceSynchronizeNative();

    /**
     * Set resource limits.
     *
     * <pre>
     * cudaError_t cudaDeviceSetLimit (
     *      cudaLimit limit,
     *      size_t value )
     * </pre>
     * <div>
     *   <p>Set resource limits.  Setting <tt>limit</tt> to <tt>value</tt> is a request by the application to
     *     update the current limit maintained by the device. The driver is free
     *     to modify the requested
     *     value to meet h/w requirements (this
     *     could be clamping to minimum or maximum values, rounding up to nearest
     *     element size,
     *     etc). The application can use
     *     cudaDeviceGetLimit() to find out exactly what the limit has been set
     *     to.
     *   </p>
     *   <p>Setting each cudaLimit has its own
     *     specific restrictions, so each is discussed here.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaLimitStackSize controls
     *         the stack size in bytes of each GPU thread. This limit is only
     *         applicable to devices of compute capability 2.0 and
     *         higher. Attempting to set this
     *         limit on devices of compute capability less than 2.0 will result in
     *         the error cudaErrorUnsupportedLimit being returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaLimitPrintfFifoSize controls
     *         the size in bytes of the shared FIFO used by the printf() and fprintf()
     *         device system calls. Setting cudaLimitPrintfFifoSize must be performed
     *         before launching any kernel that uses the printf() or fprintf() device
     *         system calls, otherwise cudaErrorInvalidValue will be returned. This
     *         limit is only applicable to devices of compute capability 2.0 and
     *         higher. Attempting to set this limit
     *         on devices of compute capability
     *         less than 2.0 will result in the error cudaErrorUnsupportedLimit being
     *         returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaLimitMallocHeapSize controls
     *         the size in bytes of the heap used by the malloc() and free() device
     *         system calls. Setting cudaLimitMallocHeapSize must be performed before
     *         launching any kernel that uses the malloc() or free() device system
     *         calls, otherwise cudaErrorInvalidValue will be returned. This limit is
     *         only applicable to devices of compute capability 2.0 and higher.
     *         Attempting to set this limit
     *         on devices of compute capability
     *         less than 2.0 will result in the error cudaErrorUnsupportedLimit being
     *         returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaLimitDevRuntimeSyncDepth
     *         controls the maximum nesting depth of a grid at which a thread can
     *         safely call cudaDeviceSynchronize(). Setting this limit must be
     *         performed before any launch of a kernel that uses the device runtime
     *         and calls cudaDeviceSynchronize() above the default sync depth, two
     *         levels of grids. Calls to cudaDeviceSynchronize() will fail with error
     *         code cudaErrorSyncDepthExceeded if the limitation is violated. This
     *         limit can be set smaller than the default or up the maximum launch
     *         depth of 24. When
     *         setting this limit, keep in mind
     *         that additional levels of sync depth require the runtime to reserve
     *         large amounts of device
     *         memory which can no longer be
     *         used for user allocations. If these reservations of device memory fail,
     *         cudaDeviceSetLimit will return cudaErrorMemoryAllocation, and the limit
     *         can be reset to a lower value. This limit is only applicable to devices
     *         of compute capability 3.5 and higher.
     *         Attempting to set this limit on
     *         devices of compute capability less than 3.5 will result in the error
     *         cudaErrorUnsupportedLimit being returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaLimitDevRuntimePendingLaunchCount
     *         controls the maximum number of outstanding device runtime launches that
     *         can be made from the current device. A grid is outstanding
     *         from the point of launch up
     *         until the grid is known to have been completed. Device runtime launches
     *         which violate this limitation
     *         fail and return
     *         cudaErrorLaunchPendingCountExceeded when cudaGetLastError() is called
     *         after launch. If more pending launches than the default (2048 launches)
     *         are needed for a module using the device
     *         runtime, this limit can be
     *         increased. Keep in mind that being able to sustain additional pending
     *         launches will require the
     *         runtime to reserve larger
     *         amounts of device memory upfront which can no longer be used for
     *         allocations. If these reservations
     *         fail, cudaDeviceSetLimit will
     *         return cudaErrorMemoryAllocation, and the limit can be reset to a lower
     *         value. This limit is only applicable to devices of compute capability
     *         3.5 and higher.
     *         Attempting to set this limit on
     *         devices of compute capability less than 3.5 will result in the error
     *         cudaErrorUnsupportedLimit being returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param limit Limit to set
     * @param value Size of limit
     *
     * @return cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue,
     * cudaErrorMemoryAllocation
     *
     * @see JCuda#cudaDeviceGetLimit
     */
    public static int cudaDeviceSetLimit(int limit, long value)
    {
        return checkResult(cudaDeviceSetLimitNative(limit, value));
    }
    private static native int cudaDeviceSetLimitNative(int limit, long value);

    /**
     * Returns resource limits.
     *
     * <pre>
     * cudaError_t cudaDeviceGetLimit (
     *      size_t* pValue,
     *      cudaLimit limit )
     * </pre>
     * <div>
     *   <p>Returns resource limits.  Returns in <tt>*pValue</tt> the current size of <tt>limit</tt>. The supported
     *     cudaLimit values are:
     *   <ul>
     *     <li>
     *       <p>cudaLimitStackSize: stack size
     *         in bytes of each GPU thread;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaLimitPrintfFifoSize: size
     *         in bytes of the shared FIFO used by the printf() and fprintf() device
     *         system calls.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaLimitMallocHeapSize: size
     *         in bytes of the heap used by the malloc() and free() device system
     *         calls;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaLimitDevRuntimeSyncDepth:
     *         maximum grid depth at which a thread can isssue the device runtime call
     *         cudaDeviceSynchronize() to wait on child grid launches to complete.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaLimitDevRuntimePendingLaunchCount:
     *         maximum number of outstanding device runtime launches.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pValue Returned size of the limit
     * @param limit Limit to query
     *
     * @return cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue
     *
     * @see JCuda#cudaDeviceSetLimit
     */
    public static int cudaDeviceGetLimit(long pValue[], int limit)
    {
        return checkResult(cudaDeviceGetLimitNative(pValue, limit));
    }
    private static native int cudaDeviceGetLimitNative(long pValue[], int limit);

    /**
     * Returns the preferred cache configuration for the current device.
     *
     * <pre>
     * cudaError_t cudaDeviceGetCacheConfig (
     *      cudaFuncCache ** pCacheConfig )
     * </pre>
     * <div>
     *   <p>Returns the preferred cache configuration
     *     for the current device.  On devices where the L1 cache and shared
     *     memory use the
     *     same hardware resources, this returns
     *     through <tt>pCacheConfig</tt> the preferred cache configuration for
     *     the current device. This is only a preference. The runtime will use
     *     the requested configuration
     *     if possible, but it is free to choose a
     *     different configuration if required to execute functions.
     *   </p>
     *   <p>This will return a <tt>pCacheConfig</tt>
     *     of cudaFuncCachePreferNone on devices where the size of the L1 cache
     *     and shared memory are fixed.
     *   </p>
     *   <p>The supported cache configurations are:
     *   <ul>
     *     <li>
     *       <p>cudaFuncCachePreferNone: no
     *         preference for shared memory or L1 (default)
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaFuncCachePreferShared:
     *         prefer larger shared memory and smaller L1 cache
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaFuncCachePreferL1: prefer
     *         larger L1 cache and smaller shared memory
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCacheConfig Returned cache configuration
     *
     * @return cudaSuccess, cudaErrorInitializationError
     *
     * @see JCuda#cudaDeviceSetCacheConfig
     * @see JCuda#cudaDeviceSetCacheConfig
     * @see JCuda#cudaDeviceSetCacheConfig
     */
    public static int cudaDeviceGetCacheConfig(int pCacheConfig[])
    {
        return checkResult(cudaDeviceGetCacheConfigNative(pCacheConfig));
    }
    private static native int cudaDeviceGetCacheConfigNative(int pCacheConfig[]);


    public static int cudaDeviceGetStreamPriorityRange(int leastPriority[], int greatestPriority[])
    {
        return checkResult(cudaDeviceGetStreamPriorityRangeNative(leastPriority, greatestPriority));
    }
    private static native int cudaDeviceGetStreamPriorityRangeNative(int leastPriority[], int greatestPriority[]);


    /**
     * Returns the shared memory configuration for the current device.
     *
     * <pre>
     * cudaError_t cudaDeviceGetSharedMemConfig (
     *      cudaSharedMemConfig ** pConfig )
     * </pre>
     * <div>
     *   <p>Returns the shared memory configuration
     *     for the current device.  This function will return in <tt>pConfig</tt>
     *     the current size of shared memory banks on the current device. On
     *     devices with configurable shared memory banks, cudaDeviceSetSharedMemConfig
     *     can be used to change this setting, so that all subsequent kernel
     *     launches will by default use the new bank size. When
     *     cudaDeviceGetSharedMemConfig is called on devices without configurable
     *     shared memory, it will return the fixed bank size of the hardware.
     *   </p>
     *   <p>The returned bank configurations can be
     *     either:
     *   <ul>
     *     <li>
     *       <p>cudaSharedMemBankSizeFourByte
     *         - shared memory bank width is four bytes.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaSharedMemBankSizeEightByte
     *         - shared memory bank width is eight bytes.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pConfig Returned cache configuration
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInitializationError
     *
     * @see JCuda#cudaDeviceSetCacheConfig
     * @see JCuda#cudaDeviceGetCacheConfig
     * @see JCuda#cudaDeviceSetSharedMemConfig
     * @see JCuda#cudaDeviceSetCacheConfig
     */
    public static int cudaDeviceGetSharedMemConfig(int pConfig[])
    {
        return checkResult(cudaDeviceGetSharedMemConfigNative(pConfig));
    }
    private static native int cudaDeviceGetSharedMemConfigNative(int pConfig[]);


    /**
     * Sets the shared memory configuration for the current device.
     *
     * <pre>
     * cudaError_t cudaDeviceSetSharedMemConfig (
     *      cudaSharedMemConfig config )
     * </pre>
     * <div>
     *   <p>Sets the shared memory configuration for
     *     the current device.  On devices with configurable shared memory banks,
     *     this function
     *     will set the shared memory bank size
     *     which is used for all subsequent kernel launches. Any per-function
     *     setting of shared
     *     memory set via cudaFuncSetSharedMemConfig
     *     will override the device wide setting.
     *   </p>
     *   <p>Changing the shared memory configuration
     *     between launches may introduce a device side synchronization
     *     point.
     *   </p>
     *   <p>Changing the shared memory bank size
     *     will not increase shared memory usage or affect occupancy of kernels,
     *     but may have major
     *     effects on performance. Larger bank sizes
     *     will allow for greater potential bandwidth to shared memory, but will
     *     change what
     *     kinds of accesses to shared memory will
     *     result in bank conflicts.
     *   </p>
     *   <p>This function will do nothing on devices
     *     with fixed shared memory bank size.
     *   </p>
     *   <p>The supported bank configurations are:
     *   <ul>
     *     <li>
     *       <p>cudaSharedMemBankSizeDefault:
     *         set bank width the device default (currently, four bytes)
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaSharedMemBankSizeFourByte:
     *         set shared memory bank width to be four bytes natively.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaSharedMemBankSizeEightByte:
     *         set shared memory bank width to be eight bytes natively.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param config Requested cache configuration
     *
     * @return cudaSuccess, cudaErrorInvalidValue,
     * cudaErrorInitializationError
     *
     * @see JCuda#cudaDeviceSetCacheConfig
     * @see JCuda#cudaDeviceGetCacheConfig
     * @see JCuda#cudaDeviceGetSharedMemConfig
     * @see JCuda#cudaDeviceSetCacheConfig
     */
    public static int cudaDeviceSetSharedMemConfig(int config)
    {
        return checkResult(cudaDeviceSetSharedMemConfigNative(config));
    }
    private static native int cudaDeviceSetSharedMemConfigNative(int config);

    /**
     * Sets the preferred cache configuration for the current device.
     *
     * <pre>
     * cudaError_t cudaDeviceSetCacheConfig (
     *      cudaFuncCache cacheConfig )
     * </pre>
     * <div>
     *   <p>Sets the preferred cache configuration
     *     for the current device.  On devices where the L1 cache and shared
     *     memory use the same
     *     hardware resources, this sets through
     *     <tt>cacheConfig</tt> the preferred cache configuration for the current
     *     device. This is only a preference. The runtime will use the requested
     *     configuration
     *     if possible, but it is free to choose a
     *     different configuration if required to execute the function. Any
     *     function preference
     *     set via cudaDeviceSetCacheConfig ( C API)
     *     or cudaDeviceSetCacheConfig ( C++ API) will be preferred over this
     *     device-wide setting. Setting the device-wide cache configuration to
     *     cudaFuncCachePreferNone will cause subsequent kernel launches to prefer
     *     to not change the cache configuration unless required to launch the
     *     kernel.
     *   </p>
     *   <p>This setting does nothing on devices
     *     where the size of the L1 cache and shared memory are fixed.
     *   </p>
     *   <p>Launching a kernel with a different
     *     preference than the most recent preference setting may insert a
     *     device-side synchronization
     *     point.
     *   </p>
     *   <p>The supported cache configurations are:
     *   <ul>
     *     <li>
     *       <p>cudaFuncCachePreferNone: no
     *         preference for shared memory or L1 (default)
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaFuncCachePreferShared:
     *         prefer larger shared memory and smaller L1 cache
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaFuncCachePreferL1: prefer
     *         larger L1 cache and smaller shared memory
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param cacheConfig Requested cache configuration
     *
     * @return cudaSuccess, cudaErrorInitializationError
     *
     * @see JCuda#cudaDeviceGetCacheConfig
     * @see JCuda#cudaDeviceSetCacheConfig
     * @see JCuda#cudaDeviceSetCacheConfig
     */
    public static int cudaDeviceSetCacheConfig(int cacheConfig)
    {
        return checkResult(cudaDeviceSetCacheConfigNative(cacheConfig));
    }
    private static native int cudaDeviceSetCacheConfigNative(int cacheConfig);






    /**
     * Returns a handle to a compute device.
     *
     * <pre>
     * cudaError_t cudaDeviceGetByPCIBusId (
     *      int* device,
     *      char* pciBusId )
     * </pre>
     * <div>
     *   <p>Returns a handle to a compute device.
     *     Returns in <tt>*device</tt> a device ordinal given a PCI bus ID
     *     string.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param device Returned device ordinal
     * @param pciBusId String in one of the following forms: [domain]:[bus]:[device].[function] [domain]:[bus]:[device] [bus]:[device].[function] where domain, bus, device, and function are all hexadecimal values
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice
     *
     * @see JCuda#cudaDeviceGetPCIBusId
     */
    public static int cudaDeviceGetByPCIBusId(int device[], String pciBusId)
    {
        return checkResult(cudaDeviceGetByPCIBusIdNative(device, pciBusId));
    }
    private static native int cudaDeviceGetByPCIBusIdNative(int device[], String pciBusId);

    /**
     * Returns a PCI Bus Id string for the device.
     *
     * <pre>
     * cudaError_t cudaDeviceGetPCIBusId (
     *      char* pciBusId,
     *      int  len,
     *      int  device )
     * </pre>
     * <div>
     *   <p>Returns a PCI Bus Id string for the
     *     device.  Returns an ASCII string identifying the device <tt>dev</tt>
     *     in the NULL-terminated string pointed to by <tt>pciBusId</tt>. <tt>len</tt> specifies the maximum length of the string that may be
     *     returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pciBusId Returned identifier string for the device in the following format [domain]:[bus]:[device].[function] where domain, bus, device, and function are all hexadecimal values. pciBusId should be large enough to store 13 characters including the NULL-terminator.
     * @param len Maximum length of string to store in name
     * @param device Device to get identifier string for
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice
     *
     * @see JCuda#cudaDeviceGetByPCIBusId
     */
    public static int cudaDeviceGetPCIBusId(String pciBusId[], int len, int device)
    {
        return checkResult(cudaDeviceGetPCIBusIdNative(pciBusId, len, device));
    }
    private static native int cudaDeviceGetPCIBusIdNative(String pciBusId[], int len, int device);

    /**
     * Gets an interprocess handle for a previously allocated event.
     *
     * <pre>
     * cudaError_t cudaIpcGetEventHandle (
     *      cudaIpcEventHandle_t* handle,
     *      cudaEvent_t event )
     * </pre>
     * <div>
     *   <p>Gets an interprocess handle for a
     *     previously allocated event.  Takes as input a previously allocated
     *     event. This event must
     *     have been created with the
     *     cudaEventInterprocess and cudaEventDisableTiming flags set. This opaque
     *     handle may be copied into other processes and opened with
     *     cudaIpcOpenEventHandle to allow efficient hardware synchronization
     *     between GPU work in different processes.
     *   </p>
     *   <p>After the event has been been opened in
     *     the importing process, cudaEventRecord, cudaEventSynchronize,
     *     cudaStreamWaitEvent and cudaEventQuery may be used in either process.
     *     Performing operations on the imported event after the exported event
     *     has been freed with cudaEventDestroy will result in undefined
     *     behavior.
     *   </p>
     *   <p>IPC functionality is restricted to
     *     devices with support for unified addressing on Linux operating
     *     systems.
     *   </p>
     * </div>
     *
     * @param handle Pointer to a user allocated cudaIpcEventHandle in which to return the opaque event handle
     * @param event Event allocated with cudaEventInterprocess and cudaEventDisableTiming flags.
     *
     * @return cudaSuccess, cudaErrorInvalidResourceHandle,
     * cudaErrorMemoryAllocation, cudaErrorMapBufferObjectFailed
     *
     * @see JCuda#cudaEventCreate
     * @see JCuda#cudaEventDestroy
     * @see JCuda#cudaEventSynchronize
     * @see JCuda#cudaEventQuery
     * @see JCuda#cudaStreamWaitEvent
     * @see JCuda#cudaIpcOpenEventHandle
     * @see JCuda#cudaIpcGetMemHandle
     * @see JCuda#cudaIpcOpenMemHandle
     * @see JCuda#cudaIpcCloseMemHandle
     */
    public static int cudaIpcGetEventHandle(cudaIpcEventHandle handle, cudaEvent_t event)
    {
        return checkResult(cudaIpcGetEventHandleNative(handle, event));
    }
    private static native int cudaIpcGetEventHandleNative(cudaIpcEventHandle handle, cudaEvent_t event);

    /**
     * Opens an interprocess event handle for use in the current process.
     *
     * <pre>
     * cudaError_t cudaIpcOpenEventHandle (
     *      cudaEvent_t* event,
     *      cudaIpcEventHandle_t handle )
     * </pre>
     * <div>
     *   <p>Opens an interprocess event handle for
     *     use in the current process.  Opens an interprocess event handle exported
     *     from another
     *     process with cudaIpcGetEventHandle. This
     *     function returns a cudaEvent_t that behaves like a locally created
     *     event with the cudaEventDisableTiming flag specified. This event must
     *     be freed with cudaEventDestroy.
     *   </p>
     *   <p>Performing operations on the imported
     *     event after the exported event has been freed with cudaEventDestroy
     *     will result in undefined behavior.
     *   </p>
     *   <p>IPC functionality is restricted to
     *     devices with support for unified addressing on Linux operating
     *     systems.
     *   </p>
     * </div>
     *
     * @param event Returns the imported event
     * @param handle Interprocess handle to open
     *
     * @return cudaSuccess, cudaErrorMapBufferObjectFailed,
     * cudaErrorInvalidResourceHandle
     *
     * @see JCuda#cudaEventCreate
     * @see JCuda#cudaEventDestroy
     * @see JCuda#cudaEventSynchronize
     * @see JCuda#cudaEventQuery
     * @see JCuda#cudaStreamWaitEvent
     * @see JCuda#cudaIpcGetEventHandle
     * @see JCuda#cudaIpcGetMemHandle
     * @see JCuda#cudaIpcOpenMemHandle
     * @see JCuda#cudaIpcCloseMemHandle
     */
    public static int cudaIpcOpenEventHandle(cudaEvent_t event, cudaIpcEventHandle handle)
    {
        return checkResult(cudaIpcOpenEventHandleNative(event, handle));
    }
    private static native int cudaIpcOpenEventHandleNative(cudaEvent_t event, cudaIpcEventHandle handle);

    /**
     *
     * <pre>
     * cudaError_t cudaIpcGetMemHandle (
     *      cudaIpcMemHandle_t* handle,
     *      void* devPtr )
     * </pre>
     * <div>
     *   <p> /brief Gets an interprocess memory
     *     handle for an existing device memory allocation
     *   </p>
     *   <p>Takes a pointer to the base of an
     *     existing device memory allocation created with cudaMalloc and exports
     *     it for use in another process. This is a lightweight operation and may
     *     be called multiple times on an allocation
     *     without adverse effects.
     *   </p>
     *   <p>If a region of memory is freed with
     *     cudaFree and a subsequent call to cudaMalloc returns memory with the
     *     same device address, cudaIpcGetMemHandle will return a unique handle
     *     for the new memory.
     *   </p>
     *   <p>IPC functionality is restricted to
     *     devices with support for unified addressing on Linux operating
     *     systems.
     *   </p>
     * </div>
     *
     * @param handle Pointer to user allocated cudaIpcMemHandle to return the handle in.
     * @param devPtr Base pointer to previously allocated device memory
     *
     * @return cudaSuccess, cudaErrorInvalidResourceHandle,
     * cudaErrorMemoryAllocation, cudaErrorMapBufferObjectFailed,
     *
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaFree
     * @see JCuda#cudaIpcGetEventHandle
     * @see JCuda#cudaIpcOpenEventHandle
     * @see JCuda#cudaIpcOpenMemHandle
     * @see JCuda#cudaIpcCloseMemHandle
     */
    public static int cudaIpcGetMemHandle(cudaIpcMemHandle handle, Pointer devPtr)
    {
        return checkResult(cudaIpcGetMemHandleNative(handle, devPtr));
    }
    private static native int cudaIpcGetMemHandleNative(cudaIpcMemHandle handle, Pointer devPtr);

    /**
     *
     * <pre>
     * cudaError_t cudaIpcOpenMemHandle (
     *      void** devPtr,
     *      cudaIpcMemHandle_t handle,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p> /brief Opens an interprocess memory
     *     handle exported from another process and returns a device pointer
     *     usable in the local
     *     process.
     *   </p>
     *   <p>Maps memory exported from another
     *     process with cudaIpcGetMemHandle into the current device address space.
     *     For contexts on different devices cudaIpcOpenMemHandle can attempt to
     *     enable peer access between the devices as if the user called
     *     cudaDeviceEnablePeerAccess. This behavior is controlled by the
     *     cudaIpcMemLazyEnablePeerAccess flag. cudaDeviceCanAccessPeer can
     *     determine if a mapping is possible.
     *   </p>
     *   <p>Contexts that may open cudaIpcMemHandles
     *     are restricted in the following way. cudaIpcMemHandles from each device
     *     in a given
     *     process may only be opened by one context
     *     per device per other process.
     *   </p>
     *   <p>Memory returned from cudaIpcOpenMemHandle
     *     must be freed with cudaIpcCloseMemHandle.
     *   </p>
     *   <p>Calling cudaFree on an exported memory
     *     region before calling cudaIpcCloseMemHandle in the importing context
     *     will result in undefined behavior.
     *   </p>
     *   <p>IPC functionality is restricted to
     *     devices with support for unified addressing on Linux operating
     *     systems.
     *   </p>
     * </div>
     *
     * @param devPtr Returned device pointer
     * @param handle cudaIpcMemHandle to open
     * @param flags Flags for this operation. Must be specified as cudaIpcMemLazyEnablePeerAccess
     *
     * @return cudaSuccess, cudaErrorMapBufferObjectFailed,
     * cudaErrorInvalidResourceHandle, cudaErrorTooManyPeers
     *
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaFree
     * @see JCuda#cudaIpcGetEventHandle
     * @see JCuda#cudaIpcOpenEventHandle
     * @see JCuda#cudaIpcGetMemHandle
     * @see JCuda#cudaIpcCloseMemHandle
     * @see JCuda#cudaDeviceEnablePeerAccess
     * @see JCuda#cudaDeviceCanAccessPeer
     */
    public static int cudaIpcOpenMemHandle(Pointer devPtr, cudaIpcMemHandle handle, int flags)
    {
        return checkResult(cudaIpcOpenMemHandleNative(devPtr, handle, flags));
    }
    private static native int cudaIpcOpenMemHandleNative(Pointer devPtr, cudaIpcMemHandle handle, int flags);

    /**
     * Close memory mapped with cudaIpcOpenMemHandle.
     *
     * <pre>
     * cudaError_t cudaIpcCloseMemHandle (
     *      void* devPtr )
     * </pre>
     * <div>
     *   <p>Close memory mapped with
     *     cudaIpcOpenMemHandle.  Unmaps memory returnd by cudaIpcOpenMemHandle.
     *     The original allocation in the exporting process as well as imported
     *     mappings in other processes will be unaffected.
     *   </p>
     *   <p>Any resources used to enable peer access
     *     will be freed if this is the last mapping using them.
     *   </p>
     *   <p>IPC functionality is restricted to
     *     devices with support for unified addressing on Linux operating
     *     systems.
     *   </p>
     * </div>
     *
     * @param devPtr Device pointer returned by cudaIpcOpenMemHandle
     *
     * @return cudaSuccess, cudaErrorMapBufferObjectFailed,
     * cudaErrorInvalidResourceHandle,
     *
     * @see JCuda#cudaMalloc
     * @see JCuda#cudaFree
     * @see JCuda#cudaIpcGetEventHandle
     * @see JCuda#cudaIpcOpenEventHandle
     * @see JCuda#cudaIpcGetMemHandle
     * @see JCuda#cudaIpcOpenMemHandle
     */
    public static int cudaIpcCloseMemHandle(Pointer devPtr)
    {
        return checkResult(cudaIpcCloseMemHandleNative(devPtr));
    }
    private static native int cudaIpcCloseMemHandleNative(Pointer devPtr);





    /**
     * Exit and clean up from CUDA launches.
     *
     * <pre>
     * cudaError_t cudaThreadExit (
     *      void )
     * </pre>
     * <div>
     *   <p>Exit and clean up from CUDA launches.
     *     Deprecated Note that this function is
     *     deprecated because its name does not reflect its behavior. Its
     *     functionality is identical to the
     *     non-deprecated function cudaDeviceReset(),
     *     which should be used instead.
     *   </p>
     *   <p>Explicitly destroys all cleans up all
     *     resources associated with the current device in the current process.
     *     Any subsequent
     *     API call to this device will reinitialize
     *     the device.
     *   </p>
     *   <p>Note that this function will reset the
     *     device immediately. It is the caller's responsibility to ensure that
     *     the device is
     *     not being accessed by any other host
     *     threads from the process when this function is called.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return cudaSuccess
     *
     * @see JCuda#cudaDeviceReset
     * 
     * @deprecated Deprecated in CUDA
     */
    public static int cudaThreadExit()
    {
        return checkResult(cudaThreadExitNative());
    }
    private static native int cudaThreadExitNative();



    /**
     * Wait for compute device to finish.
     *
     * <pre>
     * cudaError_t cudaThreadSynchronize (
     *      void )
     * </pre>
     * <div>
     *   <p>Wait for compute device to finish.
     *     Deprecated Note that this function is
     *     deprecated because its name does not reflect its behavior. Its
     *     functionality is similar to the
     *     non-deprecated function
     *     cudaDeviceSynchronize(), which should be used instead.
     *   </p>
     *   <p>Blocks until the device has completed
     *     all preceding requested tasks. cudaThreadSynchronize() returns an error
     *     if one of the preceding tasks has failed. If the
     *     cudaDeviceScheduleBlockingSync flag was set for this device, the host
     *     thread will block until the device has finished its work.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return cudaSuccess
     *
     * @see JCuda#cudaDeviceSynchronize
     * 
     * @deprecated Deprecated in CUDA
     */
    public static int cudaThreadSynchronize()
    {
        return checkResult(cudaThreadSynchronizeNative());
    }
    private static native int cudaThreadSynchronizeNative();

    /**
     * Set resource limits.
     *
     * <pre>
     * cudaError_t cudaThreadSetLimit (
     *      cudaLimit limit,
     *      size_t value )
     * </pre>
     * <div>
     *   <p>Set resource limits.
     *     Deprecated Note that this function is
     *     deprecated because its name does not reflect its behavior. Its
     *     functionality is identical to the
     *     non-deprecated function cudaDeviceSetLimit(),
     *     which should be used instead.
     *   </p>
     *   <p>Setting <tt>limit</tt> to <tt>value</tt> is a request by the application to update the current limit
     *     maintained by the device. The driver is free to modify the requested
     *     value to meet h/w requirements (this
     *     could be clamping to minimum or maximum values, rounding up to nearest
     *     element size,
     *     etc). The application can use
     *     cudaThreadGetLimit() to find out exactly what the limit has been set
     *     to.
     *   </p>
     *   <p>Setting each cudaLimit has its own
     *     specific restrictions, so each is discussed here.
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaLimitStackSize controls
     *         the stack size of each GPU thread. This limit is only applicable to
     *         devices of compute capability 2.0 and higher.
     *         Attempting to set this limit on
     *         devices of compute capability less than 2.0 will result in the error
     *         cudaErrorUnsupportedLimit being returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaLimitPrintfFifoSize controls
     *         the size of the shared FIFO used by the printf() and fprintf() device
     *         system calls. Setting cudaLimitPrintfFifoSize must be performed before
     *         launching any kernel that uses the printf() or fprintf() device system
     *         calls, otherwise cudaErrorInvalidValue will be returned. This limit is
     *         only applicable to devices of compute capability 2.0 and higher.
     *         Attempting to set this limit
     *         on devices of compute capability
     *         less than 2.0 will result in the error cudaErrorUnsupportedLimit being
     *         returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaLimitMallocHeapSize controls
     *         the size of the heap used by the malloc() and free() device system
     *         calls. Setting cudaLimitMallocHeapSize must be performed before
     *         launching any kernel that uses the malloc() or free() device system
     *         calls, otherwise cudaErrorInvalidValue will be returned. This limit is
     *         only applicable to devices of compute capability 2.0 and higher.
     *         Attempting to set this limit
     *         on devices of compute capability
     *         less than 2.0 will result in the error cudaErrorUnsupportedLimit being
     *         returned.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param limit Limit to set
     * @param value Size in bytes of limit
     *
     * @return cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue
     *
     * @see JCuda#cudaDeviceSetLimit
     * 
     * @deprecated Deprecated in CUDA
     */
    public static int cudaThreadSetLimit(int limit, long value)
    {
        return checkResult(cudaThreadSetLimitNative(limit, value));
    }
    private static native int cudaThreadSetLimitNative(int limit, long value);


    /**
     * Returns the preferred cache configuration for the current device.
     *
     * <pre>
     * cudaError_t cudaThreadGetCacheConfig (
     *      cudaFuncCache ** pCacheConfig )
     * </pre>
     * <div>
     *   <p>Returns the preferred cache configuration
     *     for the current device.
     *     Deprecated Note that this function is
     *     deprecated because its name does not reflect its behavior. Its
     *     functionality is identical to the
     *     non-deprecated function
     *     cudaDeviceGetCacheConfig(), which should be used instead.
     *   </p>
     *   <p>On devices where the L1 cache and shared
     *     memory use the same hardware resources, this returns through <tt>pCacheConfig</tt> the preferred cache configuration for the current
     *     device. This is only a preference. The runtime will use the requested
     *     configuration
     *     if possible, but it is free to choose a
     *     different configuration if required to execute functions.
     *   </p>
     *   <p>This will return a <tt>pCacheConfig</tt>
     *     of cudaFuncCachePreferNone on devices where the size of the L1 cache
     *     and shared memory are fixed.
     *   </p>
     *   <p>The supported cache configurations are:
     *   <ul>
     *     <li>
     *       <p>cudaFuncCachePreferNone: no
     *         preference for shared memory or L1 (default)
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaFuncCachePreferShared:
     *         prefer larger shared memory and smaller L1 cache
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaFuncCachePreferL1: prefer
     *         larger L1 cache and smaller shared memory
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCacheConfig Returned cache configuration
     *
     * @return cudaSuccess, cudaErrorInitializationError
     *
     * @see JCuda#cudaDeviceGetCacheConfig
     * 
     * @deprecated Deprecated in CUDA
     */
    public static int cudaThreadGetCacheConfig(int pCacheConfig[])
    {
        return checkResult(cudaThreadGetCacheConfigNative(pCacheConfig));
    }

    private static native int cudaThreadGetCacheConfigNative(int[] pCacheConfig);

    /**
     * Sets the preferred cache configuration for the current device.
     *
     * <pre>
     * cudaError_t cudaThreadSetCacheConfig (
     *      cudaFuncCache cacheConfig )
     * </pre>
     * <div>
     *   <p>Sets the preferred cache configuration
     *     for the current device.
     *     Deprecated Note that this function is
     *     deprecated because its name does not reflect its behavior. Its
     *     functionality is identical to the
     *     non-deprecated function
     *     cudaDeviceSetCacheConfig(), which should be used instead.
     *   </p>
     *   <p>On devices where the L1 cache and shared
     *     memory use the same hardware resources, this sets through <tt>cacheConfig</tt> the preferred cache configuration for the current
     *     device. This is only a preference. The runtime will use the requested
     *     configuration
     *     if possible, but it is free to choose a
     *     different configuration if required to execute the function. Any
     *     function preference
     *     set via cudaDeviceSetCacheConfig ( C API)
     *     or cudaDeviceSetCacheConfig ( C++ API) will be preferred over this
     *     device-wide setting. Setting the device-wide cache configuration to
     *     cudaFuncCachePreferNone will cause subsequent kernel launches to prefer
     *     to not change the cache configuration unless required to launch the
     *     kernel.
     *   </p>
     *   <p>This setting does nothing on devices
     *     where the size of the L1 cache and shared memory are fixed.
     *   </p>
     *   <p>Launching a kernel with a different
     *     preference than the most recent preference setting may insert a
     *     device-side synchronization
     *     point.
     *   </p>
     *   <p>The supported cache configurations are:
     *   <ul>
     *     <li>
     *       <p>cudaFuncCachePreferNone: no
     *         preference for shared memory or L1 (default)
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaFuncCachePreferShared:
     *         prefer larger shared memory and smaller L1 cache
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaFuncCachePreferL1: prefer
     *         larger L1 cache and smaller shared memory
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param cacheConfig Requested cache configuration
     *
     * @return cudaSuccess, cudaErrorInitializationError
     *
     * @see JCuda#cudaDeviceSetCacheConfig
     * 
     * @deprecated Deprecated in CUDA
     */
    public static int cudaThreadSetCacheConfig(int cacheConfig)
    {
        return checkResult(cudaThreadSetCacheConfigNative(cacheConfig));
    }

    private static native int cudaThreadSetCacheConfigNative(int cacheConfig);



    /**
     * Returns resource limits.
     *
     * <pre>
     * cudaError_t cudaThreadGetLimit (
     *      size_t* pValue,
     *      cudaLimit limit )
     * </pre>
     * <div>
     *   <p>Returns resource limits.
     *     Deprecated Note that this function is
     *     deprecated because its name does not reflect its behavior. Its
     *     functionality is identical to the
     *     non-deprecated function cudaDeviceGetLimit(),
     *     which should be used instead.
     *   </p>
     *   <p>Returns in <tt>*pValue</tt> the current
     *     size of <tt>limit</tt>. The supported cudaLimit values are:
     *   <ul>
     *     <li>
     *       <p>cudaLimitStackSize: stack size
     *         of each GPU thread;
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaLimitPrintfFifoSize: size
     *         of the shared FIFO used by the printf() and fprintf() device system
     *         calls.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaLimitMallocHeapSize: size
     *         of the heap used by the malloc() and free() device system calls;
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pValue Returned size in bytes of limit
     * @param limit Limit to query
     *
     * @return cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue
     *
     * @see JCuda#cudaDeviceGetLimit
     * 
     * @deprecated Deprecated in CUDA
     */
    public static int cudaThreadGetLimit(long pValue[], int limit)
    {
        return checkResult(cudaThreadGetLimitNative(pValue, limit));
    }
    private static native int cudaThreadGetLimitNative(long pValue[], int limit);


    /**
     * [C++ API] Finds the address associated with a CUDA symbol
     *
     * <pre>
     * template < class T > cudaError_t cudaGetSymbolAddress (
     *      void** devPtr,
     *      const T& symbol ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Finds the address associated
     *     with a CUDA symbol  Returns in <tt>*devPtr</tt> the address of symbol
     *     <tt>symbol</tt> on the device. <tt>symbol</tt> can either be a
     *     variable that resides in global or constant memory space. If <tt>symbol</tt> cannot be found, or if <tt>symbol</tt> is not declared
     *     in the global or constant memory space, <tt>*devPtr</tt> is unchanged
     *     and the error cudaErrorInvalidSymbol is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param devPtr Return device pointer associated with symbol
     * @param symbol Device symbol address
     * @param devPtr Return device pointer associated with symbol
     * @param symbol Device symbol reference
     *
     * @return cudaSuccess, cudaErrorInvalidSymbol
     *
     * @see JCuda#cudaGetSymbolAddress
     * @see JCuda#cudaGetSymbolSize
     */
    public static int cudaGetSymbolAddress(Pointer devPtr, String symbol)
    {
        if (true)
        {
            throw new UnsupportedOperationException(
                "This function is no longer supported as of CUDA 5.0");
        }
        return checkResult(cudaGetSymbolAddressNative(devPtr, symbol));
    }
    private static native int cudaGetSymbolAddressNative(Pointer devPtr, String symbol);

    /**
     * [C++ API] Finds the size of the object associated with a CUDA symbol
     *
     * <pre>
     * template < class T > cudaError_t cudaGetSymbolSize (
     *      size_t* size,
     *      const T& symbol ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Finds the size of the object
     *     associated with a CUDA symbol  Returns in <tt>*size</tt> the size of
     *     symbol <tt>symbol</tt>. <tt>symbol</tt> must be a variable that
     *     resides in global or constant memory space. If <tt>symbol</tt> cannot
     *     be found, or if <tt>symbol</tt> is not declared in global or constant
     *     memory space, <tt>*size</tt> is unchanged and the error
     *     cudaErrorInvalidSymbol is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param size Size of object associated with symbol
     * @param symbol Device symbol address
     * @param size Size of object associated with symbol
     * @param symbol Device symbol reference
     *
     * @return cudaSuccess, cudaErrorInvalidSymbol
     *
     * @see JCuda#cudaGetSymbolAddress
     * @see JCuda#cudaGetSymbolSize
     */
    public static int cudaGetSymbolSize(long size[], String symbol)
    {
        if (true)
        {
            throw new UnsupportedOperationException(
                "This function is no longer supported as of CUDA 5.0");
        }
        return checkResult(cudaGetSymbolSizeNative(size, symbol));
    }
    private static native int cudaGetSymbolSizeNative(long size[], String symbol);


    /**
     * [C++ API] Binds a memory area to a texture
     *
     * <pre>
     * template < class T, int dim, enum cudaTextureReadMode readMode >
     * cudaError_t cudaBindTexture (
     *      size_t* offset,
     *      const texture < T,
     *      dim,
     *      readMode > & tex,
     *      const void* devPtr,
     *      const cudaChannelFormatDesc& desc,
     *      size_t size = UINT_MAX ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Binds a memory area to a
     *     texture  Binds <tt>size</tt> bytes of the memory area pointed to by
     *     <tt>devPtr</tt> to texture reference <tt>tex</tt>. <tt>desc</tt>
     *     describes how the memory is interpreted when fetching values from the
     *     texture. The <tt>offset</tt> parameter is an optional byte offset as
     *     with the low-level cudaBindTexture() function. Any memory previously
     *     bound to <tt>tex</tt> is unbound.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param offset Offset in bytes
     * @param texref Texture to bind
     * @param devPtr Memory area on device
     * @param desc Channel format
     * @param size Size of the memory area pointed to by devPtr
     * @param offset Offset in bytes
     * @param tex Texture to bind
     * @param devPtr Memory area on device
     * @param size Size of the memory area pointed to by devPtr
     * @param offset Offset in bytes
     * @param tex Texture to bind
     * @param devPtr Memory area on device
     * @param desc Channel format
     * @param size Size of the memory area pointed to by devPtr
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidTexture
     *
     * @see JCuda#cudaCreateChannelDesc
     * @see JCuda#cudaGetChannelDesc
     * @see JCuda#cudaGetTextureReference
     * @see JCuda#cudaBindTexture
     * @see JCuda#cudaBindTexture
     *
     * @see JCuda#cudaBindTexture2D
     * @see JCuda#cudaBindTexture2D
     *
     * @see JCuda#cudaBindTextureToArray
     * @see JCuda#cudaBindTextureToArray
     *
     * @see JCuda#cudaUnbindTexture
     * @see JCuda#cudaGetTextureAlignmentOffset
     */
    public static int cudaBindTexture(long offset[], textureReference texref, Pointer devPtr, cudaChannelFormatDesc desc, long size)
    {
        return checkResult(cudaBindTextureNative(offset, texref, devPtr, desc, size));
    }
    private static native int cudaBindTextureNative(long offset[], textureReference texref, Pointer devPtr, cudaChannelFormatDesc desc, long size);



    /**
     * [C++ API] Binds a 2D memory area to a texture
     *
     * <pre>
     * template < class T, int dim, enum cudaTextureReadMode readMode >
     * cudaError_t cudaBindTexture2D (
     *      size_t* offset,
     *      const texture < T,
     *      dim,
     *      readMode > & tex,
     *      const void* devPtr,
     *      const cudaChannelFormatDesc& desc,
     *      size_t width,
     *      size_t height,
     *      size_t pitch ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Binds a 2D memory area to a
     *     texture  Binds the 2D memory area pointed to by <tt>devPtr</tt> to
     *     the texture reference <tt>tex</tt>. The size of the area is constrained
     *     by <tt>width</tt> in texel units, <tt>height</tt> in texel units,
     *     and <tt>pitch</tt> in byte units. <tt>desc</tt> describes how the
     *     memory is interpreted when fetching values from the texture. Any memory
     *     previously bound to <tt>tex</tt> is unbound.
     *   </p>
     *   <p>Since the hardware enforces an alignment
     *     requirement on texture base addresses, cudaBindTexture2D() returns in
     *     <tt>*offset</tt> a byte offset that must be applied to texture fetches
     *     in order to read from the desired memory. This offset must be divided
     *     by the texel size and passed to kernels
     *     that read from the texture so they can be applied to the tex2D()
     *     function. If the
     *     device memory pointer was returned from
     *     cudaMalloc(), the offset is guaranteed to be 0 and NULL may be passed
     *     as the <tt>offset</tt> parameter.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param offset Offset in bytes
     * @param texref Texture reference to bind
     * @param devPtr 2D memory area on device
     * @param desc Channel format
     * @param width Width in texel units
     * @param height Height in texel units
     * @param pitch Pitch in bytes
     * @param offset Offset in bytes
     * @param tex Texture reference to bind
     * @param devPtr 2D memory area on device
     * @param width Width in texel units
     * @param height Height in texel units
     * @param pitch Pitch in bytes
     * @param offset Offset in bytes
     * @param tex Texture reference to bind
     * @param devPtr 2D memory area on device
     * @param desc Channel format
     * @param width Width in texel units
     * @param height Height in texel units
     * @param pitch Pitch in bytes
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidTexture
     *
     * @see JCuda#cudaCreateChannelDesc
     * @see JCuda#cudaGetChannelDesc
     * @see JCuda#cudaGetTextureReference
     * @see JCuda#cudaBindTexture
     * @see JCuda#cudaBindTexture
     *
     * @see JCuda#cudaBindTexture2D
     * @see JCuda#cudaBindTexture2D
     *
     * @see JCuda#cudaBindTextureToArray
     * @see JCuda#cudaBindTextureToArray
     *
     * @see JCuda#cudaUnbindTexture
     * @see JCuda#cudaGetTextureAlignmentOffset
     */
    public static int cudaBindTexture2D (long offset[], textureReference texref, Pointer devPtr, cudaChannelFormatDesc desc, long width, long height, long pitch)
    {
        return checkResult(cudaBindTexture2DNative(offset, texref, devPtr, desc, width, height, pitch));
    }
    private static native int cudaBindTexture2DNative(long offset[], textureReference texref, Pointer devPtr, cudaChannelFormatDesc desc, long width, long height, long pitch);



    /**
     * [C++ API] Binds an array to a texture
     *
     * <pre>
     * template < class T, int dim, enum cudaTextureReadMode readMode >
     * cudaError_t cudaBindTextureToArray (
     *      const texture < T,
     *      dim,
     *      readMode > & tex,
     *      cudaArray_const_t array,
     *      const cudaChannelFormatDesc& desc ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Binds an array to a texture
     *     Binds the CUDA array <tt>array</tt> to the texture reference <tt>tex</tt>. <tt>desc</tt> describes how the memory is interpreted when
     *     fetching values from the texture. Any CUDA array previously bound to
     *     <tt>tex</tt> is unbound.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param texref Texture to bind
     * @param array Memory array on device
     * @param desc Channel format
     * @param tex Texture to bind
     * @param array Memory array on device
     * @param tex Texture to bind
     * @param array Memory array on device
     * @param desc Channel format
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidTexture
     *
     * @see JCuda#cudaCreateChannelDesc
     * @see JCuda#cudaGetChannelDesc
     * @see JCuda#cudaGetTextureReference
     * @see JCuda#cudaBindTexture
     * @see JCuda#cudaBindTexture
     *
     * @see JCuda#cudaBindTexture2D
     * @see JCuda#cudaBindTexture2D
     *
     * @see JCuda#cudaBindTextureToArray
     * @see JCuda#cudaBindTextureToArray
     *
     * @see JCuda#cudaUnbindTexture
     * @see JCuda#cudaGetTextureAlignmentOffset
     */
    public static int cudaBindTextureToArray(textureReference texref, cudaArray array, cudaChannelFormatDesc desc)
    {
        return checkResult(cudaBindTextureToArrayNative(texref, array, desc));
    }
    private static native int cudaBindTextureToArrayNative(textureReference texref, cudaArray array, cudaChannelFormatDesc desc);


    /**
     * [C++ API] Binds a mipmapped array to a texture
     *
     * <pre>
     * template < class T, int dim, enum cudaTextureReadMode readMode >
     * cudaError_t cudaBindTextureToMipmappedArray (
     *      const texture < T,
     *      dim,
     *      readMode > & tex,
     *      cudaMipmappedArray_const_t mipmappedArray,
     *      const cudaChannelFormatDesc& desc ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Binds a mipmapped array to a
     *     texture  Binds the CUDA mipmapped array <tt>mipmappedArray</tt> to
     *     the texture reference <tt>tex</tt>. <tt>desc</tt> describes how the
     *     memory is interpreted when fetching values from the texture. Any CUDA
     *     mipmapped array previously bound
     *     to <tt>tex</tt> is unbound.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param texref Texture to bind
     * @param mipmappedArray Memory mipmapped array on device
     * @param desc Channel format
     * @param tex Texture to bind
     * @param mipmappedArray Memory mipmapped array on device
     * @param tex Texture to bind
     * @param mipmappedArray Memory mipmapped array on device
     * @param desc Channel format
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevicePointer,
     * cudaErrorInvalidTexture
     *
     * @see JCuda#cudaCreateChannelDesc
     * @see JCuda#cudaGetChannelDesc
     * @see JCuda#cudaGetTextureReference
     * @see JCuda#cudaBindTexture
     * @see JCuda#cudaBindTexture
     *
     * @see JCuda#cudaBindTexture2D
     * @see JCuda#cudaBindTexture2D
     *
     * @see JCuda#cudaBindTextureToArray
     * @see JCuda#cudaBindTextureToArray
     *
     * @see JCuda#cudaUnbindTexture
     * @see JCuda#cudaGetTextureAlignmentOffset
     */
    public static int cudaBindTextureToMipmappedArray(textureReference texref, cudaMipmappedArray mipmappedArray, cudaChannelFormatDesc desc)
    {
        return checkResult(cudaBindTextureToMipmappedArrayNative(texref, mipmappedArray, desc));
    }
    private static native int cudaBindTextureToMipmappedArrayNative(textureReference texref, cudaMipmappedArray mipmappedArray, cudaChannelFormatDesc desc);

    /**
     * [C++ API] Unbinds a texture
     *
     * <pre>
     * template < class T, int dim, enum cudaTextureReadMode readMode >
     * cudaError_t cudaUnbindTexture (
     *      const texture < T,
     *      dim,
     *      readMode > & tex ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Unbinds a texture  Unbinds the
     *     texture bound to <tt>tex</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param texref Texture to unbind
     * @param tex Texture to unbind
     *
     * @return cudaSuccess
     *
     * @see JCuda#cudaCreateChannelDesc
     * @see JCuda#cudaGetChannelDesc
     * @see JCuda#cudaGetTextureReference
     * @see JCuda#cudaBindTexture
     * @see JCuda#cudaBindTexture
     *
     * @see JCuda#cudaBindTexture2D
     * @see JCuda#cudaBindTexture2D
     *
     * @see JCuda#cudaBindTextureToArray
     * @see JCuda#cudaBindTextureToArray
     *
     * @see JCuda#cudaUnbindTexture
     * @see JCuda#cudaGetTextureAlignmentOffset
     */
    public static int cudaUnbindTexture(textureReference texref)
    {
        return checkResult(cudaUnbindTextureNative(texref));
    }
    private static native int cudaUnbindTextureNative(textureReference texref);

    /**
     * [C++ API] Get the alignment offset of a texture
     *
     * <pre>
     * template < class T, int dim, enum cudaTextureReadMode readMode >
     * cudaError_t cudaGetTextureAlignmentOffset (
     *      size_t* offset,
     *      const texture < T,
     *      dim,
     *      readMode > & tex ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Get the alignment offset of a
     *     texture  Returns in <tt>*offset</tt> the offset that was returned when
     *     texture reference <tt>tex</tt> was bound.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param offset Offset of texture reference in bytes
     * @param texref Texture to get offset of
     * @param offset Offset of texture reference in bytes
     * @param tex Texture to get offset of
     *
     * @return cudaSuccess, cudaErrorInvalidTexture,
     * cudaErrorInvalidTextureBinding
     *
     * @see JCuda#cudaCreateChannelDesc
     * @see JCuda#cudaGetChannelDesc
     * @see JCuda#cudaGetTextureReference
     * @see JCuda#cudaBindTexture
     * @see JCuda#cudaBindTexture
     *
     * @see JCuda#cudaBindTexture2D
     * @see JCuda#cudaBindTexture2D
     *
     * @see JCuda#cudaBindTextureToArray
     * @see JCuda#cudaBindTextureToArray
     *
     * @see JCuda#cudaUnbindTexture
     * @see JCuda#cudaGetTextureAlignmentOffset
     */
    public static int cudaGetTextureAlignmentOffset(long offset[], textureReference texref)
    {
        return checkResult(cudaGetTextureAlignmentOffsetNative(offset, texref));
    }
    private static native int cudaGetTextureAlignmentOffsetNative(long offset[], textureReference texref);

    /**
     * Get the texture reference associated with a symbol.
     *
     * <pre>
     * cudaError_t cudaGetTextureReference (
     *      const textureReference** texref,
     *      const void* symbol )
     * </pre>
     * <div>
     *   <p>Get the texture reference associated with
     *     a symbol.  Returns in <tt>*texref</tt> the structure associated to
     *     the texture reference defined by symbol <tt>symbol</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>Use of a string naming a
     *           variable as the <tt>symbol</tt> paramater was removed in CUDA 5.0.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param texref Texture reference associated with symbol
     * @param symbol Texture to get reference for
     *
     * @return cudaSuccess, cudaErrorInvalidTexture
     *
     * @see JCuda#cudaCreateChannelDesc
     * @see JCuda#cudaGetChannelDesc
     * @see JCuda#cudaGetTextureAlignmentOffset
     * @see JCuda#cudaBindTexture
     * @see JCuda#cudaBindTexture2D
     * @see JCuda#cudaBindTextureToArray
     * @see JCuda#cudaUnbindTexture
     */
    public static int cudaGetTextureReference(textureReference texref, String symbol)
    {
        if (true)
        {
            throw new UnsupportedOperationException(
                "This function is no longer supported as of CUDA 5.0");
        }
        return checkResult(cudaGetTextureReferenceNative(texref, symbol));
    }
    private static native int cudaGetTextureReferenceNative(textureReference texref, String symbol);


    /**
     * [C++ API] Binds an array to a surface
     *
     * <pre>
     * template < class T, int dim > cudaError_t cudaBindSurfaceToArray (
     *      const surface < T,
     *      dim > & surf,
     *      cudaArray_const_t array,
     *      const cudaChannelFormatDesc& desc ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Binds an array to a surface
     *     Binds the CUDA array <tt>array</tt> to the surface reference <tt>surf</tt>. <tt>desc</tt> describes how the memory is interpreted when
     *     dealing with the surface. Any CUDA array previously bound to <tt>surf</tt> is unbound.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param surfref Surface to bind
     * @param array Memory array on device
     * @param desc Channel format
     * @param surf Surface to bind
     * @param array Memory array on device
     * @param surf Surface to bind
     * @param array Memory array on device
     * @param desc Channel format
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidSurface
     *
     * @see JCuda#cudaBindSurfaceToArray
     * @see JCuda#cudaBindSurfaceToArray
     *
     */
    public static int cudaBindSurfaceToArray(surfaceReference surfref, cudaArray array, cudaChannelFormatDesc desc)
    {
        return checkResult(cudaBindSurfaceToArrayNative(surfref, array, desc));
    }
    private static native int cudaBindSurfaceToArrayNative(surfaceReference surfref, cudaArray array, cudaChannelFormatDesc desc);


    /**
     * Get the surface reference associated with a symbol.
     *
     * <pre>
     * cudaError_t cudaGetSurfaceReference (
     *      const surfaceReference** surfref,
     *      const void* symbol )
     * </pre>
     * <div>
     *   <p>Get the surface reference associated with
     *     a symbol.  Returns in <tt>*surfref</tt> the structure associated to
     *     the surface reference defined by symbol <tt>symbol</tt>.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <ul>
     *       <li>
     *         <p>Note that this function may
     *           also return error codes from previous, asynchronous launches.
     *         </p>
     *       </li>
     *       <li>
     *         <p>Use of a string naming a
     *           variable as the <tt>symbol</tt> paramater was removed in CUDA 5.0.
     *         </p>
     *       </li>
     *     </ul>
     *   </div>
     *   </p>
     * </div>
     *
     * @param surfref Surface reference associated with symbol
     * @param symbol Surface to get reference for
     *
     * @return cudaSuccess, cudaErrorInvalidSurface
     *
     * @see JCuda#cudaBindSurfaceToArray
     */
    public static int cudaGetSurfaceReference(surfaceReference surfref, String symbol)
    {
        if (true)
        {
            throw new UnsupportedOperationException(
                "This function is no longer supported as of CUDA 5.0");
        }
        return checkResult(cudaGetSurfaceReferenceNative(surfref, symbol));
    }
    private static native int cudaGetSurfaceReferenceNative(surfaceReference surfref, String symbol);




    /**
     * Creates a texture object.
     *
     * <pre>
     * cudaError_t cudaCreateTextureObject (
     *      cudaTextureObject_t* pTexObject,
     *      const cudaResourceDesc* pResDesc,
     *      const cudaTextureDesc* pTexDesc,
     *      const cudaResourceViewDesc* pResViewDesc )
     * </pre>
     * <div>
     *   <p>Creates a texture object.  Creates a
     *     texture object and returns it in <tt>pTexObject</tt>. <tt>pResDesc</tt>
     *     describes the data to texture from. <tt>pTexDesc</tt> describes how
     *     the data should be sampled. <tt>pResViewDesc</tt> is an optional
     *     argument that specifies an alternate format for the data described by
     *     <tt>pResDesc</tt>, and also describes the subresource region to
     *     restrict access to when texturing. <tt>pResViewDesc</tt> can only be
     *     specified if the type of resource is a CUDA array or a CUDA mipmapped
     *     array.
     *   </p>
     *   <p>Texture objects are only supported on
     *     devices of compute capability 3.0 or higher.
     *   </p>
     *   <p>The cudaResourceDesc structure is
     *     defined as:
     *   <pre>        struct cudaResourceDesc {
     *             enum cudaResourceType
     *                   resType;
     *
     *             union {
     *                 struct {
     *                     cudaArray_t
     *                   array;
     *                 } array;
     *                 struct {
     *                     cudaMipmappedArray_t
     *                   mipmap;
     *                 } mipmap;
     *                 struct {
     *                     void *devPtr;
     *                     struct cudaChannelFormatDesc
     *                   desc;
     *                     size_t sizeInBytes;
     *                 } linear;
     *                 struct {
     *                     void *devPtr;
     *                     struct cudaChannelFormatDesc
     *                   desc;
     *                     size_t width;
     *                     size_t height;
     *                     size_t pitchInBytes;
     *                 } pitch2D;
     *             } res;
     *         };</pre>
     *   where:
     *   <ul>
     *     <li>
     *       <div>
     *         cudaResourceDesc::resType
     *         specifies the type of resource to texture from. CUresourceType is
     *         defined as:
     *         <pre>        enum cudaResourceType {
     *             cudaResourceTypeArray          = 0x00,
     *             cudaResourceTypeMipmappedArray = 0x01,
     *             cudaResourceTypeLinear         = 0x02,
     *             cudaResourceTypePitch2D        = 0x03
     *         };</pre>
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>If cudaResourceDesc::resType is set to
     *     cudaResourceTypeArray, cudaResourceDesc::res::array::array must be set
     *     to a valid CUDA array handle.
     *   </p>
     *   <p>If cudaResourceDesc::resType is set to
     *     cudaResourceTypeMipmappedArray, cudaResourceDesc::res::mipmap::mipmap
     *     must be set to a valid CUDA mipmapped array handle.
     *   </p>
     *   <p>If cudaResourceDesc::resType is set to
     *     cudaResourceTypeLinear, cudaResourceDesc::res::linear::devPtr must be
     *     set to a valid device pointer, that is aligned to
     *     cudaDeviceProp::textureAlignment. cudaResourceDesc::res::linear::desc
     *     describes the format and the number of components per array element.
     *     cudaResourceDesc::res::linear::sizeInBytes
     *     specifies the size of the array in bytes.
     *     The total number of elements in the linear address range cannot exceed
     *     cudaDeviceProp::maxTexture1DLinear. The number of elements is computed
     *     as (sizeInBytes / sizeof(desc)).
     *   </p>
     *   <p>If cudaResourceDesc::resType is set to
     *     cudaResourceTypePitch2D, cudaResourceDesc::res::pitch2D::devPtr must
     *     be set to a valid device pointer, that is aligned to
     *     cudaDeviceProp::textureAlignment. cudaResourceDesc::res::pitch2D::desc
     *     describes the format and the number of components per array element.
     *     cudaResourceDesc::res::pitch2D::width
     *     and cudaResourceDesc::res::pitch2D::height
     *     specify the width and height of the array in elements, and cannot
     *     exceed cudaDeviceProp::maxTexture2DLinear[0] and
     *     cudaDeviceProp::maxTexture2DLinear[1] respectively.
     *     cudaResourceDesc::res::pitch2D::pitchInBytes specifies the pitch
     *     between two rows in bytes and has to be
     *     aligned to cudaDeviceProp::texturePitchAlignment.
     *     Pitch cannot exceed cudaDeviceProp::maxTexture2DLinear[2].
     *   </p>
     *   <p>
     *     The cudaTextureDesc struct is defined as
     *   <pre>        struct cudaTextureDesc {
     *             enum cudaTextureAddressMode
     *                   addressMode[3];
     *             enum cudaTextureFilterMode
     *                   filterMode;
     *             enum cudaTextureReadMode
     *                   readMode;
     *             int                         sRGB;
     *             int                         normalizedCoords;
     *             unsigned int                maxAnisotropy;
     *             enum cudaTextureFilterMode
     *                   mipmapFilterMode;
     *             float                       mipmapLevelBias;
     *             float                       minMipmapLevelClamp;
     *             float                       maxMipmapLevelClamp;
     *         };</pre>
     *   where
     *   <ul>
     *     <li>
     *       <div>
     *         cudaTextureDesc::addressMode
     *         specifies the addressing mode for each dimension of the texture data.
     *         cudaTextureAddressMode is defined as:
     *         <pre>        enum
     * cudaTextureAddressMode {
     *             cudaAddressModeWrap   = 0,
     *             cudaAddressModeClamp  = 1,
     *             cudaAddressModeMirror = 2,
     *             cudaAddressModeBorder = 3
     *         };</pre>
     *         This is ignored if cudaResourceDesc::resType is
     *         cudaResourceTypeLinear. Also, if cudaTextureDesc::normalizedCoords is
     *         set to zero, the only supported address mode is cudaAddressModeClamp.
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <div>
     *         cudaTextureDesc::filterMode
     *         specifies the filtering mode to be used when fetching from the texture.
     *         cudaTextureFilterMode is defined as:
     *         <pre>        enum
     * cudaTextureFilterMode {
     *             cudaFilterModePoint  = 0,
     *             cudaFilterModeLinear = 1
     *         };</pre>
     *         This is ignored if cudaResourceDesc::resType is
     *         cudaResourceTypeLinear.
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <div>
     *         cudaTextureDesc::readMode
     *         specifies whether integer data should be converted to floating point
     *         or not. cudaTextureReadMode is defined as:
     *         <pre>        enum
     * cudaTextureReadMode {
     *             cudaReadModeElementType     = 0,
     *             cudaReadModeNormalizedFloat = 1
     *         };</pre>
     *         Note that this applies only to 8-bit and 16-bit
     *         integer formats. 32-bit integer format would not be promoted,
     *         regardless
     *         of whether or not this
     *         cudaTextureDesc::readMode is set cudaReadModeNormalizedFloat is
     *         specified.
     *       </div>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaTextureDesc::sRGB specifies
     *         whether sRGB to linear conversion should be performed during texture
     *         fetch.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaTextureDesc::normalizedCoords
     *         specifies whether the texture coordinates will be normalized or not.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaTextureDesc::maxAnisotropy
     *         specifies the maximum anistropy ratio to be used when doing anisotropic
     *         filtering. This value will be clamped to the range
     *         [1,16].
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaTextureDesc::mipmapFilterMode
     *         specifies the filter mode when the calculated mipmap level lies between
     *         two defined mipmap levels.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaTextureDesc::mipmapLevelBias
     *         specifies the offset to be applied to the calculated mipmap level.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaTextureDesc::minMipmapLevelClamp
     *         specifies the lower end of the mipmap level range to clamp access to.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaTextureDesc::maxMipmapLevelClamp
     *         specifies the upper end of the mipmap level range to clamp access to.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The cudaResourceViewDesc struct is
     *     defined as
     *   <pre>        struct cudaResourceViewDesc {
     *             enum cudaResourceViewFormat
     *                   format;
     *             size_t                      width;
     *             size_t                      height;
     *             size_t                      depth;
     *             unsigned int                firstMipmapLevel;
     *             unsigned int                lastMipmapLevel;
     *             unsigned int                firstLayer;
     *             unsigned int                lastLayer;
     *         };</pre>
     *   where:
     *   <ul>
     *     <li>
     *       <p>cudaResourceViewDesc::format
     *         specifies how the data contained in the CUDA array or CUDA mipmapped
     *         array should be interpreted. Note that this can incur
     *         a change in size of the texture
     *         data. If the resource view format is a block compressed format, then
     *         the underlying CUDA array
     *         or CUDA mipmapped array has to
     *         have a 32-bit unsigned integer format with 2 or 4 channels, depending
     *         on the block compressed
     *         format. For ex., BC1 and BC4
     *         require the underlying CUDA array to have a 32-bit unsigned int with 2
     *         channels. The other BC
     *         formats require the underlying
     *         resource to have the same 32-bit unsigned int format but with 4
     *         channels.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaResourceViewDesc::width
     *         specifies the new width of the texture data. If the resource view
     *         format is a block compressed format, this value has to
     *         be 4 times the original width
     *         of the resource. For non block compressed formats, this value has to
     *         be equal to that of the
     *         original resource.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaResourceViewDesc::height
     *         specifies the new height of the texture data. If the resource view
     *         format is a block compressed format, this value has to
     *         be 4 times the original height
     *         of the resource. For non block compressed formats, this value has to
     *         be equal to that of the
     *         original resource.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaResourceViewDesc::depth
     *         specifies the new depth of the texture data. This value has to be equal
     *         to that of the original resource.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaResourceViewDesc::firstMipmapLevel
     *         specifies the most detailed mipmap level. This will be the new mipmap
     *         level zero. For non-mipmapped resources, this value
     *         has to be
     *         zero.cudaTextureDesc::minMipmapLevelClamp and
     *         cudaTextureDesc::maxMipmapLevelClamp will be relative to this value.
     *         For ex., if the firstMipmapLevel is set to 2, and a minMipmapLevelClamp
     *         of 1.2 is specified,
     *         then the actual minimum mipmap
     *         level clamp will be 3.2.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaResourceViewDesc::lastMipmapLevel
     *         specifies the least detailed mipmap level. For non-mipmapped resources,
     *         this value has to be zero.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaResourceViewDesc::firstLayer
     *         specifies the first layer index for layered textures. This will be the
     *         new layer zero. For non-layered resources, this value
     *         has to be zero.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaResourceViewDesc::lastLayer
     *         specifies the last layer index for layered textures. For non-layered
     *         resources, this value has to be zero.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     * </div>
     *
     * @param pTexObject Texture object to create
     * @param pResDesc Resource descriptor
     * @param pTexDesc Texture descriptor
     * @param pResViewDesc Resource view descriptor
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaDestroyTextureObject
     */
    public static int cudaCreateTextureObject(cudaTextureObject pTexObject, cudaResourceDesc pResDesc, cudaTextureDesc pTexDesc, cudaResourceViewDesc pResViewDesc)
    {
        return checkResult(cudaCreateTextureObjectNative(pTexObject, pResDesc, pTexDesc, pResViewDesc));
    }
    private static native int cudaCreateTextureObjectNative(cudaTextureObject pTexObject, cudaResourceDesc pResDesc, cudaTextureDesc pTexDesc, cudaResourceViewDesc pResViewDesc);

    /**
     * Destroys a texture object.
     *
     * <pre>
     * cudaError_t cudaDestroyTextureObject (
     *      cudaTextureObject_t texObject )
     * </pre>
     * <div>
     *   <p>Destroys a texture object.  Destroys the
     *     texture object specified by <tt>texObject</tt>.
     *   </p>
     * </div>
     *
     * @param texObject Texture object to destroy
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaCreateTextureObject
     */
    public static int cudaDestroyTextureObject(cudaTextureObject texObject)
    {
        return checkResult(cudaDestroyTextureObjectNative(texObject));
    }
    private static native int cudaDestroyTextureObjectNative(cudaTextureObject texObject);

    /**
     * Returns a texture object's resource descriptor.
     *
     * <pre>
     * cudaError_t cudaGetTextureObjectResourceDesc (
     *      cudaResourceDesc* pResDesc,
     *      cudaTextureObject_t texObject )
     * </pre>
     * <div>
     *   <p>Returns a texture object's resource
     *     descriptor.  Returns the resource descriptor for the texture object
     *     specified by <tt>texObject</tt>.
     *   </p>
     * </div>
     *
     * @param pResDesc Resource descriptor
     * @param texObject Texture object
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaCreateTextureObject
     */
    public static int cudaGetTextureObjectResourceDesc(cudaResourceDesc pResDesc, cudaTextureObject texObject)
    {
        return checkResult(cudaGetTextureObjectResourceDescNative(pResDesc, texObject));
    }
    private static native int cudaGetTextureObjectResourceDescNative(cudaResourceDesc pResDesc, cudaTextureObject texObject);

    /**
     * Returns a texture object's texture descriptor.
     *
     * <pre>
     * cudaError_t cudaGetTextureObjectTextureDesc (
     *      cudaTextureDesc* pTexDesc,
     *      cudaTextureObject_t texObject )
     * </pre>
     * <div>
     *   <p>Returns a texture object's texture
     *     descriptor.  Returns the texture descriptor for the texture object
     *     specified by <tt>texObject</tt>.
     *   </p>
     * </div>
     *
     * @param pTexDesc Texture descriptor
     * @param texObject Texture object
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaCreateTextureObject
     */
    public static int cudaGetTextureObjectTextureDesc(cudaTextureDesc pTexDesc, cudaTextureObject texObject)
    {
        return checkResult(cudaGetTextureObjectTextureDescNative(pTexDesc, texObject));
    }
    private static native int cudaGetTextureObjectTextureDescNative(cudaTextureDesc pTexDesc, cudaTextureObject texObject);

    /**
     * Returns a texture object's resource view descriptor.
     *
     * <pre>
     * cudaError_t cudaGetTextureObjectResourceViewDesc (
     *      cudaResourceViewDesc* pResViewDesc,
     *      cudaTextureObject_t texObject )
     * </pre>
     * <div>
     *   <p>Returns a texture object's resource view
     *     descriptor.  Returns the resource view descriptor for the texture
     *     object specified
     *     by <tt>texObject</tt>. If no resource
     *     view was specified, cudaErrorInvalidValue is returned.
     *   </p>
     * </div>
     *
     * @param pResViewDesc Resource view descriptor
     * @param texObject Texture object
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaCreateTextureObject
     */
    public static int cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc pResViewDesc, cudaTextureObject texObject)
    {
        return checkResult(cudaGetTextureObjectResourceViewDescNative(pResViewDesc, texObject));
    }
    private static native int cudaGetTextureObjectResourceViewDescNative(cudaResourceViewDesc pResViewDesc, cudaTextureObject texObject);

    /**
     * Creates a surface object.
     *
     * <pre>
     * cudaError_t cudaCreateSurfaceObject (
     *      cudaSurfaceObject_t* pSurfObject,
     *      const cudaResourceDesc* pResDesc )
     * </pre>
     * <div>
     *   <p>Creates a surface object.  Creates a
     *     surface object and returns it in <tt>pSurfObject</tt>. <tt>pResDesc</tt> describes the data to perform surface load/stores on.
     *     cudaResourceDesc::resType must be cudaResourceTypeArray and
     *     cudaResourceDesc::res::array::array must be set to a valid CUDA array
     *     handle.
     *   </p>
     *   <p>Surface objects are only supported on
     *     devices of compute capability 3.0 or higher.
     *   </p>
     * </div>
     *
     * @param pSurfObject Surface object to create
     * @param pResDesc Resource descriptor
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaDestroySurfaceObject
     */
    public static int cudaCreateSurfaceObject(cudaSurfaceObject pSurfObject, cudaResourceDesc pResDesc)
    {
        return checkResult(cudaCreateSurfaceObjectNative(pSurfObject, pResDesc));
    }
    private static native int cudaCreateSurfaceObjectNative(cudaSurfaceObject pSurfObject, cudaResourceDesc pResDesc);

    /**
     * Destroys a surface object.
     *
     * <pre>
     * cudaError_t cudaDestroySurfaceObject (
     *      cudaSurfaceObject_t surfObject )
     * </pre>
     * <div>
     *   <p>Destroys a surface object.  Destroys the
     *     surface object specified by <tt>surfObject</tt>.
     *   </p>
     * </div>
     *
     * @param surfObject Surface object to destroy
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaCreateSurfaceObject
     */
    public static int cudaDestroySurfaceObject(cudaSurfaceObject surfObject)
    {
        return checkResult(cudaDestroySurfaceObjectNative(surfObject));
    }
    private static native int cudaDestroySurfaceObjectNative(cudaSurfaceObject surfObject);

    /**
     * Returns a surface object's resource descriptor Returns the resource descriptor for the surface object specified by surfObject.
     *
     * <pre>
     * cudaError_t cudaGetSurfaceObjectResourceDesc (
     *      cudaResourceDesc* pResDesc,
     *      cudaSurfaceObject_t surfObject )
     * </pre>
     * <div>
     *   <p>Returns a surface object's resource
     *     descriptor Returns the resource descriptor for the surface object
     *     specified by <tt>surfObject</tt>.
     *   </p>
     * </div>
     *
     * @param pResDesc Resource descriptor
     * @param surfObject Surface object
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaCreateSurfaceObject
     */
    public static int cudaGetSurfaceObjectResourceDesc(cudaResourceDesc pResDesc, cudaSurfaceObject surfObject)
    {
        return checkResult(cudaGetSurfaceObjectResourceDescNative(pResDesc, surfObject));
    }
    private static native int cudaGetSurfaceObjectResourceDescNative(cudaResourceDesc pResDesc, cudaSurfaceObject surfObject);



    /**
     * Configure a device-launch.
     *
     * <pre>
     * cudaError_t cudaConfigureCall (
     *      dim3 gridDim,
     *      dim3 blockDim,
     *      size_t sharedMem = 0,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Configure a device-launch.  Specifies
     *     the grid and block dimensions for the device call to be executed
     *     similar to the execution
     *     configuration syntax. cudaConfigureCall()
     *     is stack based. Each call pushes data on top of an execution stack.
     *     This data contains the dimension for the grid and thread
     *     blocks, together with any arguments for
     *     the call.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param gridDim Grid dimensions
     * @param blockDim Block dimensions
     * @param sharedMem Shared memory
     * @param stream Stream identifier
     *
     * @return cudaSuccess, cudaErrorInvalidConfiguration
     *
     * @see JCuda#cudaDeviceSetCacheConfig
     * @see JCuda#cudaFuncGetAttributes
     * @see JCuda#cudaLaunch
     * @see JCuda#cudaSetDoubleForDevice
     * @see JCuda#cudaSetDoubleForHost
     * @see JCuda#cudaSetupArgument
     *
     * @deprecated This function is deprecated as of CUDA 7.0
     */
    public static int cudaConfigureCall(dim3 gridDim, dim3 blockDim, long sharedMem, cudaStream_t stream)
    {
        return checkResult(cudaConfigureCallNative(gridDim, blockDim, sharedMem, stream));
    }
    private static native int cudaConfigureCallNative(dim3 gridDim, dim3 blockDim, long sharedMem, cudaStream_t stream);

    /**
     * [C++ API] Configure a device launch
     *
     * <pre>
     * template < class T > cudaError_t cudaSetupArgument (
     *      T arg,
     *      size_t offset ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Configure a device launch
     *     Pushes <tt>size</tt> bytes of the argument pointed to by <tt>arg</tt>
     *     at <tt>offset</tt> bytes from the start of the parameter passing area,
     *     which starts at offset 0. The arguments are stored in the top of the
     *     execution stack. cudaSetupArgument() must
     *     be preceded by a call to cudaConfigureCall().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param arg Argument to push for a kernel launch
     * @param size Size of argument
     * @param offset Offset in argument stack to push new arg
     * @param arg Argument to push for a kernel launch
     * @param offset Offset in argument stack to push new arg
     *
     * @return cudaSuccess
     *
     * @see JCuda#cudaConfigureCall
     * @see JCuda#cudaFuncGetAttributes
     * @see JCuda#cudaLaunch
     * @see JCuda#cudaSetDoubleForDevice
     * @see JCuda#cudaSetDoubleForHost
     * @see JCuda#cudaSetupArgument
     *
     * @deprecated This function is deprecated as of CUDA 7.0
     */
    public static int cudaSetupArgument(Pointer arg, long size, long offset)
    {
        return checkResult(cudaSetupArgumentNative(arg, size, offset));
    }
    private static native int cudaSetupArgumentNative(Pointer arg, long size, long offset);


    /**
     * [C++ API] Find out attributes for a given function
     *
     * <pre>
     * template < class T > cudaError_t cudaFuncGetAttributes (
     *      cudaFuncAttributes* attr,
     *      T* entry ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Find out attributes for a given
     *     function  This function obtains the attributes of a function specified
     *     via <tt>entry</tt>. The parameter <tt>entry</tt> must be a pointer
     *     to a function that executes on the device. The parameter specified by
     *     <tt>entry</tt> must be declared as a <tt>__global__</tt> function.
     *     The fetched attributes are placed in <tt>attr</tt>. If the specified
     *     function does not exist, then cudaErrorInvalidDeviceFunction is
     *     returned.
     *   </p>
     *   <p>Note that some function attributes such
     *     as maxThreadsPerBlock may vary based on the device that is currently
     *     being used.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param attr Return pointer to function's attributes
     * @param func Device function symbol
     * @param attr Return pointer to function's attributes
     * @param entry Function to get attributes of
     *
     * @return cudaSuccess, cudaErrorInitializationError,
     * cudaErrorInvalidDeviceFunction
     *
     * @see JCuda#cudaConfigureCall
     * @see JCuda#cudaDeviceSetCacheConfig
     * @see JCuda#cudaFuncGetAttributes
     * @see JCuda#cudaLaunch
     * @see JCuda#cudaSetDoubleForDevice
     * @see JCuda#cudaSetDoubleForHost
     * @see JCuda#cudaSetupArgument
     */
    public static int cudaFuncGetAttributes (cudaFuncAttributes attr, String func)
    {
        if (true)
        {
            throw new UnsupportedOperationException(
                "This function is no longer supported as of CUDA 5.0");
        }
        return checkResult(cudaFuncGetAttributesNative(attr, func));
    }
    private static native int cudaFuncGetAttributesNative(cudaFuncAttributes attr, String func);



    /**
     * [C++ API] Launches a device function
     *
     * <pre>
     * template < class T > cudaError_t cudaLaunch (
     *      T* func ) [inline]
     * </pre>
     * <div>
     *   <p>[C++ API] Launches a device function
     *     Launches the function <tt>entry</tt> on the device. The parameter <tt>entry</tt> must be a function that executes on the device. The
     *     parameter specified by <tt>entry</tt> must be declared as a <tt>__global__</tt> function. cudaLaunch() must be preceded by a call to
     *     cudaConfigureCall() since it pops the data that was pushed by
     *     cudaConfigureCall() from the execution stack.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param func Device function symbol
     *
     * @return cudaSuccess, cudaErrorInvalidDeviceFunction,
     * cudaErrorInvalidConfiguration, cudaErrorLaunchFailure,
     * cudaErrorLaunchTimeout, cudaErrorLaunchOutOfResources,
     * cudaErrorSharedObjectSymbolNotFound, cudaErrorSharedObjectInitFailed
     *
     * @see JCuda#cudaConfigureCall
     * @see JCuda#cudaDeviceSetCacheConfig
     * @see JCuda#cudaFuncGetAttributes
     * @see JCuda#cudaLaunch
     * @see JCuda#cudaSetDoubleForDevice
     * @see JCuda#cudaSetDoubleForHost
     * @see JCuda#cudaSetupArgument
     * @see JCuda#cudaThreadGetCacheConfig
     * @see JCuda#cudaThreadSetCacheConfig
     */
    public static int cudaLaunch(String symbol)
    {
        if (true)
        {
            throw new UnsupportedOperationException(
                "This function is no longer supported as of CUDA 5.0");
        }
        return checkResult(cudaLaunchNative(symbol));
    }
    private static native int cudaLaunchNative(String symbol);




    /**
     * Sets a CUDA device to use OpenGL interoperability.
     *
     * <pre>
     * cudaError_t cudaGLSetGLDevice (
     *      int  device )
     * </pre>
     * <div>
     *   <p>Sets a CUDA device to use OpenGL
     *     interoperability.
     *     Deprecated<span>This function is
     *     deprecated as of CUDA 5.0.</span>This function is deprecated and should
     *     no longer be used. It is no longer necessary to associate a CUDA device
     *     with an OpenGL
     *     context in order to achieve maximum
     *     interoperability performance.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param device Device to use for OpenGL interoperability
     *
     * @return cudaSuccess, cudaErrorInvalidDevice,
     * cudaErrorSetOnActiveProcess
     *
     * @see JCuda#cudaGraphicsGLRegisterBuffer
     * @see JCuda#cudaGraphicsGLRegisterImage
     * 
     * @deprecated Deprecated as of CUDA 5.0
     */
    public static int cudaGLSetGLDevice(int device)
    {
        return checkResult(cudaGLSetGLDeviceNative(device));
    }
    private static native int cudaGLSetGLDeviceNative(int device);



    /**
     * Gets the CUDA devices associated with the current OpenGL context.
     *
     * <pre>
     * cudaError_t cudaGLGetDevices (
     *      unsigned int* pCudaDeviceCount,
     *      int* pCudaDevices,
     *      unsigned int  cudaDeviceCount,
     *      cudaGLDeviceList deviceList )
     * </pre>
     * <div>
     *   <p>Gets the CUDA devices associated with
     *     the current OpenGL context.  Returns in <tt>*pCudaDeviceCount</tt>
     *     the number of CUDA-compatible devices corresponding to the current
     *     OpenGL context. Also returns in <tt>*pCudaDevices</tt> at most <tt>cudaDeviceCount</tt> of the CUDA-compatible devices corresponding to
     *     the current OpenGL context. If any of the GPUs being used by the
     *     current
     *     OpenGL context are not CUDA capable then
     *     the call will return cudaErrorNoDevice.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param pCudaDeviceCount Returned number of CUDA devices corresponding to the current OpenGL context
     * @param pCudaDevices Returned CUDA devices corresponding to the current OpenGL context
     * @param cudaDeviceCount The size of the output device array pCudaDevices
     * @param deviceList The set of devices to return. This set may be cudaGLDeviceListAll for all devices, cudaGLDeviceListCurrentFrame for the devices used to render the current frame (in SLI), or cudaGLDeviceListNextFrame for the devices used to render the next frame (in SLI).
     *
     * @return cudaSuccess, cudaErrorNoDevice, cudaErrorUnknown
     *
     * @see JCuda#cudaGraphicsUnregisterResource
     * @see JCuda#cudaGraphicsMapResources
     * @see JCuda#cudaGraphicsSubResourceGetMappedArray
     * @see JCuda#cudaGraphicsResourceGetMappedPointer
     */
    public static int cudaGLGetDevices(int pCudaDeviceCount[], int pCudaDevices[], int cudaDeviceCount, int cudaGLDeviceList_deviceList)
    {
        return checkResult(cudaGLGetDevicesNative(pCudaDeviceCount, pCudaDevices, cudaDeviceCount, cudaGLDeviceList_deviceList));
    }
    private static native int cudaGLGetDevicesNative(int pCudaDeviceCount[], int pCudaDevices[], int cudaDeviceCount, int cudaGLDeviceList_deviceList);




    /**
     * Register an OpenGL texture or renderbuffer object.
     *
     * <pre>
     * cudaError_t cudaGraphicsGLRegisterImage (
     *      cudaGraphicsResource** resource,
     *      GLuint image,
     *      GLenum target,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Register an OpenGL texture or renderbuffer
     *     object.  Registers the texture or renderbuffer object specified by <tt>image</tt> for access by CUDA. A handle to the registered object is
     *     returned as <tt>resource</tt>.
     *   </p>
     *   <p><tt>target</tt> must match the type of
     *     the object, and must be one of GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE,
     *     GL_TEXTURE_CUBE_MAP, GL_TEXTURE_3D,
     *     GL_TEXTURE_2D_ARRAY, or GL_RENDERBUFFER.
     *   </p>
     *   <p>The register flags <tt>flags</tt>
     *     specify the intended usage, as follows:
     *   <ul>
     *     <li>
     *       <p>cudaGraphicsRegisterFlagsNone:
     *         Specifies no hints about how this resource will be used. It is therefore
     *         assumed that this resource will be read from and
     *         written to by CUDA. This is the
     *         default value.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaGraphicsRegisterFlagsReadOnly:
     *         Specifies that CUDA will not write to this resource.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaGraphicsRegisterFlagsWriteDiscard:
     *         Specifies that CUDA will not read from this resource and will write
     *         over the entire contents of the resource, so none of
     *         the data previously stored in
     *         the resource will be preserved.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaGraphicsRegisterFlagsSurfaceLoadStore: Specifies that CUDA will
     *         bind this resource to a surface reference.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaGraphicsRegisterFlagsTextureGather:
     *         Specifies that CUDA will perform texture gather operations on this
     *         resource.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The following image formats are
     *     supported. For brevity's sake, the list is abbreviated. For ex., {GL_R,
     *     GL_RG} X {8, 16} would
     *     expand to the following 4 formats {GL_R8,
     *     GL_R16, GL_RG8, GL_RG16} :
     *   <ul>
     *     <li>
     *       <p>GL_RED, GL_RG, GL_RGBA,
     *         GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY
     *       </p>
     *     </li>
     *     <li>
     *       <p>{GL_R, GL_RG, GL_RGBA} X {8,
     *         16, 16F, 32F, 8UI, 16UI, 32UI, 8I, 16I, 32I}
     *       </p>
     *     </li>
     *     <li>
     *       <p>{GL_LUMINANCE, GL_ALPHA,
     *         GL_LUMINANCE_ALPHA, GL_INTENSITY} X {8, 16, 16F_ARB, 32F_ARB, 8UI_EXT,
     *         16UI_EXT, 32UI_EXT, 8I_EXT,
     *         16I_EXT, 32I_EXT}
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>The following image classes are currently
     *     disallowed:
     *   <ul>
     *     <li>
     *       <p>Textures with borders</p>
     *     </li>
     *     <li>
     *       <p>Multisampled renderbuffers</p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param resource Pointer to the returned object handle
     * @param image name of texture or renderbuffer object to be registered
     * @param target Identifies the type of object specified by image
     * @param flags Register flags
     *
     * @return cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue,
     * cudaErrorInvalidResourceHandle, cudaErrorUnknown
     *
     * @see JCuda#cudaGraphicsUnregisterResource
     * @see JCuda#cudaGraphicsMapResources
     * @see JCuda#cudaGraphicsSubResourceGetMappedArray
     */
    public static int cudaGraphicsGLRegisterImage(cudaGraphicsResource resource, int image, int target, int Flags)
    {
        return checkResult(cudaGraphicsGLRegisterImageNative(resource, image, target, Flags));
    }

    private static native int cudaGraphicsGLRegisterImageNative(cudaGraphicsResource resource, int image, int target, int Flags);


    /**
     * Registers an OpenGL buffer object.
     *
     * <pre>
     * cudaError_t cudaGraphicsGLRegisterBuffer (
     *      cudaGraphicsResource** resource,
     *      GLuint buffer,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Registers an OpenGL buffer object.
     *     Registers the buffer object specified by <tt>buffer</tt> for access
     *     by CUDA. A handle to the registered object is returned as <tt>resource</tt>. The register flags <tt>flags</tt> specify the intended
     *     usage, as follows:
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaGraphicsRegisterFlagsNone:
     *         Specifies no hints about how this resource will be used. It is therefore
     *         assumed that this resource will be read from and
     *         written to by CUDA. This is the
     *         default value.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaGraphicsRegisterFlagsReadOnly:
     *         Specifies that CUDA will not write to this resource.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaGraphicsRegisterFlagsWriteDiscard:
     *         Specifies that CUDA will not read from this resource and will write
     *         over the entire contents of the resource, so none of
     *         the data previously stored in
     *         the resource will be preserved.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param resource Pointer to the returned object handle
     * @param buffer name of buffer object to be registered
     * @param flags Register flags
     *
     * @return cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue,
     * cudaErrorInvalidResourceHandle, cudaErrorUnknown
     *
     * @see JCuda#cudaGraphicsUnregisterResource
     * @see JCuda#cudaGraphicsMapResources
     * @see JCuda#cudaGraphicsResourceGetMappedPointer
     */
    public static int cudaGraphicsGLRegisterBuffer(cudaGraphicsResource resource, int buffer, int Flags)
    {
        return checkResult(cudaGraphicsGLRegisterBufferNative(resource, buffer, Flags));
    }

    private static native int cudaGraphicsGLRegisterBufferNative(cudaGraphicsResource resource, int buffer, int Flags);




    /**
     * Registers a buffer object for access by CUDA.
     *
     * <pre>
     * cudaError_t cudaGLRegisterBufferObject (
     *      GLuint bufObj )
     * </pre>
     * <div>
     *   <p>Registers a buffer object for access by
     *     CUDA.
     *     Deprecated<span>This function is
     *     deprecated as of CUDA 3.0.</span>Registers the buffer object of ID <tt>bufObj</tt> for access by CUDA. This function must be called before
     *     CUDA can map the buffer object. The OpenGL context used to create
     *     the buffer, or another context from the
     *     same share group, must be bound to the current thread when this is
     *     called.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param bufObj Buffer object ID to register
     *
     * @return cudaSuccess, cudaErrorInitializationError
     *
     * @see JCuda#cudaGraphicsGLRegisterBuffer
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    public static int cudaGLRegisterBufferObject(int bufObj)
    {
        if (true)
        {
            throw new UnsupportedOperationException(
                "This function is deprecated as of CUDA 3.0");
        }
        return checkResult(cudaGLRegisterBufferObjectNative(bufObj));
    }
    private static native int cudaGLRegisterBufferObjectNative(int bufObj);


    /**
     * Maps a buffer object for access by CUDA.
     *
     * <pre>
     * cudaError_t cudaGLMapBufferObject (
     *      void** devPtr,
     *      GLuint bufObj )
     * </pre>
     * <div>
     *   <p>Maps a buffer object for access by CUDA.
     *     Deprecated<span>This function is
     *     deprecated as of CUDA 3.0.</span>Maps the buffer object of ID <tt>bufObj</tt> into the address space of CUDA and returns in <tt>*devPtr</tt> the base pointer of the resulting mapping. The buffer
     *     must have previously been registered by calling cudaGLRegisterBufferObject().
     *     While a buffer is mapped by CUDA, any OpenGL operation which references
     *     the buffer will result in undefined behavior. The
     *     OpenGL context used to create the buffer,
     *     or another context from the same share group, must be bound to the
     *     current thread
     *     when this is called.
     *   </p>
     *   <p>All streams in the current thread are
     *     synchronized with the current GL context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param devPtr Returned device pointer to CUDA object
     * @param bufObj Buffer object ID to map
     *
     * @return cudaSuccess, cudaErrorMapBufferObjectFailed
     *
     * @see JCuda#cudaGraphicsMapResources
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    public static int cudaGLMapBufferObject(Pointer devPtr, int bufObj)
    {
        return checkResult(cudaGLMapBufferObjectNative(devPtr, bufObj));
    }
    private static native int cudaGLMapBufferObjectNative(Pointer devPtr, int bufObj);


    /**
     * Unmaps a buffer object for access by CUDA.
     *
     * <pre>
     * cudaError_t cudaGLUnmapBufferObject (
     *      GLuint bufObj )
     * </pre>
     * <div>
     *   <p>Unmaps a buffer object for access by
     *     CUDA.
     *     Deprecated<span>This function is
     *     deprecated as of CUDA 3.0.</span>Unmaps the buffer object of ID <tt>bufObj</tt> for access by CUDA. When a buffer is unmapped, the base
     *     address returned by cudaGLMapBufferObject() is invalid and subsequent
     *     references to the address result in undefined behavior. The OpenGL
     *     context used to create the buffer,
     *     or another context from the same share
     *     group, must be bound to the current thread when this is called.
     *   </p>
     *   <p>All streams in the current thread are
     *     synchronized with the current GL context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param bufObj Buffer object to unmap
     *
     * @return cudaSuccess, cudaErrorInvalidDevicePointer,
     * cudaErrorUnmapBufferObjectFailed
     *
     * @see JCuda#cudaGraphicsUnmapResources
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    public static int cudaGLUnmapBufferObject(int bufObj)
    {
        return checkResult(cudaGLUnmapBufferObjectNative(bufObj));
    }
    private static native int cudaGLUnmapBufferObjectNative(int bufObj);


    /**
     * Unregisters a buffer object for access by CUDA.
     *
     * <pre>
     * cudaError_t cudaGLUnregisterBufferObject (
     *      GLuint bufObj )
     * </pre>
     * <div>
     *   <p>Unregisters a buffer object for access
     *     by CUDA.
     *     Deprecated<span>This function is
     *     deprecated as of CUDA 3.0.</span>Unregisters the buffer object of ID
     *     <tt>bufObj</tt> for access by CUDA and releases any CUDA resources
     *     associated with the buffer. Once a buffer is unregistered, it may no
     *     longer
     *     be mapped by CUDA. The GL context used
     *     to create the buffer, or another context from the same share group,
     *     must be bound to
     *     the current thread when this is called.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param bufObj Buffer object to unregister
     *
     * @return cudaSuccess
     *
     * @see JCuda#cudaGraphicsUnregisterResource
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    public static int cudaGLUnregisterBufferObject(int bufObj)
    {
        return checkResult(cudaGLUnregisterBufferObjectNative(bufObj));
    }
    private static native int cudaGLUnregisterBufferObjectNative(int bufObj);



    /**
     * Set usage flags for mapping an OpenGL buffer.
     *
     * <pre>
     * cudaError_t cudaGLSetBufferObjectMapFlags (
     *      GLuint bufObj,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Set usage flags for mapping an OpenGL
     *     buffer.
     *     Deprecated<span>This function is
     *     deprecated as of CUDA 3.0.</span>Set flags for mapping the OpenGL
     *     buffer <tt>bufObj</tt>
     *   </p>
     *   <p>Changes to flags will take effect the
     *     next time <tt>bufObj</tt> is mapped. The <tt>flags</tt> argument may
     *     be any of the following:
     *   </p>
     *   <ul>
     *     <li>
     *       <p>cudaGLMapFlagsNone: Specifies
     *         no hints about how this buffer will be used. It is therefore assumed
     *         that this buffer will be read from and written
     *         to by CUDA kernels. This is the
     *         default value.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaGLMapFlagsReadOnly:
     *         Specifies that CUDA kernels which access this buffer will not write to
     *         the buffer.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaGLMapFlagsWriteDiscard:
     *         Specifies that CUDA kernels which access this buffer will not read from
     *         the buffer and will write over the entire contents
     *         of the buffer, so none of the
     *         data previously stored in the buffer will be preserved.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>If <tt>bufObj</tt> has not been
     *     registered for use with CUDA, then cudaErrorInvalidResourceHandle is
     *     returned. If <tt>bufObj</tt> is presently mapped for access by CUDA,
     *     then cudaErrorUnknown is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param bufObj Registered buffer object to set flags for
     * @param flags Parameters for buffer mapping
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
     * cudaErrorUnknown
     *
     * @see JCuda#cudaGraphicsResourceSetMapFlags
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    public static int cudaGLSetBufferObjectMapFlags(int bufObj, int flags)
    {
        return checkResult(cudaGLSetBufferObjectMapFlagsNative(bufObj, flags));
    }

    private static native int cudaGLSetBufferObjectMapFlagsNative(int bufObj, int flags);;


    /**
     * Maps a buffer object for access by CUDA.
     *
     * <pre>
     * cudaError_t cudaGLMapBufferObjectAsync (
     *      void** devPtr,
     *      GLuint bufObj,
     *      cudaStream_t stream )
     * </pre>
     * <div>
     *   <p>Maps a buffer object for access by CUDA.
     *     Deprecated<span>This function is
     *     deprecated as of CUDA 3.0.</span>Maps the buffer object of ID <tt>bufObj</tt> into the address space of CUDA and returns in <tt>*devPtr</tt> the base pointer of the resulting mapping. The buffer
     *     must have previously been registered by calling cudaGLRegisterBufferObject().
     *     While a buffer is mapped by CUDA, any OpenGL operation which references
     *     the buffer will result in undefined behavior. The
     *     OpenGL context used to create the buffer,
     *     or another context from the same share group, must be bound to the
     *     current thread
     *     when this is called.
     *   </p>
     *   <p>Stream /p stream is synchronized with
     *     the current GL context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param devPtr Returned device pointer to CUDA object
     * @param bufObj Buffer object ID to map
     * @param stream Stream to synchronize
     *
     * @return cudaSuccess, cudaErrorMapBufferObjectFailed
     *
     * @see JCuda#cudaGraphicsMapResources
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    public static int cudaGLMapBufferObjectAsync(Pointer devPtr, int bufObj, cudaStream_t stream)
    {
        return checkResult(cudaGLMapBufferObjectAsyncNative(devPtr, bufObj, stream));
    }

    private static native int cudaGLMapBufferObjectAsyncNative(Pointer devPtr, int bufObj, cudaStream_t stream);

    /**
     * Unmaps a buffer object for access by CUDA.
     *
     * <pre>
     * cudaError_t cudaGLUnmapBufferObjectAsync (
     *      GLuint bufObj,
     *      cudaStream_t stream )
     * </pre>
     * <div>
     *   <p>Unmaps a buffer object for access by
     *     CUDA.
     *     Deprecated<span>This function is
     *     deprecated as of CUDA 3.0.</span>Unmaps the buffer object of ID <tt>bufObj</tt> for access by CUDA. When a buffer is unmapped, the base
     *     address returned by cudaGLMapBufferObject() is invalid and subsequent
     *     references to the address result in undefined behavior. The OpenGL
     *     context used to create the buffer,
     *     or another context from the same share
     *     group, must be bound to the current thread when this is called.
     *   </p>
     *   <p>Stream /p stream is synchronized with
     *     the current GL context.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param bufObj Buffer object to unmap
     * @param stream Stream to synchronize
     *
     * @return cudaSuccess, cudaErrorInvalidDevicePointer,
     * cudaErrorUnmapBufferObjectFailed
     *
     * @see JCuda#cudaGraphicsUnmapResources
     * 
     * @deprecated Deprecated as of CUDA 3.0
     */
    public static int cudaGLUnmapBufferObjectAsync(int bufObj, cudaStream_t stream)
    {
        return checkResult(cudaGLUnmapBufferObjectAsyncNative(bufObj, stream));
    }

    private static native int cudaGLUnmapBufferObjectAsyncNative(int bufObj, cudaStream_t stream);






    /**
     * Returns the CUDA driver version.
     *
     * <pre>
     * cudaError_t cudaDriverGetVersion (
     *      int* driverVersion )
     * </pre>
     * <div>
     *   <p>Returns the CUDA driver version.  Returns
     *     in <tt>*driverVersion</tt> the version number of the installed CUDA
     *     driver. If no driver is installed, then 0 is returned as the driver
     *     version (via
     *     <tt>driverVersion</tt>). This function
     *     automatically returns cudaErrorInvalidValue if the <tt>driverVersion</tt>
     *     argument is NULL.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param driverVersion Returns the CUDA driver version.
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaRuntimeGetVersion
     */
    public static int cudaDriverGetVersion(int driverVersion[])
    {
        return checkResult(cudaDriverGetVersionNative(driverVersion));
    }
    private static native int cudaDriverGetVersionNative(int driverVersion[]);

    /**
     * Returns the CUDA Runtime version.
     *
     * <pre>
     * cudaError_t cudaRuntimeGetVersion (
     *      int* runtimeVersion )
     * </pre>
     * <div>
     *   <p>Returns the CUDA Runtime version.
     *     Returns in <tt>*runtimeVersion</tt> the version number of the installed
     *     CUDA Runtime. This function automatically returns cudaErrorInvalidValue
     *     if the <tt>runtimeVersion</tt> argument is NULL.
     *   </p>
     * </div>
     *
     * @param runtimeVersion Returns the CUDA Runtime version.
     *
     * @return cudaSuccess, cudaErrorInvalidValue
     *
     * @see JCuda#cudaDriverGetVersion
     */
    public static int cudaRuntimeGetVersion(int runtimeVersion[])
    {
        return checkResult(cudaRuntimeGetVersionNative(runtimeVersion));
    }
    private static native int cudaRuntimeGetVersionNative(int runtimeVersion[]);





    /**
     * Returns attributes about a specified pointer.
     *
     * <pre>
     * cudaError_t cudaPointerGetAttributes (
     *      cudaPointerAttributes* attributes,
     *      const void* ptr )
     * </pre>
     * <div>
     *   <p>Returns attributes about a specified
     *     pointer.  Returns in <tt>*attributes</tt> the attributes of the
     *     pointer <tt>ptr</tt>.
     *   </p>
     *   <p>The cudaPointerAttributes structure is
     *     defined as:
     *   <pre>    struct cudaPointerAttributes {
     *         enum cudaMemoryType
     *                   memoryType;
     *         int device;
     *         void *devicePointer;
     *         void *hostPointer;
     *     }</pre>
     *   In this structure, the individual fields mean</p>
     *   <ul>
     *     <li>
     *       <p>memoryType identifies the
     *         physical location of the memory associated with pointer <tt>ptr</tt>.
     *         It can be cudaMemoryTypeHost for host memory or cudaMemoryTypeDevice
     *         for device memory.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>device is the device against
     *         which <tt>ptr</tt> was allocated. If <tt>ptr</tt> has memory type
     *         cudaMemoryTypeDevice then this identifies the device on which the
     *         memory referred to by <tt>ptr</tt> physically resides. If <tt>ptr</tt>
     *         has memory type cudaMemoryTypeHost then this identifies the device
     *         which was current when the allocation was made (and if that device is
     *         deinitialized then
     *         this allocation will vanish with
     *         that device's state).
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>devicePointer is the device
     *         pointer alias through which the memory referred to by <tt>ptr</tt>
     *         may be accessed on the current device. If the memory referred to by
     *         <tt>ptr</tt> cannot be accessed directly by the current device then
     *         this is NULL.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <ul>
     *     <li>
     *       <p>hostPointer is the host pointer
     *         alias through which the memory referred to by <tt>ptr</tt> may be
     *         accessed on the host. If the memory referred to by <tt>ptr</tt> cannot
     *         be accessed directly by the host then this is NULL.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     * </div>
     *
     * @param attributes Attributes for the specified pointer
     * @param ptr Pointer to get attributes for
     *
     * @return cudaSuccess, cudaErrorInvalidDevice
     *
     * @see JCuda#cudaGetDeviceCount
     * @see JCuda#cudaGetDevice
     * @see JCuda#cudaSetDevice
     * @see JCuda#cudaChooseDevice
     */
    public static int cudaPointerGetAttributes(cudaPointerAttributes attributes, Pointer ptr)
    {
        return checkResult(cudaPointerGetAttributesNative(attributes, ptr));
    }
    private static native int cudaPointerGetAttributesNative(cudaPointerAttributes attributes, Pointer ptr);

    /**
     * Queries if a device may directly access a peer device's memory.
     *
     * <pre>
     * cudaError_t cudaDeviceCanAccessPeer (
     *      int* canAccessPeer,
     *      int  device,
     *      int  peerDevice )
     * </pre>
     * <div>
     *   <p>Queries if a device may directly access
     *     a peer device's memory.  Returns in <tt>*canAccessPeer</tt> a value
     *     of 1 if device <tt>device</tt> is capable of directly accessing memory
     *     from <tt>peerDevice</tt> and 0 otherwise. If direct access of <tt>peerDevice</tt> from <tt>device</tt> is possible, then access may be
     *     enabled by calling cudaDeviceEnablePeerAccess().
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param canAccessPeer Returned access capability
     * @param device Device from which allocations on peerDevice are to be directly accessed.
     * @param peerDevice Device on which the allocations to be directly accessed by device reside.
     *
     * @return cudaSuccess, cudaErrorInvalidDevice
     *
     * @see JCuda#cudaDeviceEnablePeerAccess
     * @see JCuda#cudaDeviceDisablePeerAccess
     */
    public static int cudaDeviceCanAccessPeer(int canAccessPeer[], int device, int peerDevice)
    {
        return checkResult(cudaDeviceCanAccessPeerNative(canAccessPeer, device, peerDevice));
    }
    private static native int cudaDeviceCanAccessPeerNative(int canAccessPeer[], int device, int peerDevice);

    /**
     * Enables direct access to memory allocations on a peer device.
     *
     * <pre>
     * cudaError_t cudaDeviceEnablePeerAccess (
     *      int  peerDevice,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Enables direct access to memory
     *     allocations on a peer device.  On success, all allocations from <tt>peerDevice</tt> will immediately be accessible by the current device.
     *     They will remain accessible until access is explicitly disabled using
     *     cudaDeviceDisablePeerAccess() or either
     *     device is reset using cudaDeviceReset().
     *   </p>
     *   <p>Note that access granted by this call
     *     is unidirectional and that in order to access memory on the current
     *     device from <tt>peerDevice</tt>, a separate symmetric call to
     *     cudaDeviceEnablePeerAccess() is required.
     *   </p>
     *   <p>Each device can support a system-wide maximum of eight peer connections.
     *   <p>Peer access is not supported in 32 bit
     *     applications.
     *   </p>
     *   <p>Returns cudaErrorInvalidDevice if
     *     cudaDeviceCanAccessPeer() indicates that the current device cannot
     *     directly access memory from <tt>peerDevice</tt>.
     *   </p>
     *   <p>Returns cudaErrorPeerAccessAlreadyEnabled
     *     if direct access of <tt>peerDevice</tt> from the current device has
     *     already been enabled.
     *   </p>
     *   <p>Returns cudaErrorInvalidValue if <tt>flags</tt> is not 0.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param peerDevice Peer device to enable direct access to from the current device
     * @param flags Reserved for future use and must be set to 0
     *
     * @return cudaSuccess, cudaErrorInvalidDevice,
     * cudaErrorPeerAccessAlreadyEnabled, cudaErrorInvalidValue
     *
     * @see JCuda#cudaDeviceCanAccessPeer
     * @see JCuda#cudaDeviceDisablePeerAccess
     */
    public static int cudaDeviceEnablePeerAccess(int peerDevice, int flags)
    {
        return checkResult(cudaDeviceEnablePeerAccessNative(peerDevice, flags));
    }
    private static native int cudaDeviceEnablePeerAccessNative(int peerDevice, int flags);


    /**
     * Disables direct access to memory allocations on a peer device.
     *
     * <pre>
     * cudaError_t cudaDeviceDisablePeerAccess (
     *      int  peerDevice )
     * </pre>
     * <div>
     *   <p>Disables direct access to memory
     *     allocations on a peer device.  Returns cudaErrorPeerAccessNotEnabled
     *     if direct access to memory on <tt>peerDevice</tt> has not yet been
     *     enabled from the current device.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param peerDevice Peer device to disable direct access to
     *
     * @return cudaSuccess, cudaErrorPeerAccessNotEnabled,
     * cudaErrorInvalidDevice
     *
     * @see JCuda#cudaDeviceCanAccessPeer
     * @see JCuda#cudaDeviceEnablePeerAccess
     */
    public static int cudaDeviceDisablePeerAccess(int peerDevice)
    {
        return checkResult(cudaDeviceDisablePeerAccessNative(peerDevice));
    }
    private static native int cudaDeviceDisablePeerAccessNative(int peerDevice);




    /**
     * Unregisters a graphics resource for access by CUDA.
     *
     * <pre>
     * cudaError_t cudaGraphicsUnregisterResource (
     *      cudaGraphicsResource_t resource )
     * </pre>
     * <div>
     *   <p>Unregisters a graphics resource for
     *     access by CUDA.  Unregisters the graphics resource <tt>resource</tt>
     *     so it is not accessible by CUDA unless registered again.
     *   </p>
     *   <p>If <tt>resource</tt> is invalid then
     *     cudaErrorInvalidResourceHandle is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param resource Resource to unregister
     *
     * @return cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorUnknown
     *
     * @see JCuda#cudaGraphicsD3D9RegisterResource
     * @see JCuda#cudaGraphicsD3D10RegisterResource
     * @see JCuda#cudaGraphicsD3D11RegisterResource
     * @see JCuda#cudaGraphicsGLRegisterBuffer
     * @see JCuda#cudaGraphicsGLRegisterImage
     */
    public static int cudaGraphicsUnregisterResource(cudaGraphicsResource resource)
    {
        return checkResult(cudaGraphicsUnregisterResourceNative(resource));
    }

    private static native int cudaGraphicsUnregisterResourceNative(cudaGraphicsResource resource);


    /**
     * Set usage flags for mapping a graphics resource.
     *
     * <pre>
     * cudaError_t cudaGraphicsResourceSetMapFlags (
     *      cudaGraphicsResource_t resource,
     *      unsigned int  flags )
     * </pre>
     * <div>
     *   <p>Set usage flags for mapping a graphics
     *     resource.  Set <tt>flags</tt> for mapping the graphics resource <tt>resource</tt>.
     *   </p>
     *   <p>Changes to <tt>flags</tt> will take
     *     effect the next time <tt>resource</tt> is mapped. The <tt>flags</tt>
     *     argument may be any of the following:
     *   <ul>
     *     <li>
     *       <p>cudaGraphicsMapFlagsNone:
     *         Specifies no hints about how <tt>resource</tt> will be used. It is
     *         therefore assumed that CUDA may read from or write to <tt>resource</tt>.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaGraphicsMapFlagsReadOnly:
     *         Specifies that CUDA will not write to <tt>resource</tt>.
     *       </p>
     *     </li>
     *     <li>
     *       <p>cudaGraphicsMapFlagsWriteDiscard:
     *         Specifies CUDA will not read from <tt>resource</tt> and will write
     *         over the entire contents of <tt>resource</tt>, so none of the data
     *         previously stored in <tt>resource</tt> will be preserved.
     *       </p>
     *     </li>
     *   </ul>
     *   </p>
     *   <p>If <tt>resource</tt> is presently
     *     mapped for access by CUDA then cudaErrorUnknown is returned. If <tt>flags</tt> is not one of the above values then cudaErrorInvalidValue
     *     is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param resource Registered resource to set flags for
     * @param flags Parameters for resource mapping
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
     * cudaErrorUnknown,
     *
     * @see JCuda#cudaGraphicsMapResources
     */
    public static int cudaGraphicsResourceSetMapFlags(cudaGraphicsResource resource, int flags)
    {
        return checkResult(cudaGraphicsResourceSetMapFlagsNative(resource, flags));
    }

    private static native int cudaGraphicsResourceSetMapFlagsNative(cudaGraphicsResource resource, int flags);

    /**
     * Map graphics resources for access by CUDA.
     *
     * <pre>
     * cudaError_t cudaGraphicsMapResources (
     *      int  count,
     *      cudaGraphicsResource_t* resources,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Map graphics resources for access by
     *     CUDA.  Maps the <tt>count</tt> graphics resources in <tt>resources</tt>
     *     for access by CUDA.
     *   </p>
     *   <p>The resources in <tt>resources</tt>
     *     may be accessed by CUDA until they are unmapped. The graphics API from
     *     which <tt>resources</tt> were registered should not access any
     *     resources while they are mapped by CUDA. If an application does so,
     *     the results are
     *     undefined.
     *   </p>
     *   <p>This function provides the synchronization
     *     guarantee that any graphics calls issued before cudaGraphicsMapResources()
     *     will complete before any subsequent CUDA work issued in <tt>stream</tt>
     *     begins.
     *   </p>
     *   <p>If <tt>resources</tt> contains any
     *     duplicate entries then cudaErrorInvalidResourceHandle is returned. If
     *     any of <tt>resources</tt> are presently mapped for access by CUDA then
     *     cudaErrorUnknown is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param count Number of resources to map
     * @param resources Resources to map for CUDA
     * @param stream Stream for synchronization
     *
     * @return cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorUnknown
     *
     * @see JCuda#cudaGraphicsResourceGetMappedPointer
     * @see JCuda#cudaGraphicsSubResourceGetMappedArray
     * @see JCuda#cudaGraphicsUnmapResources
     */
    public static int cudaGraphicsMapResources(int count, cudaGraphicsResource resources[], cudaStream_t stream)
    {
        return checkResult(cudaGraphicsMapResourcesNative(count, resources, stream));
    }

    private static native int cudaGraphicsMapResourcesNative(int count, cudaGraphicsResource resources[], cudaStream_t stream);


    /**
     * Unmap graphics resources.
     *
     * <pre>
     * cudaError_t cudaGraphicsUnmapResources (
     *      int  count,
     *      cudaGraphicsResource_t* resources,
     *      cudaStream_t stream = 0 )
     * </pre>
     * <div>
     *   <p>Unmap graphics resources.  Unmaps the
     *     <tt>count</tt> graphics resources in <tt>resources</tt>.
     *   </p>
     *   <p>Once unmapped, the resources in <tt>resources</tt> may not be accessed by CUDA until they are mapped
     *     again.
     *   </p>
     *   <p>This function provides the synchronization
     *     guarantee that any CUDA work issued in <tt>stream</tt> before
     *     cudaGraphicsUnmapResources() will complete before any subsequently
     *     issued graphics work begins.
     *   </p>
     *   <p>If <tt>resources</tt> contains any
     *     duplicate entries then cudaErrorInvalidResourceHandle is returned. If
     *     any of <tt>resources</tt> are not presently mapped for access by CUDA
     *     then cudaErrorUnknown is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param count Number of resources to unmap
     * @param resources Resources to unmap
     * @param stream Stream for synchronization
     *
     * @return cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorUnknown
     *
     * @see JCuda#cudaGraphicsMapResources
     */
    public static int cudaGraphicsUnmapResources(int count, cudaGraphicsResource resources[], cudaStream_t stream)
    {
        return checkResult(cudaGraphicsUnmapResourcesNative(count, resources, stream));
    }

    private static native int cudaGraphicsUnmapResourcesNative(int count, cudaGraphicsResource resources[], cudaStream_t stream);

    /**
     * Get an device pointer through which to access a mapped graphics resource.
     *
     * <pre>
     * cudaError_t cudaGraphicsResourceGetMappedPointer (
     *      void** devPtr,
     *      size_t* size,
     *      cudaGraphicsResource_t resource )
     * </pre>
     * <div>
     *   <p>Get an device pointer through which to
     *     access a mapped graphics resource.  Returns in <tt>*devPtr</tt> a
     *     pointer through which the mapped graphics resource <tt>resource</tt>
     *     may be accessed. Returns in <tt>*size</tt> the size of the memory in
     *     bytes which may be accessed from that pointer. The value set in <tt>devPtr</tt> may change every time that <tt>resource</tt> is mapped.
     *   </p>
     *   <p>If <tt>resource</tt> is not a buffer
     *     then it cannot be accessed via a pointer and cudaErrorUnknown is
     *     returned. If <tt>resource</tt> is not mapped then cudaErrorUnknown is
     *     returned. *
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param devPtr Returned pointer through which resource may be accessed
     * @param size Returned size of the buffer accessible starting at *devPtr
     * @param resource Mapped resource to access
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
     * cudaErrorUnknown
     *
     * @see JCuda#cudaGraphicsMapResources
     * @see JCuda#cudaGraphicsSubResourceGetMappedArray
     */
    public static int cudaGraphicsResourceGetMappedPointer(Pointer devPtr, long size[], cudaGraphicsResource resource)
    {
        return checkResult(cudaGraphicsResourceGetMappedPointerNative(devPtr, size, resource));
    }

    private static native int cudaGraphicsResourceGetMappedPointerNative(Pointer devPtr, long size[], cudaGraphicsResource resource);

    /**
     * Get an array through which to access a subresource of a mapped graphics resource.
     *
     * <pre>
     * cudaError_t cudaGraphicsSubResourceGetMappedArray (
     *      cudaArray_t* array,
     *      cudaGraphicsResource_t resource,
     *      unsigned int  arrayIndex,
     *      unsigned int  mipLevel )
     * </pre>
     * <div>
     *   <p>Get an array through which to access a
     *     subresource of a mapped graphics resource.  Returns in <tt>*array</tt>
     *     an array through which the subresource of the mapped graphics resource
     *     <tt>resource</tt> which corresponds to array index <tt>arrayIndex</tt>
     *     and mipmap level <tt>mipLevel</tt> may be accessed. The value set in
     *     <tt>array</tt> may change every time that <tt>resource</tt> is
     *     mapped.
     *   </p>
     *   <p>If <tt>resource</tt> is not a texture
     *     then it cannot be accessed via an array and cudaErrorUnknown is
     *     returned. If <tt>arrayIndex</tt> is not a valid array index for <tt>resource</tt> then cudaErrorInvalidValue is returned. If <tt>mipLevel</tt> is not a valid mipmap level for <tt>resource</tt> then
     *     cudaErrorInvalidValue is returned. If <tt>resource</tt> is not mapped
     *     then cudaErrorUnknown is returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param array Returned array through which a subresource of resource may be accessed
     * @param resource Mapped resource to access
     * @param arrayIndex Array index for array textures or cubemap face index as defined by cudaGraphicsCubeFace for cubemap textures for the subresource to access
     * @param mipLevel Mipmap level for the subresource to access
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
     * cudaErrorUnknown
     *
     * @see JCuda#cudaGraphicsResourceGetMappedPointer
     */
    public static int cudaGraphicsSubResourceGetMappedArray(cudaArray arrayPtr, cudaGraphicsResource resource, int arrayIndex, int mipLevel)
    {
        return checkResult(cudaGraphicsSubResourceGetMappedArrayNative(arrayPtr, resource, arrayIndex, mipLevel));
    }

    private static native int cudaGraphicsSubResourceGetMappedArrayNative(cudaArray arrayPtr, cudaGraphicsResource resource, int arrayIndex, int mipLevel);


    /**
     * Get a mipmapped array through which to access a mapped graphics resource.
     *
     * <pre>
     * cudaError_t cudaGraphicsResourceGetMappedMipmappedArray (
     *      cudaMipmappedArray_t* mipmappedArray,
     *      cudaGraphicsResource_t resource )
     * </pre>
     * <div>
     *   <p>Get a mipmapped array through which to
     *     access a mapped graphics resource.  Returns in <tt>*mipmappedArray</tt>
     *     a mipmapped array through which the mapped graphics resource <tt>resource</tt> may be accessed. The value set in <tt>mipmappedArray</tt>
     *     may change every time that <tt>resource</tt> is mapped.
     *   </p>
     *   <p>If <tt>resource</tt> is not a texture
     *     then it cannot be accessed via an array and cudaErrorUnknown is
     *     returned. If <tt>resource</tt> is not mapped then cudaErrorUnknown is
     *     returned.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param mipmappedArray Returned mipmapped array through which resource may be accessed
     * @param resource Mapped resource to access
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle,
     * cudaErrorUnknown
     *
     * @see JCuda#cudaGraphicsResourceGetMappedPointer
     */
    public static int cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray mipmappedArray, cudaGraphicsResource resource)
    {
        return checkResult(cudaGraphicsResourceGetMappedMipmappedArrayNative(mipmappedArray, resource));
    }
    private static native int cudaGraphicsResourceGetMappedMipmappedArrayNative(cudaMipmappedArray mipmappedArray, cudaGraphicsResource resource);


    /**
     * Initialize the CUDA profiler.
     *
     * <pre>
     * cudaError_t cudaProfilerInitialize (
     *      const char* configFile,
     *      const char* outputFile,
     *      cudaOutputMode_t outputMode )
     * </pre>
     * <div>
     *   <p>Initialize the CUDA profiler.  Using this
     *     API user can initialize the CUDA profiler by specifying the configuration
     *     file,
     *     output file and output file format. This
     *     API is generally used to profile different set of counters by looping
     *     the kernel
     *     launch. The <tt>configFile</tt> parameter
     *     can be used to select profiling options including profiler counters.
     *     Refer to the "Compute Command Line Profiler
     *     User Guide" for supported profiler
     *     options and counters.
     *   </p>
     *   <p>Limitation: The CUDA profiler cannot be
     *     initialized with this API if another profiling tool is already active,
     *     as indicated
     *     by the cudaErrorProfilerDisabled return
     *     code.
     *   </p>
     *   <p>Typical usage of the profiling APIs is
     *     as follows:
     *   </p>
     *   <p>for each set of counters/options
     *     {
     *     cudaProfilerInitialize(); //Initialize
     *     profiling,set the counters/options in the config file
     *     ...
     *     cudaProfilerStart();
     *     // code to be profiled
     *     cudaProfilerStop();
     *     ...
     *     cudaProfilerStart();
     *     // code to be profiled
     *     cudaProfilerStop();
     *     ...
     *     }
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     * @param configFile Name of the config file that lists the counters/options for profiling.
     * @param outputFile Name of the outputFile where the profiling results will be stored.
     * @param outputMode outputMode, can be cudaKeyValuePair OR cudaCSV.
     *
     * @return cudaSuccess, cudaErrorInvalidValue, cudaErrorProfilerDisabled
     *
     * @see JCuda#cudaProfilerStart
     * @see JCuda#cudaProfilerStop
     */
    public static int cudaProfilerInitialize(String configFile, String outputFile, int outputMode)
    {
        return checkResult(cudaProfilerInitializeNative(configFile, outputFile, outputMode));
    }
    private static native int cudaProfilerInitializeNative(String configFile, String outputFile, int outputMode);

    /**
     * Enable profiling.
     *
     * <pre>
     * cudaError_t cudaProfilerStart (
     *      void )
     * </pre>
     * <div>
     *   <p>Enable profiling.  Enables profile
     *     collection by the active profiling tool. If profiling is already
     *     enabled, then cudaProfilerStart() has no effect.
     *   </p>
     *   <p>cudaProfilerStart and cudaProfilerStop
     *     APIs are used to programmatically control the profiling granularity by
     *     allowing profiling
     *     to be done only on selective pieces of
     *     code.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return cudaSuccess
     *
     * @see JCuda#cudaProfilerInitialize
     * @see JCuda#cudaProfilerStop
     */
    public static int cudaProfilerStart()
    {
        return checkResult(cudaProfilerStartNative());
    }
    private static native int cudaProfilerStartNative();

    /**
     * Disable profiling.
     *
     * <pre>
     * cudaError_t cudaProfilerStop (
     *      void )
     * </pre>
     * <div>
     *   <p>Disable profiling.  Disables profile
     *     collection by the active profiling tool. If profiling is already
     *     disabled, then cudaProfilerStop() has no effect.
     *   </p>
     *   <p>cudaProfilerStart and cudaProfilerStop
     *     APIs are used to programmatically control the profiling granularity by
     *     allowing profiling
     *     to be done only on selective pieces of
     *     code.
     *   </p>
     *   <div>
     *     <span>Note:</span>
     *     <p>Note that this
     *       function may also return error codes from previous, asynchronous
     *       launches.
     *     </p>
     *   </div>
     *   </p>
     * </div>
     *
     *
     * @return cudaSuccess
     *
     * @see JCuda#cudaProfilerInitialize
     * @see JCuda#cudaProfilerStart
     */
    public static int cudaProfilerStop()
    {
        return checkResult(cudaProfilerStopNative());
    }
    private static native int cudaProfilerStopNative();

}



