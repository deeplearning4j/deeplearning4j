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

package jcuda.driver;

/**
 * Error codes.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 */
public class CUresult
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     */
    public static final int CUDA_SUCCESS                              = 0;

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    public static final int CUDA_ERROR_INVALID_VALUE                  = 1;

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    public static final int CUDA_ERROR_OUT_OF_MEMORY                  = 2;

    /**
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    public static final int CUDA_ERROR_NOT_INITIALIZED                = 3;

    /**
     * This indicates that the CUDA driver is in the process of shutting down.
     */
    public static final int CUDA_ERROR_DEINITIALIZED                  = 4;

    /**
     * This indicates profiling APIs are called while application is running
     * in visual profiler mode.
     */
    public static final int CUDA_ERROR_PROFILER_DISABLED           = 5;

    /**
     * This indicates profiling has not been initialized for this context.
     * Call cuProfilerInitialize() to resolve this.
     * @deprecated This error return is deprecated as of CUDA 5.0.
     * It is no longer an error to attempt to enable/disable the
     * profiling via ::cuProfilerStart or ::cuProfilerStop without
     * initialization.
     */
    public static final int CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6;

    /**
     * This indicates profiler has already been started and probably
     * cuProfilerStart() is incorrectly called.
     * @deprecated This error return is deprecated as of CUDA 5.0.
     * It is no longer an error to call cuProfilerStart() when
     * profiling is already enabled.
     */
    public static final int CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7;

    /**
     * This indicates profiler has already been stopped and probably
     * cuProfilerStop() is incorrectly called.
     * @deprecated This error return is deprecated as of CUDA 5.0.
     * It is no longer an error to call cuProfilerStop() when
     * profiling is already disabled.
     */
    public static final int CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8;

    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    public static final int CUDA_ERROR_NO_DEVICE                      = 100;

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    public static final int CUDA_ERROR_INVALID_DEVICE                 = 101;


    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     */
    public static final int CUDA_ERROR_INVALID_IMAGE                  = 200;

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    public static final int CUDA_ERROR_INVALID_CONTEXT                = 201;

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     */
    public static final int CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202;

    /**
     * This indicates that a map or register operation has failed.
     */
    public static final int CUDA_ERROR_MAP_FAILED                     = 205;

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    public static final int CUDA_ERROR_UNMAP_FAILED                   = 206;

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    public static final int CUDA_ERROR_ARRAY_IS_MAPPED                = 207;

    /**
     * This indicates that the resource is already mapped.
     */
    public static final int CUDA_ERROR_ALREADY_MAPPED                 = 208;

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    public static final int CUDA_ERROR_NO_BINARY_FOR_GPU              = 209;

    /**
     * This indicates that a resource has already been acquired.
     */
    public static final int CUDA_ERROR_ALREADY_ACQUIRED               = 210;

    /**
     * This indicates that a resource is not mapped.
     */
    public static final int CUDA_ERROR_NOT_MAPPED                     = 211;

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    public static final int CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212;

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    public static final int CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213;

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    public static final int CUDA_ERROR_ECC_UNCORRECTABLE              = 214;

    /**
     * This indicates that the ::CUlimit passed to the API call is not
     * supported by the active device.
     */
    public static final int CUDA_ERROR_UNSUPPORTED_LIMIT              = 215;

    /**
     * This indicates that the ::CUcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already
     * bound to a CPU thread.
     */
    public static final int CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216;

    /**
     * This indicates that peer access is not supported across the given
     * devices.
     */
    public static final int CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        = 217;

    /**
     * This indicates that a PTX JIT compilation failed.
     */
    public static final int CUDA_ERROR_INVALID_PTX                    = 218;

    /**
     * This indicates an error with OpenGL or DirectX context.
     */
    public static final int CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       = 219;

    /**
     * This indicates that the device kernel source is invalid.
     */
    public static final int CUDA_ERROR_INVALID_SOURCE                 = 300;

    /**
     * This indicates that the file specified was not found.
     */
    public static final int CUDA_ERROR_FILE_NOT_FOUND                 = 301;

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    public static final int CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302;

    /**
     * This indicates that initialization of a shared object failed.
     */
    public static final int CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303;

    /**
     * This indicates that an OS call failed.
     */
    public static final int CUDA_ERROR_OPERATING_SYSTEM               = 304;


    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
     */
    public static final int CUDA_ERROR_INVALID_HANDLE                 = 400;


    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names, and surface names.
     */
    public static final int CUDA_ERROR_NOT_FOUND                      = 500;


    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    public static final int CUDA_ERROR_NOT_READY                      = 600;

    /**
     * While executing a kernel, the device encountered a
     * load or store instruction on an invalid memory address.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int CUDA_ERROR_ILLEGAL_ADDRESS                = 700;

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    public static final int CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701;

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    public static final int CUDA_ERROR_LAUNCH_TIMEOUT                 = 702;

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    public static final int CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703;


    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    public static final int CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704;

    /**
     * This error indicates that a call to ::cuMemPeerRegister is trying to
     * register memory from a context which has not had peer access
     * enabled yet via ::cuCtxEnablePeerAccess(), or that
     * ::cuCtxDisablePeerAccess() is trying to disable peer access
     * which has not been enabled yet.
     */
    public static final int CUDA_ERROR_PEER_ACCESS_NOT_ENABLED    = 705;

    /**
     * This error indicates that a call to ::cuMemPeerRegister is trying to
     * register already-registered memory.
     * @deprecated This value has been added in CUDA 4.0 RC,
     * and removed in CUDA 4.0 RC2
     */
    public static final int CUDA_ERROR_PEER_MEMORY_ALREADY_REGISTERED = 706;

    /**
     * This error indicates that a call to ::cuMemPeerUnregister is trying to
     * unregister memory that has not been registered.
     * @deprecated This value has been added in CUDA 4.0 RC,
     * and removed in CUDA 4.0 RC2
     */
    public static final int CUDA_ERROR_PEER_MEMORY_NOT_REGISTERED     = 707;

    /**
     * This error indicates that ::cuCtxCreate was called with the flag
     * ::CU_CTX_PRIMARY on a device which already has initialized its
     * primary context.
     */
    public static final int CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708;

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    public static final int CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709;


    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    public static final int CUDA_ERROR_ASSERT                         = 710;

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices
     * passed to ::cuCtxEnablePeerAccess().
     */
    public static final int CUDA_ERROR_TOO_MANY_PEERS                 = 711;

    /**
     * This error indicates that the memory range passed to ::cuMemHostRegister()
     * has already been registered.
     */
    public static final int CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712;

    /**
     * This error indicates that the pointer passed to ::cuMemHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    public static final int CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713;

    /**
     * While executing a kernel, the device encountered a stack error.
     * This can be due to stack corruption or exceeding the stack size limit.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int CUDA_ERROR_HARDWARE_STACK_ERROR           = 714;

    /**
     * While executing a kernel, the device encountered an illegal instruction.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int CUDA_ERROR_ILLEGAL_INSTRUCTION            = 715;

    /**
     * While executing a kernel, the device encountered a load or store instruction
     * on a memory address which is not aligned.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int CUDA_ERROR_MISALIGNED_ADDRESS             = 716;

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int CUDA_ERROR_INVALID_ADDRESS_SPACE          = 717;

    /**
     * While executing a kernel, the device program counter wrapped its address space.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int CUDA_ERROR_INVALID_PC                     = 718;

    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    public static final int CUDA_ERROR_LAUNCH_FAILED                  = 719;

    /**
     * This error indicates that the attempted operation is not permitted.
     */
    public static final int CUDA_ERROR_NOT_PERMITTED                  = 800;

    /**
     * This error indicates that the attempted operation is not supported
     * on the current system or device.
     */
    public static final int CUDA_ERROR_NOT_SUPPORTED                  = 801;

    /**
     * This indicates that an unknown internal error has occurred.
     */
    public static final int CUDA_ERROR_UNKNOWN                        = 999;


    /**
     * Returns the String identifying the given CUresult
     *
     * @param result The CUresult value
     * @return The String identifying the given CUresult
     */
    public static String stringFor(int result)
    {
        switch (result)
        {
            case CUDA_SUCCESS                              : return "CUDA_SUCCESS";
            case CUDA_ERROR_INVALID_VALUE                  : return "CUDA_ERROR_INVALID_VALUE";
            case CUDA_ERROR_OUT_OF_MEMORY                  : return "CUDA_ERROR_OUT_OF_MEMORY";
            case CUDA_ERROR_NOT_INITIALIZED                : return "CUDA_ERROR_NOT_INITIALIZED";
            case CUDA_ERROR_DEINITIALIZED                  : return "CUDA_ERROR_DEINITIALIZED";
            case CUDA_ERROR_PROFILER_DISABLED              : return "CUDA_ERROR_PROFILER_DISABLED";
            case CUDA_ERROR_PROFILER_NOT_INITIALIZED       : return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
            case CUDA_ERROR_PROFILER_ALREADY_STARTED       : return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
            case CUDA_ERROR_PROFILER_ALREADY_STOPPED       : return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
            case CUDA_ERROR_NO_DEVICE                      : return "CUDA_ERROR_NO_DEVICE";
            case CUDA_ERROR_INVALID_DEVICE                 : return "CUDA_ERROR_INVALID_DEVICE";
            case CUDA_ERROR_INVALID_IMAGE                  : return "CUDA_ERROR_INVALID_IMAGE";
            case CUDA_ERROR_INVALID_CONTEXT                : return "CUDA_ERROR_INVALID_CONTEXT";
            case CUDA_ERROR_CONTEXT_ALREADY_CURRENT        : return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
            case CUDA_ERROR_MAP_FAILED                     : return "CUDA_ERROR_MAP_FAILED";
            case CUDA_ERROR_UNMAP_FAILED                   : return "CUDA_ERROR_UNMAP_FAILED";
            case CUDA_ERROR_ARRAY_IS_MAPPED                : return "CUDA_ERROR_ARRAY_IS_MAPPED";
            case CUDA_ERROR_ALREADY_MAPPED                 : return "CUDA_ERROR_ALREADY_MAPPED";
            case CUDA_ERROR_NO_BINARY_FOR_GPU              : return "CUDA_ERROR_NO_BINARY_FOR_GPU";
            case CUDA_ERROR_ALREADY_ACQUIRED               : return "CUDA_ERROR_ALREADY_ACQUIRED";
            case CUDA_ERROR_NOT_MAPPED                     : return "CUDA_ERROR_NOT_MAPPED";
            case CUDA_ERROR_NOT_MAPPED_AS_ARRAY            : return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
            case CUDA_ERROR_NOT_MAPPED_AS_POINTER          : return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
            case CUDA_ERROR_ECC_UNCORRECTABLE              : return "CUDA_ERROR_ECC_UNCORRECTABLE";
            case CUDA_ERROR_UNSUPPORTED_LIMIT              : return "CUDA_ERROR_UNSUPPORTED_LIMIT";
            case CUDA_ERROR_CONTEXT_ALREADY_IN_USE         : return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
            case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        : return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
            case CUDA_ERROR_INVALID_PTX                    : return "CUDA_ERROR_INVALID_PTX";
            case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       : return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
            case CUDA_ERROR_INVALID_SOURCE                 : return "CUDA_ERROR_INVALID_SOURCE";
            case CUDA_ERROR_FILE_NOT_FOUND                 : return "CUDA_ERROR_FILE_NOT_FOUND";
            case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND : return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
            case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      : return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
            case CUDA_ERROR_OPERATING_SYSTEM               : return "CUDA_ERROR_OPERATING_SYSTEM";
            case CUDA_ERROR_INVALID_HANDLE                 : return "CUDA_ERROR_INVALID_HANDLE";
            case CUDA_ERROR_NOT_FOUND                      : return "CUDA_ERROR_NOT_FOUND";
            case CUDA_ERROR_NOT_READY                      : return "CUDA_ERROR_NOT_READY";
            case CUDA_ERROR_ILLEGAL_ADDRESS                : return "CUDA_ERROR_ILLEGAL_ADDRESS";
            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        : return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
            case CUDA_ERROR_LAUNCH_TIMEOUT                 : return "CUDA_ERROR_LAUNCH_TIMEOUT";
            case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  : return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
            case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    : return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
            case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        : return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
            case CUDA_ERROR_PEER_MEMORY_ALREADY_REGISTERED : return "CUDA_ERROR_PEER_MEMORY_ALREADY_REGISTERED";
            case CUDA_ERROR_PEER_MEMORY_NOT_REGISTERED     : return "CUDA_ERROR_PEER_MEMORY_NOT_REGISTERED";
            case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         : return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
            case CUDA_ERROR_CONTEXT_IS_DESTROYED           : return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
            case CUDA_ERROR_ASSERT                         : return "CUDA_ERROR_ASSERT";
            case CUDA_ERROR_TOO_MANY_PEERS                 : return "CUDA_ERROR_TOO_MANY_PEERS";
            case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED : return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
            case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     : return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
            case CUDA_ERROR_HARDWARE_STACK_ERROR           : return "CUDA_ERROR_HARDWARE_STACK_ERROR";
            case CUDA_ERROR_ILLEGAL_INSTRUCTION            : return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
            case CUDA_ERROR_MISALIGNED_ADDRESS             : return "CUDA_ERROR_MISALIGNED_ADDRESS";
            case CUDA_ERROR_INVALID_ADDRESS_SPACE          : return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
            case CUDA_ERROR_INVALID_PC                     : return "CUDA_ERROR_INVALID_PC";
            case CUDA_ERROR_LAUNCH_FAILED                  : return "CUDA_ERROR_LAUNCH_FAILED";
            case CUDA_ERROR_NOT_PERMITTED                  : return "CUDA_ERROR_NOT_PERMITTED";
            case CUDA_ERROR_NOT_SUPPORTED                  : return "CUDA_ERROR_NOT_SUPPORTED";
            case CUDA_ERROR_UNKNOWN                        : return "CUDA_ERROR_UNKNOWN";
        }
        return "INVALID CUresult: "+result;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUresult()
    {
    }

}
