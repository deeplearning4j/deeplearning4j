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

import jcuda.driver.CUcontext;

/**
 * Error codes. The documentation is extracted from the CUDA header files.
 */
public class cudaError
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * {@link JCuda#cudaEventQuery} and {@link JCuda#cudaStreamQuery}).
     */
    public static final int cudaSuccess                           =      0;

    /**
     * The device function being invoked (usually via {@link JCuda#cudaLaunch}) was not
     * previously configured via the {@link JCuda#cudaConfigureCall} function.
     */
    public static final int cudaErrorMissingConfiguration         =      1;

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    public static final int cudaErrorMemoryAllocation             =      2;

    /**
     * The API call failed because the CUDA driver and runtime could not be
     * initialized.
     */
    public static final int cudaErrorInitializationError          =      3;

    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The device cannot be used until
     * {@link JCuda#cudaThreadExit} is called. All existing device memory allocations
     * are invalid and must be reconstructed if the program is to continue
     * using CUDA.
     */
    public static final int cudaErrorLaunchFailure                =      4;

    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorPriorLaunchFailure           =      5;

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device property
     * {@link cudaDeviceProp#kernelExecTimeoutEnabled} "kernelExecTimeoutEnabled"
     * for more information. The device cannot be used until {@link JCuda#cudaThreadExit}
     * is called. All existing device memory allocations are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    public static final int cudaErrorLaunchTimeout                =      6;

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * {@link cudaError#cudaErrorInvalidConfiguration}this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    public static final int cudaErrorLaunchOutOfResources         =      7;

    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    public static final int cudaErrorInvalidDeviceFunction        =      8;

    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See {@link cudaDeviceProp }
     * for more device limitations.
     */
    public static final int cudaErrorInvalidConfiguration         =      9;

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    public static final int cudaErrorInvalidDevice                =     10;

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    public static final int cudaErrorInvalidValue                 =     11;

    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    public static final int cudaErrorInvalidPitchValue            =     12;

    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    public static final int cudaErrorInvalidSymbol                =     13;

    /**
     * This indicates that the buffer object could not be mapped.
     */
    public static final int cudaErrorMapBufferObjectFailed        =     14;

    /**
     * This indicates that the buffer object could not be unmapped.
     */
    public static final int cudaErrorUnmapBufferObjectFailed      =     15;

    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     */
    public static final int cudaErrorInvalidHostPointer           =     16;

    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     */
    public static final int cudaErrorInvalidDevicePointer         =     17;

    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    public static final int cudaErrorInvalidTexture               =     18;

    /**
     * This indicates that the texture binding is not valid. This occurs if you
     * call {@link JCuda#cudaGetTextureAlignmentOffset} with an unbound texture.
     */
    public static final int cudaErrorInvalidTextureBinding        =     19;

    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This occurs if the format is not one of the formats specified by
     * {@link cudaChannelFormatKind}or if one of the dimensions is invalid.
     */
    public static final int cudaErrorInvalidChannelDescriptor     =     20;

    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by {@link cudaMemcpyKind}   */
    public static final int cudaErrorInvalidMemcpyDirection       =     21;

    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the CUDA 3.1 release.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * {@link JCuda#cudaGetSymbolAddress}.
     */
    public static final int cudaErrorAddressOfConstant            =     22;

    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorTextureFetchFailed           =     23;

    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorTextureNotBound              =     24;

    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorSynchronizationError         =     25;

    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by CUDA.
     */
    public static final int cudaErrorInvalidFilterSetting         =     26;

    /**
     * This indicates that an attempt was made to read a non-float texture as a
     * normalized float. This is not supported by CUDA.
     */
    public static final int cudaErrorInvalidNormSetting           =     27;

    /**
     * Mixing of device and device emulation code was not allowed.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorMixedDeviceExecution         =     28;

    /**
     * This indicates that a CUDA Runtime API call cannot be executed because
     * it is being called during process shut down, at a point in time after
     * CUDA driver has been unloaded.
     */
    public static final int cudaErrorCudartUnloading              =     29;

    /**
     * This indicates that an unknown internal error has occurred.
     */
    public static final int cudaErrorUnknown                      =     30;

    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of CUDA will never return this error.
     * @deprecated This error return is deprecated as of CUDA 4.1.
     */
    public static final int cudaErrorNotYetImplemented            =     31;

    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * @deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    public static final int cudaErrorMemoryValueTooLarge          =     32;

    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like {@link cudaStream_t } or
     * {@link cudaEvent_t}
     */
    public static final int cudaErrorInvalidResourceHandle        =     33;

    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than {@link cudaError#cudaSuccess } which indicates completion). Calls that
     * may return this value include {@link JCuda#cudaEventQuery} and {@link JCuda#cudaStreamQuery}.
     */
    public static final int cudaErrorNotReady                     =     34;

    /**
     * This indicates that the installed NVIDIA CUDA driver is older than the
     * CUDA runtime library. This is not a supported configuration. Users should
     * install an updated NVIDIA display driver to allow the application to run.
     */
    public static final int cudaErrorInsufficientDriver           =     35;

    /**
     * This indicates that the user has called
     * {@link JCuda#cudaSetValidDevices(int[], int)},
     * {@link JCuda#cudaSetDeviceFlags(int)},
     * after initializing the CUDA runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing {@link CUcontext}
     * active on the host thread.
     */
    public static final int cudaErrorSetOnActiveProcess           =     36;

    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    public static final int cudaErrorInvalidSurface               =     37;

    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    public static final int cudaErrorNoDevice                     =     38;

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    public static final int cudaErrorECCUncorrectable             =     39;

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    public static final int cudaErrorSharedObjectSymbolNotFound   =     40;

    /**
     * This indicates that initialization of a shared object failed.
     */
    public static final int cudaErrorSharedObjectInitFailed       =     41;

    /**
     * This indicates that the ::cudaLimit passed to the API call is not
     * supported by the active device.
     */
    public static final int cudaErrorUnsupportedLimit             =     42;

    /**
     * This indicates that multiple global or constant variables (across separate
     * CUDA source files in the application) share the same string name.
     */
    public static final int cudaErrorDuplicateVariableName        =     43;

    /**
     * This indicates that multiple textures (across separate CUDA source
     * files in the application) share the same string name.
     */
    public static final int cudaErrorDuplicateTextureName         =     44;

    /**
     * This indicates that multiple surfaces (across separate CUDA source
     * files in the application) share the same string name.
     */
    public static final int cudaErrorDuplicateSurfaceName         =     45;

    /**
     * This indicates that all CUDA devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * {@link cudaComputeMode#cudaComputeModeExclusive }
     * {@link cudaComputeMode#cudaComputeModeProhibited}. They can also
     * be unavailable due to memory constraints on a device that already has
     * active CUDA work being performed.
     */
    public static final int cudaErrorDevicesUnavailable           =     46;

    /**
     * This indicates that the device kernel image is invalid.
     */
    public static final int cudaErrorInvalidKernelImage           =     47;

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    public static final int cudaErrorNoKernelImageForDevice       =     48;

    /**
     * This indicates that the current context is not compatible with this
     * the CUDA Runtime. This can only occur if you are using CUDA
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version
     * of the API, because the Runtime API call expects a primary driver
     * contextand the Driver context is not primary, or because the Driver
     * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions
     * with the CUDA Driver API" for more information.
     */
    public static final int cudaErrorIncompatibleDriverContext    =     49;

    /**
     * This error indicates that a call to {@link JCuda#cudaDeviceEnablePeerAccess} is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    public static final int cudaErrorPeerAccessAlreadyEnabled     =     50;

    /**
     * This error indicates that a call to {@link JCuda#cudaDeviceEnablePeerAccess } trying to
     * register memory from a context which has not had peer addressing
     * enabled yet via {@link JCuda#cudaDeviceEnablePeerAccess}, or that
     * {@link JCuda#cudaDeviceDisablePeerAccess} is trying to disable peer addressing
     * which has not been enabled yet.
     */
    public static final int cudaErrorPeerAccessNotEnabled         =     51;

    /**
     * This indicates that a call tried to access an exclusive-thread device that
     * is already in use by a different thread.
     */
    public static final int cudaErrorDeviceAlreadyInUse           =     54;

    /**
     * This indicates profiler has been disabled for this run and thus runtime
     * APIs cannot be used to profile subsets of the program. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    public static final int cudaErrorProfilerDisabled             =     55;

    /**
     * This indicates profiler has not been initialized yet. cudaProfilerInitialize()
     * must be called before calling cudaProfilerStart and cudaProfilerStop to
     * initialize profiler.
     *
     * @deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via {@link JCuda#cudaProfilerStart()} or
     *  {@link JCuda#cudaProfilerStop()} without initialization.
     */
    public static final int cudaErrorProfilerNotInitialized       =     56;

    /**
     * This indicates profiler is already started. This error can be returned if
     * cudaProfilerStart() is called multiple times without subsequent call
     * to cudaProfilerStop().
     *
     * @deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call  {@link JCuda#cudaProfilerStart()} when profiling is already enabled.
     */
    public static final int cudaErrorProfilerAlreadyStarted       =     57;

    /**
     * This indicates profiler is already stopped. This error can be returned if
     * cudaProfilerStop() is called without starting profiler using cudaProfilerStart().
     *
     * @deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call  {@link JCuda#cudaProfilerStop()} when profiling is already disabled.
     */
    public static final int cudaErrorProfilerAlreadyStopped       =     58;

    /**
     * An assert triggered in device code during kernel execution. The device
     * cannot be used again until {@link JCuda#cudaThreadExit()} is called. All existing
     * allocations are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    public static final int cudaErrorAssert                        =    59;

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices
     * passed to ::cudaEnablePeerAccess().
     */
    public static final int cudaErrorTooManyPeers                 =     60;

    /**
     * This error indicates that the memory range passed to
     * {@link JCuda#cudaHostRegister(jcuda.Pointer, long, int)}
     * has already been registered.
     */
    public static final int cudaErrorHostMemoryAlreadyRegistered  =     61;

    /**
     * This error indicates that the pointer passed to
     * {@link JCuda#cudaHostUnregister(jcuda.Pointer)}
     * does not correspond to any currently registered memory region.
     */
    public static final int cudaErrorHostMemoryNotRegistered      =     62;

    /**
     * This error indicates that an OS call failed.
     */
    public static final int cudaErrorOperatingSystem              =     63;

    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    public static final int cudaErrorPeerAccessUnsupported        =     64;

    /**
     * This error indicates that a device runtime grid launch did not occur
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches.
     */
    public static final int cudaErrorLaunchMaxDepthExceeded       =     65;

    /**
     * This error indicates that a grid launch did not occur because the kernel
     * uses file-scoped textures which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support textures created with
     * the Texture Object API's.
     */
    public static final int cudaErrorLaunchFileScopedTex          =     66;

    /**
     * This error indicates that a grid launch did not occur because the kernel
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    public static final int cudaErrorLaunchFileScopedSurf         =     67;

    /**
     * This error indicates that a call to ::cudaDeviceSynchronize made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device
     * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on
     * launched grids at a greater depth successfully, the maximum nested
     * depth at which ::cudaDeviceSynchronize will be called must be specified
     * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
     * api before the host-side launch of a kernel using the device runtime.
     * Keep in mind that additional levels of sync depth require the runtime
     * to reserve large amounts of device memory that cannot be used for
     * user allocations.
     */
    public static final int cudaErrorSyncDepthExceeded            =     68;

    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
     * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    public static final int cudaErrorLaunchPendingCountExceeded   =     69;

    /**
     * This error indicates the attempted operation is not permitted.
     */
    public static final int cudaErrorNotPermitted                 =     70;

    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    public static final int cudaErrorNotSupported                 =     71;

    /**
     * Device encountered an error in the call stack during kernel execution,
     * possibly due to stack corruption or exceeding the stack size limit.
     * The context cannot be used, so it must be destroyed (and a new one
     * should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int cudaErrorHardwareStackError           =     72;

    /**
     * The device encountered an illegal instruction during kernel execution
     * The context cannot be used, so it must be destroyed (and a new one
     * should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int cudaErrorIllegalInstruction           =     73;

    /**
     * The device encountered a load or store instruction
     * on a memory address which is not aligned.
     * The context cannot be used, so it must be destroyed (and a new one
     * should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int cudaErrorMisalignedAddress            =     74;

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * The context cannot be used, so it must be destroyed (and a new one
     * should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int cudaErrorInvalidAddressSpace          =     75;

    /**
     * The device encountered an invalid program counter.
     * The context cannot be used, so it must be destroyed (and a new one
     * should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int cudaErrorInvalidPc                    =     76;

    /**
     * The device encountered a load or store instruction on an invalid
     * memory address.
     * The context cannot be used, so it must be destroyed (and a new one
     * should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    public static final int cudaErrorIllegalAddress               =     77;

    /**
     * A PTX compilation failed. The runtime may fall back to compiling PTX if
     * an application does not contain a suitable binary for the current device.
     */
    public static final int cudaErrorInvalidPtx                   =     78;

    /**
     * This indicates an error with the OpenGL or DirectX context.
     */
    public static final int cudaErrorInvalidGraphicsContext       =     79;


    /**
     * This indicates an internal startup failure in the CUDA runtime.
     */
    public static final int cudaErrorStartupFailure               =   0x7f;

    /**
     * Any unhandled CUDA driver error is added to this value and returned via
     * the runtime. Production releases of CUDA should not return such errors.
     * @deprecated This error return is deprecated as of CUDA 4.1.
     */
    public static final int cudaErrorApiFailureBase               =  10000;

    /**
     * An internal JCuda error occurred
     */
    public static final int jcudaInternalError = 0x80000001;

    /**
     * Returns the String identifying the given cudaError
     *
     * @param error The cudaError
     * @return The String identifying the given cudaError
     */
    public static String stringFor(int error)
    {
        switch (error)
        {
            case cudaSuccess                           : return "cudaSuccess";
            case cudaErrorMissingConfiguration         : return "cudaErrorMissingConfiguration";
            case cudaErrorMemoryAllocation             : return "cudaErrorMemoryAllocation";
            case cudaErrorInitializationError          : return "cudaErrorInitializationError";
            case cudaErrorLaunchFailure                : return "cudaErrorLaunchFailure";
            case cudaErrorPriorLaunchFailure           : return "cudaErrorPriorLaunchFailure";
            case cudaErrorLaunchTimeout                : return "cudaErrorLaunchTimeout";
            case cudaErrorLaunchOutOfResources         : return "cudaErrorLaunchOutOfResources";
            case cudaErrorInvalidDeviceFunction        : return "cudaErrorInvalidDeviceFunction";
            case cudaErrorInvalidConfiguration         : return "cudaErrorInvalidConfiguration";
            case cudaErrorInvalidDevice                : return "cudaErrorInvalidDevice";
            case cudaErrorInvalidValue                 : return "cudaErrorInvalidValue";
            case cudaErrorInvalidPitchValue            : return "cudaErrorInvalidPitchValue";
            case cudaErrorInvalidSymbol                : return "cudaErrorInvalidSymbol";
            case cudaErrorMapBufferObjectFailed        : return "cudaErrorMapBufferObjectFailed";
            case cudaErrorUnmapBufferObjectFailed      : return "cudaErrorUnmapBufferObjectFailed";
            case cudaErrorInvalidHostPointer           : return "cudaErrorInvalidHostPointer";
            case cudaErrorInvalidDevicePointer         : return "cudaErrorInvalidDevicePointer";
            case cudaErrorInvalidTexture               : return "cudaErrorInvalidTexture";
            case cudaErrorInvalidTextureBinding        : return "cudaErrorInvalidTextureBinding";
            case cudaErrorInvalidChannelDescriptor     : return "cudaErrorInvalidChannelDescriptor";
            case cudaErrorInvalidMemcpyDirection       : return "cudaErrorInvalidMemcpyDirection";
            case cudaErrorAddressOfConstant            : return "cudaErrorAddressOfConstant";
            case cudaErrorTextureFetchFailed           : return "cudaErrorTextureFetchFailed";
            case cudaErrorTextureNotBound              : return "cudaErrorTextureNotBound";
            case cudaErrorSynchronizationError         : return "cudaErrorSynchronizationError";
            case cudaErrorInvalidFilterSetting         : return "cudaErrorInvalidFilterSetting";
            case cudaErrorInvalidNormSetting           : return "cudaErrorInvalidNormSetting";
            case cudaErrorMixedDeviceExecution         : return "cudaErrorMixedDeviceExecution";
            case cudaErrorCudartUnloading              : return "cudaErrorCudartUnloading";
            case cudaErrorUnknown                      : return "cudaErrorUnknown";
            case cudaErrorNotYetImplemented            : return "cudaErrorNotYetImplemented";
            case cudaErrorMemoryValueTooLarge          : return "cudaErrorMemoryValueTooLarge";
            case cudaErrorInvalidResourceHandle        : return "cudaErrorInvalidResourceHandle";
            case cudaErrorNotReady                     : return "cudaErrorNotReady";
            case cudaErrorInsufficientDriver           : return "cudaErrorInsufficientDriver";
            case cudaErrorSetOnActiveProcess           : return "cudaErrorSetOnActiveProcess";
            case cudaErrorInvalidSurface               : return "cudaErrorInvalidSurface";
            case cudaErrorNoDevice                     : return "cudaErrorNoDevice";
            case cudaErrorECCUncorrectable             : return "cudaErrorECCUncorrectable";
            case cudaErrorSharedObjectSymbolNotFound   : return "cudaErrorSharedObjectSymbolNotFound";
            case cudaErrorSharedObjectInitFailed       : return "cudaErrorSharedObjectInitFailed";
            case cudaErrorUnsupportedLimit             : return "cudaErrorUnsupportedLimit";
            case cudaErrorDuplicateVariableName        : return "cudaErrorDuplicateVariableName";
            case cudaErrorDuplicateTextureName         : return "cudaErrorDuplicateTextureName";
            case cudaErrorDuplicateSurfaceName         : return "cudaErrorDuplicateSurfaceName";
            case cudaErrorDevicesUnavailable           : return "cudaErrorDevicesUnavailable";
            case cudaErrorInvalidKernelImage           : return "cudaErrorInvalidKernelImage";
            case cudaErrorNoKernelImageForDevice       : return "cudaErrorNoKernelImageForDevice";
            case cudaErrorIncompatibleDriverContext    : return "cudaErrorIncompatibleDriverContext";
            case cudaErrorPeerAccessAlreadyEnabled     : return "cudaErrorPeerAccessAlreadyEnabled";
            case cudaErrorPeerAccessNotEnabled         : return "cudaErrorPeerAccessNotEnabled";
            case cudaErrorDeviceAlreadyInUse           : return "cudaErrorDeviceAlreadyInUse";
            case cudaErrorProfilerDisabled             : return "cudaErrorProfilerDisabled";
            case cudaErrorProfilerNotInitialized       : return "cudaErrorProfilerNotInitialized";
            case cudaErrorProfilerAlreadyStarted       : return "cudaErrorProfilerAlreadyStarted";
            case cudaErrorProfilerAlreadyStopped       : return "cudaErrorProfilerAlreadyStopped";
            case cudaErrorAssert                       : return "cudaErrorAssert";
            case cudaErrorTooManyPeers                 : return "cudaErrorTooManyPeers";
            case cudaErrorHostMemoryAlreadyRegistered  : return "cudaErrorHostMemoryAlreadyRegistered";
            case cudaErrorHostMemoryNotRegistered      : return "cudaErrorHostMemoryNotRegistered";
            case cudaErrorOperatingSystem              : return "cudaErrorOperatingSystem";
            case cudaErrorPeerAccessUnsupported        : return "cudaErrorPeerAccessUnsupported";
            case cudaErrorLaunchMaxDepthExceeded       : return "cudaErrorLaunchMaxDepthExceeded";
            case cudaErrorLaunchFileScopedTex          : return "cudaErrorLaunchFileScopedTex";
            case cudaErrorLaunchFileScopedSurf         : return "cudaErrorLaunchFileScopedSurf";
            case cudaErrorSyncDepthExceeded            : return "cudaErrorSyncDepthExceeded";
            case cudaErrorLaunchPendingCountExceeded   : return "cudaErrorLaunchPendingCountExceeded";
            case cudaErrorNotPermitted                 : return "cudaErrorNotPermitted";
            case cudaErrorNotSupported                 : return "cudaErrorNotSupported";
            case cudaErrorHardwareStackError           : return "cudaErrorHardwareStackError";
            case cudaErrorIllegalInstruction           : return "cudaErrorIllegalInstruction";
            case cudaErrorMisalignedAddress            : return "cudaErrorMisalignedAddress";
            case cudaErrorInvalidAddressSpace          : return "cudaErrorInvalidAddressSpace";
            case cudaErrorInvalidPc                    : return "cudaErrorInvalidPc";
            case cudaErrorIllegalAddress               : return "cudaErrorIllegalAddress";
            case cudaErrorInvalidPtx                   : return "cudaErrorInvalidPtx";
            case cudaErrorInvalidGraphicsContext       : return "cudaErrorInvalidGraphicsContext";
            case cudaErrorStartupFailure               : return "cudaErrorStartupFailure";
            case jcudaInternalError                    : return "jcudaInternalError";
        }
        if (error >= cudaErrorApiFailureBase)
        {
            return stringFor(error-cudaErrorApiFailureBase);
        }
        return "INVALID cudaError: "+error;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaError()
    {
    }

};
