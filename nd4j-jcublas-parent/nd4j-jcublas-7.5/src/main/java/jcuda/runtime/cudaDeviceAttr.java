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

/**
 * CUDA device attributes
 */
public class cudaDeviceAttr
{

    /**
     * Maximum number of threads per block
     */
    public static final int cudaDevAttrMaxThreadsPerBlock             = 1;

    /**
     * Maximum block dimension X
     */
    public static final int cudaDevAttrMaxBlockDimX                   = 2;

    /**
     * Maximum block dimension Y
     */
    public static final int cudaDevAttrMaxBlockDimY                   = 3;

    /**
     * Maximum block dimension Z
     */
    public static final int cudaDevAttrMaxBlockDimZ                   = 4;

    /**
     * Maximum grid dimension X
     */
    public static final int cudaDevAttrMaxGridDimX                    = 5;

    /**
     * Maximum grid dimension Y
     */
    public static final int cudaDevAttrMaxGridDimY                    = 6;

    /**
     * Maximum grid dimension Z
     */
    public static final int cudaDevAttrMaxGridDimZ                    = 7;

    /**
     * Maximum shared memory available per block in bytes
     */
    public static final int cudaDevAttrMaxSharedMemoryPerBlock        = 8;

    /**
     * Memory available on device for __constant__ variables in a CUDA C kernel in bytes
     */
    public static final int cudaDevAttrTotalConstantMemory            = 9;

    /**
     * Warp size in threads
     */
    public static final int cudaDevAttrWarpSize                       = 10;

    /**
     * Maximum pitch in bytes allowed by memory copies
     */
    public static final int cudaDevAttrMaxPitch                       = 11;

    /**
     * Maximum number of 32-bit registers available per block
     */
    public static final int cudaDevAttrMaxRegistersPerBlock           = 12;

    /**
     * Peak clock frequency in kilohertz
     */
    public static final int cudaDevAttrClockRate                      = 13;

    /**
     * Alignment requirement for textures
     */
    public static final int cudaDevAttrTextureAlignment               = 14;

    /**
     * Device can possibly copy memory and execute a kernel concurrently
     */
    public static final int cudaDevAttrGpuOverlap                     = 15;

    /**
     * Number of multiprocessors on device
     */
    public static final int cudaDevAttrMultiProcessorCount            = 16;

    /**
     * Specifies whether there is a run time limit on kernels
     */
    public static final int cudaDevAttrKernelExecTimeout              = 17;

    /**
     * Device is integrated with host memory
     */
    public static final int cudaDevAttrIntegrated                     = 18;

    /**
     * Device can map host memory into CUDA address space
     */
    public static final int cudaDevAttrCanMapHostMemory               = 19;

    /**
     * Compute mode (See ::cudaComputeMode for details)
     */
    public static final int cudaDevAttrComputeMode                    = 20;

    /**
     * Maximum 1D texture width
     */
    public static final int cudaDevAttrMaxTexture1DWidth              = 21;

    /**
     * Maximum 2D texture width
     */
    public static final int cudaDevAttrMaxTexture2DWidth              = 22;

    /**
     * Maximum 2D texture height
     */
    public static final int cudaDevAttrMaxTexture2DHeight             = 23;

    /**
     * Maximum 3D texture width
     */
    public static final int cudaDevAttrMaxTexture3DWidth              = 24;

    /**
     * Maximum 3D texture height
     */
    public static final int cudaDevAttrMaxTexture3DHeight             = 25;

    /**
     * Maximum 3D texture depth
     */
    public static final int cudaDevAttrMaxTexture3DDepth              = 26;

    /**
     * Maximum 2D layered texture width
     */
    public static final int cudaDevAttrMaxTexture2DLayeredWidth       = 27;

    /**
     * Maximum 2D layered texture height
     */
    public static final int cudaDevAttrMaxTexture2DLayeredHeight      = 28;

    /**
     * Maximum layers in a 2D layered texture
     */
    public static final int cudaDevAttrMaxTexture2DLayeredLayers      = 29;

    /**
     * Alignment requirement for surfaces
     */
    public static final int cudaDevAttrSurfaceAlignment               = 30;

    /**
     * Device can possibly execute multiple kernels concurrently
     */
    public static final int cudaDevAttrConcurrentKernels              = 31;

    /**
     * Device has ECC support enabled
     */
    public static final int cudaDevAttrEccEnabled                     = 32;

    /**
     * PCI bus ID of the device
     */
    public static final int cudaDevAttrPciBusId                       = 33;

    /**
     * PCI device ID of the device
     */
    public static final int cudaDevAttrPciDeviceId                    = 34;

    /**
     * Device is using TCC driver model
     */
    public static final int cudaDevAttrTccDriver                      = 35;

    /**
     * Peak memory clock frequency in kilohertz
     */
    public static final int cudaDevAttrMemoryClockRate                = 36;

    /**
     * Global memory bus width in bits
     */
    public static final int cudaDevAttrGlobalMemoryBusWidth           = 37;

    /**
     * Size of L2 cache in bytes
     */
    public static final int cudaDevAttrL2CacheSize                    = 38;

    /**
     * Maximum resident threads per multiprocessor
     */
    public static final int cudaDevAttrMaxThreadsPerMultiProcessor    = 39;

    /**
     * Number of asynchronous engines
     */
    public static final int cudaDevAttrAsyncEngineCount               = 40;

    /**
     * Device shares a unified address space with the host */

    public static final int cudaDevAttrUnifiedAddressing              = 41;

    /**
     * Maximum 1D layered texture width
     */
    public static final int cudaDevAttrMaxTexture1DLayeredWidth       = 42;

    /**
     * Maximum layers in a 1D layered texture
     */
    public static final int cudaDevAttrMaxTexture1DLayeredLayers      = 43;

    /**
     * Maximum 2D texture width if cudaArrayTextureGather is set
     */
    public static final int cudaDevAttrMaxTexture2DGatherWidth        = 45;

    /**
     * Maximum 2D texture height if cudaArrayTextureGather is set
     */
    public static final int cudaDevAttrMaxTexture2DGatherHeight       = 46;

    /**
     * Alternate maximum 3D texture width
     */
    public static final int cudaDevAttrMaxTexture3DWidthAlt           = 47;

    /**
     * Alternate maximum 3D texture height
     */
    public static final int cudaDevAttrMaxTexture3DHeightAlt          = 48;

    /**
     * Alternate maximum 3D texture depth
     */
    public static final int cudaDevAttrMaxTexture3DDepthAlt           = 49;

    /**
     * PCI domain ID of the device
     */
    public static final int cudaDevAttrPciDomainId                    = 50;

    /**
     * Pitch alignment requirement for textures
     */
    public static final int cudaDevAttrTexturePitchAlignment          = 51;

    /**
     * Maximum cubemap texture width/height
     */
    public static final int cudaDevAttrMaxTextureCubemapWidth         = 52;

    /**
     * Maximum cubemap layered texture width/height
     */
    public static final int cudaDevAttrMaxTextureCubemapLayeredWidth  = 53;

    /**
     * Maximum layers in a cubemap layered texture
     */
    public static final int cudaDevAttrMaxTextureCubemapLayeredLayers = 54;

    /**
     * Maximum 1D surface width
     */
    public static final int cudaDevAttrMaxSurface1DWidth              = 55;

    /**
     * Maximum 2D surface width
     */
    public static final int cudaDevAttrMaxSurface2DWidth              = 56;

    /**
     * Maximum 2D surface height
     */
    public static final int cudaDevAttrMaxSurface2DHeight             = 57;

    /**
     * Maximum 3D surface width
     */
    public static final int cudaDevAttrMaxSurface3DWidth              = 58;

    /**
     * Maximum 3D surface height
     */
    public static final int cudaDevAttrMaxSurface3DHeight             = 59;

    /**
     * Maximum 3D surface depth
     */
    public static final int cudaDevAttrMaxSurface3DDepth              = 60;

    /**
     * Maximum 1D layered surface width
     */
    public static final int cudaDevAttrMaxSurface1DLayeredWidth       = 61;

    /**
     * Maximum layers in a 1D layered surface
     */
    public static final int cudaDevAttrMaxSurface1DLayeredLayers      = 62;

    /**
     * Maximum 2D layered surface width
     */
    public static final int cudaDevAttrMaxSurface2DLayeredWidth       = 63;

    /**
     * Maximum 2D layered surface height
     */
    public static final int cudaDevAttrMaxSurface2DLayeredHeight      = 64;

    /**
     * Maximum layers in a 2D layered surface
     */
    public static final int cudaDevAttrMaxSurface2DLayeredLayers      = 65;

    /**
     * Maximum cubemap surface width
     */
    public static final int cudaDevAttrMaxSurfaceCubemapWidth         = 66;

    /**
     * Maximum cubemap layered surface width
     */
    public static final int cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67;

    /**
     * Maximum layers in a cubemap layered surface
     */
    public static final int cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68;

    /**
     * Maximum 1D linear texture width
     */
    public static final int cudaDevAttrMaxTexture1DLinearWidth        = 69;

    /**
     * Maximum 2D linear texture width
     */
    public static final int cudaDevAttrMaxTexture2DLinearWidth        = 70;

    /**
     * Maximum 2D linear texture height
     */
    public static final int cudaDevAttrMaxTexture2DLinearHeight       = 71;

    /**
     * Maximum 2D linear texture pitch in bytes
     */
    public static final int cudaDevAttrMaxTexture2DLinearPitch        = 72;

    /**
     * Maximum mipmapped 2D texture width
     */
    public static final int cudaDevAttrMaxTexture2DMipmappedWidth     = 73;

    /**
     * Maximum mipmapped 2D texture height
     */
    public static final int cudaDevAttrMaxTexture2DMipmappedHeight    = 74;

    /**
     * Major compute capability version number
     */
    public static final int cudaDevAttrComputeCapabilityMajor         = 75;

    /**
     * Minor compute capability version number
     */
    public static final int cudaDevAttrComputeCapabilityMinor         = 76;

    /**
     * Maximum mipmapped 1D texture width
     */
    public static final int cudaDevAttrMaxTexture1DMipmappedWidth     = 77;

    /**
     * Device supports stream priorities
     */
    public static final int cudaDevAttrStreamPrioritiesSupported      = 78;

    /**
     * Device supports caching globals in L1
     */
    public static final int cudaDevAttrGlobalL1CacheSupported         = 79;

    /**
     * Device supports caching locals in L1
     */
    public static final int cudaDevAttrLocalL1CacheSupported          = 80;

    /**
     * Maximum shared memory available per multiprocessor in bytes
     */
    public static final int cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81;

    /**
     * Maximum number of 32-bit registers available per multiprocessor
     */
    public static final int cudaDevAttrMaxRegistersPerMultiprocessor  = 82;

    /**
     * Device can allocate managed memory on this system
     */
    public static final int cudaDevAttrManagedMemory                  = 83;

    /**
     * Device is on a multi-GPU board
     */
    public static final int cudaDevAttrIsMultiGpuBoard                = 84;

    /**
     * Unique identifier for a group of devices on the same multi-GPU board
     */
    public static final int cudaDevAttrMultiGpuBoardGroupID           = 85;

    /**
     * Returns the String identifying the given cudaDeviceAttr
     *
     * @param n The cudaDeviceAttr
     * @return The String identifying the given cudaDeviceAttr
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaDevAttrMaxThreadsPerBlock               : return "cudaDevAttrMaxThreadsPerBlock";
            case cudaDevAttrMaxBlockDimX                     : return "cudaDevAttrMaxBlockDimX";
            case cudaDevAttrMaxBlockDimY                     : return "cudaDevAttrMaxBlockDimY";
            case cudaDevAttrMaxBlockDimZ                     : return "cudaDevAttrMaxBlockDimZ";
            case cudaDevAttrMaxGridDimX                      : return "cudaDevAttrMaxGridDimX";
            case cudaDevAttrMaxGridDimY                      : return "cudaDevAttrMaxGridDimY";
            case cudaDevAttrMaxGridDimZ                      : return "cudaDevAttrMaxGridDimZ";
            case cudaDevAttrMaxSharedMemoryPerBlock          : return "cudaDevAttrMaxSharedMemoryPerBlock";
            case cudaDevAttrTotalConstantMemory              : return "cudaDevAttrTotalConstantMemory";
            case cudaDevAttrWarpSize                         : return "cudaDevAttrWarpSize";
            case cudaDevAttrMaxPitch                         : return "cudaDevAttrMaxPitch";
            case cudaDevAttrMaxRegistersPerBlock             : return "cudaDevAttrMaxRegistersPerBlock";
            case cudaDevAttrClockRate                        : return "cudaDevAttrClockRate";
            case cudaDevAttrTextureAlignment                 : return "cudaDevAttrTextureAlignment";
            case cudaDevAttrGpuOverlap                       : return "cudaDevAttrGpuOverlap";
            case cudaDevAttrMultiProcessorCount              : return "cudaDevAttrMultiProcessorCount";
            case cudaDevAttrKernelExecTimeout                : return "cudaDevAttrKernelExecTimeout";
            case cudaDevAttrIntegrated                       : return "cudaDevAttrIntegrated";
            case cudaDevAttrCanMapHostMemory                 : return "cudaDevAttrCanMapHostMemory";
            case cudaDevAttrComputeMode                      : return "cudaDevAttrComputeMode";
            case cudaDevAttrMaxTexture1DWidth                : return "cudaDevAttrMaxTexture1DWidth";
            case cudaDevAttrMaxTexture2DWidth                : return "cudaDevAttrMaxTexture2DWidth";
            case cudaDevAttrMaxTexture2DHeight               : return "cudaDevAttrMaxTexture2DHeight";
            case cudaDevAttrMaxTexture3DWidth                : return "cudaDevAttrMaxTexture3DWidth";
            case cudaDevAttrMaxTexture3DHeight               : return "cudaDevAttrMaxTexture3DHeight";
            case cudaDevAttrMaxTexture3DDepth                : return "cudaDevAttrMaxTexture3DDepth";
            case cudaDevAttrMaxTexture2DLayeredWidth         : return "cudaDevAttrMaxTexture2DLayeredWidth";
            case cudaDevAttrMaxTexture2DLayeredHeight        : return "cudaDevAttrMaxTexture2DLayeredHeight";
            case cudaDevAttrMaxTexture2DLayeredLayers        : return "cudaDevAttrMaxTexture2DLayeredLayers";
            case cudaDevAttrSurfaceAlignment                 : return "cudaDevAttrSurfaceAlignment";
            case cudaDevAttrConcurrentKernels                : return "cudaDevAttrConcurrentKernels";
            case cudaDevAttrEccEnabled                       : return "cudaDevAttrEccEnabled";
            case cudaDevAttrPciBusId                         : return "cudaDevAttrPciBusId";
            case cudaDevAttrPciDeviceId                      : return "cudaDevAttrPciDeviceId";
            case cudaDevAttrTccDriver                        : return "cudaDevAttrTccDriver";
            case cudaDevAttrMemoryClockRate                  : return "cudaDevAttrMemoryClockRate";
            case cudaDevAttrGlobalMemoryBusWidth             : return "cudaDevAttrGlobalMemoryBusWidth";
            case cudaDevAttrL2CacheSize                      : return "cudaDevAttrL2CacheSize";
            case cudaDevAttrMaxThreadsPerMultiProcessor      : return "cudaDevAttrMaxThreadsPerMultiProcessor";
            case cudaDevAttrAsyncEngineCount                 : return "cudaDevAttrAsyncEngineCount";
            case cudaDevAttrUnifiedAddressing                : return "cudaDevAttrUnifiedAddressing";
            case cudaDevAttrMaxTexture1DLayeredWidth         : return "cudaDevAttrMaxTexture1DLayeredWidth";
            case cudaDevAttrMaxTexture1DLayeredLayers        : return "cudaDevAttrMaxTexture1DLayeredLayers";
            case cudaDevAttrMaxTexture2DGatherWidth          : return "cudaDevAttrMaxTexture2DGatherWidth";
            case cudaDevAttrMaxTexture2DGatherHeight         : return "cudaDevAttrMaxTexture2DGatherHeight";
            case cudaDevAttrMaxTexture3DWidthAlt             : return "cudaDevAttrMaxTexture3DWidthAlt";
            case cudaDevAttrMaxTexture3DHeightAlt            : return "cudaDevAttrMaxTexture3DHeightAlt";
            case cudaDevAttrMaxTexture3DDepthAlt             : return "cudaDevAttrMaxTexture3DDepthAlt";
            case cudaDevAttrPciDomainId                      : return "cudaDevAttrPciDomainId";
            case cudaDevAttrTexturePitchAlignment            : return "cudaDevAttrTexturePitchAlignment";
            case cudaDevAttrMaxTextureCubemapWidth           : return "cudaDevAttrMaxTextureCubemapWidth";
            case cudaDevAttrMaxTextureCubemapLayeredWidth    : return "cudaDevAttrMaxTextureCubemapLayeredWidth";
            case cudaDevAttrMaxTextureCubemapLayeredLayers   : return "cudaDevAttrMaxTextureCubemapLayeredLayers";
            case cudaDevAttrMaxSurface1DWidth                : return "cudaDevAttrMaxSurface1DWidth";
            case cudaDevAttrMaxSurface2DWidth                : return "cudaDevAttrMaxSurface2DWidth";
            case cudaDevAttrMaxSurface2DHeight               : return "cudaDevAttrMaxSurface2DHeight";
            case cudaDevAttrMaxSurface3DWidth                : return "cudaDevAttrMaxSurface3DWidth";
            case cudaDevAttrMaxSurface3DHeight               : return "cudaDevAttrMaxSurface3DHeight";
            case cudaDevAttrMaxSurface3DDepth                : return "cudaDevAttrMaxSurface3DDepth";
            case cudaDevAttrMaxSurface1DLayeredWidth         : return "cudaDevAttrMaxSurface1DLayeredWidth";
            case cudaDevAttrMaxSurface1DLayeredLayers        : return "cudaDevAttrMaxSurface1DLayeredLayers";
            case cudaDevAttrMaxSurface2DLayeredWidth         : return "cudaDevAttrMaxSurface2DLayeredWidth";
            case cudaDevAttrMaxSurface2DLayeredHeight        : return "cudaDevAttrMaxSurface2DLayeredHeight";
            case cudaDevAttrMaxSurface2DLayeredLayers        : return "cudaDevAttrMaxSurface2DLayeredLayers";
            case cudaDevAttrMaxSurfaceCubemapWidth           : return "cudaDevAttrMaxSurfaceCubemapWidth";
            case cudaDevAttrMaxSurfaceCubemapLayeredWidth    : return "cudaDevAttrMaxSurfaceCubemapLayeredWidth";
            case cudaDevAttrMaxSurfaceCubemapLayeredLayers   : return "cudaDevAttrMaxSurfaceCubemapLayeredLayers";
            case cudaDevAttrMaxTexture1DLinearWidth          : return "cudaDevAttrMaxTexture1DLinearWidth";
            case cudaDevAttrMaxTexture2DLinearWidth          : return "cudaDevAttrMaxTexture2DLinearWidth";
            case cudaDevAttrMaxTexture2DLinearHeight         : return "cudaDevAttrMaxTexture2DLinearHeight";
            case cudaDevAttrMaxTexture2DLinearPitch          : return "cudaDevAttrMaxTexture2DLinearPitch";
            case cudaDevAttrMaxTexture2DMipmappedWidth       : return "cudaDevAttrMaxTexture2DMipmappedWidth";
            case cudaDevAttrMaxTexture2DMipmappedHeight      : return "cudaDevAttrMaxTexture2DMipmappedHeight";
            case cudaDevAttrComputeCapabilityMajor           : return "cudaDevAttrComputeCapabilityMajor";
            case cudaDevAttrComputeCapabilityMinor           : return "cudaDevAttrComputeCapabilityMinor";
            case cudaDevAttrMaxTexture1DMipmappedWidth       : return "cudaDevAttrMaxTexture1DMipmappedWidth";
            case cudaDevAttrStreamPrioritiesSupported        : return "cudaDevAttrStreamPrioritiesSupported";
            case cudaDevAttrGlobalL1CacheSupported           : return "cudaDevAttrGlobalL1CacheSupported";
            case cudaDevAttrLocalL1CacheSupported            : return "cudaDevAttrLocalL1CacheSupported";
            case cudaDevAttrMaxSharedMemoryPerMultiprocessor : return "cudaDevAttrMaxSharedMemoryPerMultiprocessor";
            case cudaDevAttrMaxRegistersPerMultiprocessor    : return "cudaDevAttrMaxRegistersPerMultiprocessor";
            case cudaDevAttrManagedMemory                    : return "cudaDevAttrManagedMemory";
            case cudaDevAttrIsMultiGpuBoard                  : return "cudaDevAttrIsMultiGpuBoard";
            case cudaDevAttrMultiGpuBoardGroupID             : return "cudaDevAttrMultiGpuBoardGroupID";
        }
        return "INVALID cudaDeviceAttr: "+n;
    }


    /**
     * Private constructor to prevent instantiation
     */
    private cudaDeviceAttr()
    {

    }
}
