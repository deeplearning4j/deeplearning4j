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
import java.util.*;

/**
 * Java port of the cudaDeviceProp.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see JCuda#cudaGetDeviceProperties
 */
public class cudaDeviceProp
{
    /**
     * Creates a new, uninitialized cudaDeviceProp object
     */
    public cudaDeviceProp()
    {
    }

    /**
     * An ASCII string identifying the device;
     */
    public byte name[] = new byte[256];

    /**
     * The total amount of global memory available on the device in bytes;
     */
    public long totalGlobalMem;

    /**
     * The maximum amount of shared memory available to a thread block in bytes;
     * this amount is shared by all thread blocks simultaneously resident on a multiprocessor;
     */
    public long sharedMemPerBlock;

    /**
     * The maximum number of 32-bit registers available to a thread block;
     * this number is shared by all thread blocks simultaneously resident on a multiprocessor;
     */
    public int regsPerBlock;

    /**
     * The warp size in threads;
     */
    public int warpSize;

    /**
     * The maximum pitch in bytes allowed by the memory copy functions that
     * involve memory regions allocated through cudaMallocPitch();
     */
    public long memPitch;

    /**
     * The maximum number of threads per block;
     */
    public int maxThreadsPerBlock;

    /**
     * The maximum sizes of each dimension of a block;
     */
    public int maxThreadsDim[] = new int[3];

    /**
     * The maximum sizes of each dimension of a grid;
     */
    public int maxGridSize[] = new int[3];

    /**
     * The clock frequency in kilohertz;
     */
    public int clockRate;

    /**
     * The total amount of constant memory available on the device in bytes;
     */
    public long totalConstMem;

    /**
     * Major revision number defining the device's compute capability;
     */
    public int major;

    /**
     * Minor revision number defining the device's compute capability;
     */
    public int minor;

    /**
     * The alignment requirement; texture base addresses that are aligned to textureAlignment
     * bytes do not need an offset applied to texture fetches;
     */
    public long textureAlignment;



    /**
     * Pitch alignment requirement for texture references bound to pitched
     * memory
     */
    public long texturePitchAlignment;

    /**
     * Device can concurrently copy memory and execute a kernel. Deprecated. Use
     * instead asyncEngineCount.
     */
    public int deviceOverlap;

    /**
     * Number of multiprocessors on device
     */
    public int multiProcessorCount;

    /**
     * Specified whether there is a run time limit on kernels
     */
    public int kernelExecTimeoutEnabled;

    /**
     * Device is integrated as opposed to discrete
     */
    public int integrated;

    /**
     * Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
     */
    public int canMapHostMemory;

    /**
     * Compute mode (See ::cudaComputeMode)
     */
    public int computeMode;

    /**
     * Maximum 1D texture size
     */
    public int maxTexture1D;

    /**
     * Maximum 1D mipmapped texture size
     */
    public int maxTexture1DMipmap;

    /**
     * Maximum size for 1D textures bound to linear memory
     */
    public int maxTexture1DLinear;

    /**
     * Maximum 2D texture dimensions
     */
    public int maxTexture2D[] = new int[2];

    /**
     * Maximum 2D mipmapped texture dimensions
     */
    public int maxTexture2DMipmap[] = new int[2];

    /**
     * Maximum dimensions (width, height, pitch) for 2D textures bound to
     * pitched memory
     */
    public int maxTexture2DLinear[] = new int[3];

    /**
     * Maximum 2D texture dimensions if texture gather operations have to be
     * performed
     */
    public int maxTexture2DGather[] = new int[2];

    /**
     * Maximum 3D texture dimensions
     */
    public int maxTexture3D[] = new int[3];

    /**
     * Contains the maximum alternate 3D texture dimensions
     */
    public int maxTexture3DAlt[] = new int[3];

    /**
     * Maximum Cubemap texture dimensions
     */
    public int maxTextureCubemap;

    /**
     * Maximum 1D layered texture dimensions
     */
    public int maxTexture1DLayered[] = new int[2];

    /**
     * Maximum 2D layered texture dimensions
     */
    public int maxTexture2DLayered[] = new int[3];

    /**
     * Maximum Cubemap layered texture dimensions
     */
    public int maxTextureCubemapLayered[] = new int[2];

    /**
     * Maximum 1D surface size
     */
    public int maxSurface1D;

    /**
     * Maximum 2D surface dimensions
     */
    public int maxSurface2D[] = new int[2];

    /**
     * Maximum 3D surface dimensions
     */
    public int maxSurface3D[] = new int[3];

    /**
     * Maximum 1D layered surface dimensions
     */
    public int maxSurface1DLayered[] = new int[2];

    /**
     * Maximum 2D layered surface dimensions
     */
    public int maxSurface2DLayered[] = new int[3];

    /**
     * Maximum Cubemap surface dimensions
     */
    public int maxSurfaceCubemap;

    /**
     * Maximum Cubemap layered surface dimensions
     */
    public int maxSurfaceCubemapLayered[] = new int[2];

    /**
     * Alignment requirements for surfaces
     */
    public long surfaceAlignment;

    /**
     * Device can possibly execute multiple kernels concurrently
     */
    public int concurrentKernels;

    /**
     * Device has ECC support enabled
     */
    public int    ECCEnabled;

    /**
     * PCI bus ID of the device
     */
    public int    pciBusID;

    /**
     * PCI device ID of the device
     */
    public int    pciDeviceID;

    /**
     * PCI domain ID of the device
     */
    public int pciDomainID;

    /**
     *  1 if device is a Tesla device using TCC driver, 0 otherwise
     */
    public int    tccDriver;

    /**
     *  1 when the device can concurrently copy memory between host and device
     *  while executing a kernel. It is 2 when the device can concurrently
     *  copy memory between host and device in both directions and execute a
     *  kernel at the same time. It is 0 if neither of these is supported.
     */
    public int    asyncEngineCount;

    /**
     *  1 if the device shares a unified address space with the host and 0 otherwise.
     */
    public int    unifiedAddressing;

    /**
     * The peak memory clock frequency in kilohertz.
     */
    public int memoryClockRate;

    /**
     * The memory bus width in bits
     */
    public int memoryBusWidth;

    /**
     * L2 cache size in bytes
     */
    public int l2CacheSize;

    /**
     * The number of maximum resident threads per multiprocessor.
     */
    public int maxThreadsPerMultiProcessor;

    /**
     * Is 1 if the device supports stream priorities,
     * or 0 if it is not supported
     */
    public int streamPrioritiesSupported;

    /**
     * Device supports caching globals in L1
     */
    public int globalL1CacheSupported;

    /**
     * Device supports caching locals in L1
     */
    public int localL1CacheSupported;

    /**
     * Shared memory available per multiprocessor in bytes
     */
    public long sharedMemPerMultiprocessor;

    /**
     * 32-bit registers available per multiprocessor
     */
    public int regsPerMultiprocessor;

    /**
     * Device supports allocating managed memory on this system
     */
    public int managedMemory;

    /**
     * Device is on a multi-GPU board
     */
    public int isMultiGpuBoard;

    /**
     * Unique identifier for a group of devices on the same multi-GPU board
     */
    public int multiGpuBoardGroupID;

    /**
     * Returns the String describing the name of this cudaDeviceProp
     *
     * @return String The String describing the name of this cudaDeviceProp
     */
    public String getName()
    {
        return createString(name);
    }

    /**
     * Set the name of this cudaDeviceProp to the given name
     *
     * @param nameString The name for this cudaDeviceProp
     */
    public void setName(String nameString)
    {
        byte bytes[] = nameString.getBytes();
        int n = Math.min(name.length, bytes.length);
        System.arraycopy(bytes, 0, name, 0, n);
    }


    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "cudaDeviceProp["+createString(",")+"]";
    }

    /**
     * Creates and returns a formatted (aligned, multi-line) String
     * representation of this object
     *
     * @return A formatted String representation of this object
     */
    public String toFormattedString()
    {
        return "Device properties:\n    "+createString("\n    ");
    }

    /**
     * Creates and returns a string representation of this object,
     * using the given separator for the fields
     *
     * @return A String representation of this object
     */
    private String createString(String f)
    {
        return
            "name="+createString(name)+f+
            "totalGlobalMem="+totalGlobalMem+f+
            "sharedMemPerBlock="+sharedMemPerBlock+f+
            "regsPerBlock="+regsPerBlock+f+
            "warpSize="+warpSize+f+
            "memPitch="+memPitch+f+
            "maxThreadsPerBlock="+maxThreadsPerBlock+f+
            "maxThreadsDim="+Arrays.toString(maxThreadsDim)+f+
            "maxGridSize="+Arrays.toString(maxGridSize)+f+
            "clockRate="+clockRate+f+
            "totalConstMem="+totalConstMem+f+
            "major="+major+f+
            "minor="+minor+f+
            "textureAlignment="+textureAlignment+f+
            "texturePitchAlignment="+texturePitchAlignment+f+
            "deviceOverlap="+deviceOverlap+f+
            "multiProcessorCount="+multiProcessorCount+f+
            "kernelExecTimeoutEnabled="+kernelExecTimeoutEnabled+f+
            "integrated="+integrated+f+
            "canMapHostMemory="+canMapHostMemory+f+
            "computeMode="+cudaComputeMode.stringFor(computeMode)+f+
            "maxTexture1D="+maxTexture1D+f+
            "maxTexture1DMipmap="+maxTexture1DMipmap+f+
            "maxTexture1DLinear="+maxTexture1DLinear+f+
            "maxTexture2D="+Arrays.toString(maxTexture2D)+f+
            "maxTexture2DMipmap="+Arrays.toString(maxTexture2DMipmap)+f+
            "maxTexture2DLinear="+Arrays.toString(maxTexture2DLinear)+f+
            "maxTexture2DGather="+Arrays.toString(maxTexture2DGather)+f+
            "maxTexture3D="+Arrays.toString(maxTexture3D)+f+
            "maxTexture3DAlt="+Arrays.toString(maxTexture3DAlt)+f+
            "maxTextureCubemap="+maxTextureCubemap+f+
            "maxTexture1DLayered="+Arrays.toString(maxTexture1DLayered)+f+
            "maxTexture2DLayered="+Arrays.toString(maxTexture2DLayered)+f+
            "maxTextureCubemapLayered="+Arrays.toString(maxTextureCubemapLayered)+f+
            "maxSurface1D="+maxSurface1D+f+
            "maxSurface2D="+Arrays.toString(maxSurface2D)+f+
            "maxSurface3D="+Arrays.toString(maxSurface3D)+f+
            "maxSurface1DLayered="+Arrays.toString(maxSurface1DLayered)+f+
            "maxSurface2DLayered="+Arrays.toString(maxSurface2DLayered)+f+
            "maxSurfaceCubemap="+maxSurfaceCubemap+f+
            "maxSurfaceCubemapLayered="+Arrays.toString(maxSurfaceCubemapLayered)+f+
            "surfaceAlignment="+surfaceAlignment+f+
            "concurrentKernels="+concurrentKernels+f+
            "ECCEnabled="+ECCEnabled+f+
            "pciBusID="+pciBusID+f+
            "pciDeviceID="+pciDeviceID+f+
            "pciDomainID="+pciDomainID+f+
            "tccDriver="+tccDriver+f+
            "asyncEngineCount="+asyncEngineCount+f+
            "unifiedAddressing="+unifiedAddressing+f+
            "memoryClockRate="+memoryClockRate+f+
            "memoryBusWidth="+memoryBusWidth+f+
            "l2CacheSize="+l2CacheSize+f+
            "maxThreadsPerMultiProcessor="+maxThreadsPerMultiProcessor+f+
            "streamPrioritiesSupported="+streamPrioritiesSupported+f+
            "globalL1CacheSupported="+globalL1CacheSupported+f+
            "localL1CacheSupported="+localL1CacheSupported+f+
            "sharedMemPerMultiprocessor="+sharedMemPerMultiprocessor+f+
            "regsPerMultiprocessor="+regsPerMultiprocessor+f+
            "managedMemory="+managedMemory+f+
            "isMultiGpuBoard="+isMultiGpuBoard+f+
            "multiGpuBoardGroupID="+multiGpuBoardGroupID+f;
    }

    /**
     * Creates a String from the letter, digit and whitespace
     * characters in the given byte array
     *
     * @param bytes The bytes for the String
     * @return The String
     */
    private static String createString(byte bytes[])
    {
        StringBuffer sb = new StringBuffer();
        for (byte b : bytes)
        {
            if (Character.isLetterOrDigit(b) || Character.isWhitespace(b))
            {
                sb.append((char)b);
            }
        }
        return sb.toString();
    }


}
