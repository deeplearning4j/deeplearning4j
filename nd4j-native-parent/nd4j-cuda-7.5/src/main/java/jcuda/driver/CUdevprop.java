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

import java.util.Arrays;

/**
 * Legacy device properties.
 *
 * @see jcuda.driver.JCudaDriver#cuDeviceGetProperties(CUdevprop, CUdevice)
 */
public class CUdevprop
{
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
     * The total amount of shared memory available per block in bytes;
     */
    public int sharedMemPerBlock;

    /**
     * The total amount of constant memory available on the device in bytes;
     */
    public int totalConstantMemory;

    /**
     * The warp size;
     */
    public int SIMDWidth;

    /**
     * The maximum pitch allowed by the memory copy functions that involve memory regions allocated through cuMemAllocPitch();
     */
    public int memPitch;

    /**
     * The total number of registers available per block;
     */
    public int regsPerBlock;

    /**
     * The clock frequency in kilohertz;
     */
    public int clockRate;

    /**
     * The alignment requirement; texture base addresses that are aligned to textureAlign bytes do not need an offset applied to texture fetches
     */
    public int textureAlign;


    /**
     * Creates a new, uninitialized CUdevprop
     */
    public CUdevprop()
    {
    }

    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "CUdevprop["+createString(",")+"]";
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
            "maxThreadsPerBlock="+maxThreadsPerBlock+f+
            "maxThreadsDim="+Arrays.toString(maxThreadsDim)+f+
            "maxGridSize="+Arrays.toString(maxGridSize)+f+
            "sharedMemPerBlock="+sharedMemPerBlock+f+
            "totalConstantMemory="+totalConstantMemory+f+
            "regsPerBlock="+regsPerBlock+f+
            "SIMDWidth="+SIMDWidth+f+
            "memPitch="+memPitch+f+
            "regsPerBlock="+regsPerBlock+f+
            "clockRate="+clockRate+f+
            "textureAlign="+textureAlign;
    }
}
