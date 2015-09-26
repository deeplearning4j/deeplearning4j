/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package jcuda.driver;

import jcuda.Pointer;

/**
 * Java port of a CUDA_MEMCPY3D setup.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual
 * <br />
 * @see JCudaDriver#cuMemcpy3D(CUDA_MEMCPY3D)
 */
public class CUDA_MEMCPY3D
{
    /**
     * srcXInBytes, srcY and srcZ specify the base address of the source data for the copy.
     */
    public long srcXInBytes;

    /**
     * srcXInBytes, srcY and srcZ specify the base address of the source data for the copy.
     */
    public long srcY;

    /**
     * srcXInBytes, srcY and srcZ specify the base address of the source data for the copy.
     */
    public long srcZ;

    /**
     * Must be set to 0
     */
    public long srcLOD;

    /**
     * The source memory type.
     * @see CUmemorytype
     */
    public int srcMemoryType;

    /**
     * The source pointer.
     */
    public Pointer srcHost = new CUdeviceptr();

    /**
     * The source pointer.
     */
    public CUdeviceptr srcDevice = new CUdeviceptr();

    /**
     * The source array.
     */
    public CUarray srcArray = new CUarray();

    /**
     * The source pitch - ignored when src is array.
     */
    public long srcPitch;

    /**
     * The source height - ignored when src is array and may be 0 if Depth==1
     */
    public long srcHeight;

    /**
     * dstXInBytes, dstY and dstZ specify the base address of the destination data for the copy.
     */
    public long dstXInBytes;

    /**
     * dstXInBytes, dstY and dstZ specify the base address of the destination data for the copy.
     */
    public long dstY;

    /**
     * dstXInBytes, dstY and dstZ specify the base address of the destination data for the copy.
     */
    public long dstZ;

    /**
     * Must be set to 0
     */
    public long dstLOD;


    /**
     * The destination memory type.
     * @see CUmemorytype
     */
    public int dstMemoryType;

    /**
     * The destination pointer.
     */
    public Pointer dstHost = new Pointer();

    /**
     * The destination pointer.
     */
    public CUdeviceptr dstDevice = new CUdeviceptr();

    /**
     * The destination array.
     */
    public CUarray dstArray = new CUarray();

    /**
     * The destination pitch - ignored when dst is array.
     */
    public long dstPitch;

    /**
     * The destination height - ignored when dst is array and may be 0 if Depth==1
     */
    public long dstHeight;

    /**
     * WidthInBytes, Height and Depth specify the width (in bytes), height and depth of the 3D copy
     * being performed. Any pitches must be greater than or equal to WidthInBytes.
     */
    public long WidthInBytes;

    /**
     * WidthInBytes, Height and Depth specify the width (in bytes), height and depth of the 3D copy
     * being performed. Any pitches must be greater than or equal to WidthInBytes.
     */
    public long Height;

    /**
     * WidthInBytes, Height and Depth specify the width (in bytes), height and depth of the 3D copy
     * being performed. Any pitches must be greater than or equal to WidthInBytes.
     */
    public long Depth;


    /**
     * Creates a new, uninitialized CUDA_MEMCPY3D
     */
    public CUDA_MEMCPY3D()
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
        return "CUDA_MEMCPY3D["+createString(",")+"]";
    }

    /**
     * Creates and returns a formatted (aligned, multi-line) String
     * representation of this object
     *
     * @return A formatted String representation of this object
     */
    public String toFormattedString()
    {
        return "3D memory copy setup:\n    "+createString("\n    ");
    }

    /**
     * Creates and returns a string representation of this object,
     * using the given separator for the fields
     *
     * @param f Separator
     * @return A String representation of this object
     */
    private String createString(String f)
    {
        return
            "srcXInBytes="+srcXInBytes+f+
            "srcY="+srcY+f+
            "srcZ="+srcZ+f+
            "srcLOD="+srcLOD+f+
            "srcMemoryType="+CUmemorytype.stringFor(srcMemoryType)+f+
            "srcHost ="+srcHost +f+
            "srcDevice ="+srcDevice +f+
            "srcArray ="+srcArray +f+
            "srcPitch="+srcPitch+f+
            "srcHeight="+srcHeight+f+
            "dstXInBytes="+dstXInBytes+f+
            "dstY="+dstY+f+
            "dstZ="+dstZ+f+
            "dstLOD="+dstLOD+f+
            "dstMemoryType="+CUmemorytype.stringFor(dstMemoryType)+f+
            "dstHost ="+dstHost +f+
            "dstDevice ="+dstDevice +f+
            "dstArray ="+dstArray +f+
            "dstPitch="+dstPitch+f+
            "dstHeight="+dstHeight+f+
            "WidthInBytes="+WidthInBytes+f+
            "Height="+Height+f+
            "Depth="+Depth;
    }


};
