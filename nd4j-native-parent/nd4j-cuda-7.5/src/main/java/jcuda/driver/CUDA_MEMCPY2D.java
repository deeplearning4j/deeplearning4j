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

import jcuda.*;

/**
 * Java port of a CUDA_MEMCPY2D setup.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual<br />
 * <br />
 * @see JCudaDriver#cuMemcpy2D(CUDA_MEMCPY2D)
 */
public class CUDA_MEMCPY2D
{
    /**
     * srcXInBytes and srcY specify the base address of the source data for the copy.
     */
    public long srcXInBytes;

    /**
     * srcXInBytes and srcY specify the base address of the source data for the copy.
     */
    public long srcY;

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
     * dstXInBytes and dstY specify the base address of the destination data for the copy.
     */
    public long dstXInBytes;

    /**
     * dstXInBytes and dstY specify the base address of the destination data for the copy.
     */
    public long dstY;

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
     * WidthInBytes and Height specify the width (in bytes) and height of the 2D copy being performed.
     * Any pitches must be greater than or equal to WidthInBytes.
     */
    public long WidthInBytes;

    /**
     * WidthInBytes and Height specify the width (in bytes) and height of the 2D copy being performed.
     * Any pitches must be greater than or equal to WidthInBytes.
     */
    public long Height;


    /**
     * Creates a new, uninitialized CUDA_MEMCPY2D
     */
    public CUDA_MEMCPY2D()
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
        return "CUDA_MEMCPY2D["+createString(",")+"]";
    }

    /**
     * Creates and returns a formatted (aligned, multi-line) String
     * representation of this object
     *
     * @return A formatted String representation of this object
     */
    public String toFormattedString()
    {
        return "2D memory copy setup:\n    "+createString("\n    ");
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
            "srcMemoryType="+CUmemorytype.stringFor(srcMemoryType)+f+
            "srcHost ="+srcHost +f+
            "srcDevice ="+srcDevice +f+
            "srcArray ="+srcArray +f+
            "srcPitch="+srcPitch+f+
            "dstXInBytes="+dstXInBytes+f+
            "dstY="+dstY+f+
            "dstMemoryType="+CUmemorytype.stringFor(dstMemoryType)+f+
            "dstHost ="+dstHost +f+
            "dstDevice ="+dstDevice +f+
            "dstArray ="+dstArray +f+
            "dstPitch="+dstPitch+f+
            "WidthInBytes="+WidthInBytes+f+
            "Height="+Height;
    }


};
