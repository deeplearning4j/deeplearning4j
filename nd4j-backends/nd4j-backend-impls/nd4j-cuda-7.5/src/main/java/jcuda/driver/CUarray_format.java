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
 * Array formats.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see jcuda.driver.CUDA_ARRAY_DESCRIPTOR
 * @see jcuda.driver.CUDA_ARRAY3D_DESCRIPTOR
 */
public class CUarray_format
{

    /**
     * Unsigned 8-bit integers
     */
    public static final int CU_AD_FORMAT_UNSIGNED_INT8  = 0x01;

    /**
     * Unsigned 16-bit integers
     */
    public static final int CU_AD_FORMAT_UNSIGNED_INT16 = 0x02;

    /**
     * Unsigned 32-bit integers
     */
    public static final int CU_AD_FORMAT_UNSIGNED_INT32 = 0x03;

    /**
     * Signed 8-bit integers
     */
    public static final int CU_AD_FORMAT_SIGNED_INT8    = 0x08;

    /**
     * Signed 16-bit integers
     */
    public static final int CU_AD_FORMAT_SIGNED_INT16   = 0x09;

    /**
     * Signed 32-bit integers
     */
    public static final int CU_AD_FORMAT_SIGNED_INT32   = 0x0a;

    /**
     * 16-bit floating point
     */
    public static final int CU_AD_FORMAT_HALF           = 0x10;

    /**
     * 32-bit floating point
     */
    public static final int CU_AD_FORMAT_FLOAT          = 0x20;


    /**
     * Returns the String identifying the given CUarray_format
     *
     * @param n The CUarray_format
     * @return The String identifying the given CUarray_format
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_AD_FORMAT_UNSIGNED_INT8 : return "CU_AD_FORMAT_UNSIGNED_INT8";
            case CU_AD_FORMAT_UNSIGNED_INT16 : return "CU_AD_FORMAT_UNSIGNED_INT16";
            case CU_AD_FORMAT_UNSIGNED_INT32 : return "CU_AD_FORMAT_UNSIGNED_INT32";
            case CU_AD_FORMAT_SIGNED_INT8 : return "CU_AD_FORMAT_SIGNED_INT8";
            case CU_AD_FORMAT_SIGNED_INT16 : return "CU_AD_FORMAT_SIGNED_INT16";
            case CU_AD_FORMAT_SIGNED_INT32 : return "CU_AD_FORMAT_SIGNED_INT32";
            case CU_AD_FORMAT_HALF : return "CU_AD_FORMAT_HALF";
            case CU_AD_FORMAT_FLOAT : return "CU_AD_FORMAT_FLOAT";
        }
        return "INVALID CUarray_format: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUarray_format()
    {
    }

}
