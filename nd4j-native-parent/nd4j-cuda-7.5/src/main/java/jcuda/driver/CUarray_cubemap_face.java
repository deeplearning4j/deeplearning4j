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
 * Array indices for cube faces
 */
public class CUarray_cubemap_face
{
    /**
     * Positive X face of cubemap
     */
    public static final int CU_CUBEMAP_FACE_POSITIVE_X  = 0x00;

    /**
     * Negative X face of cubemap
     */
    public static final int CU_CUBEMAP_FACE_NEGATIVE_X  = 0x01;

    /**
     * Positive Y face of cubemap
     */
    public static final int CU_CUBEMAP_FACE_POSITIVE_Y  = 0x02;

    /**
     * Negative Y face of cubemap
     */
    public static final int CU_CUBEMAP_FACE_NEGATIVE_Y  = 0x03;

    /**
     * Positive Z face of cubemap
     */
    public static final int CU_CUBEMAP_FACE_POSITIVE_Z  = 0x04;

    /**
     * Negative Z face of cubemap
     */
    public static final int CU_CUBEMAP_FACE_NEGATIVE_Z  = 0x05;


    /**
     * Returns the String identifying the given CUarray_cubemap_face
     *
     * @param n The CUarray_cubemap_face
     * @return The String identifying the given CUarray_cubemap_face
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_CUBEMAP_FACE_POSITIVE_X: return "CU_CUBEMAP_FACE_POSITIVE_X";
            case CU_CUBEMAP_FACE_NEGATIVE_X: return "CU_CUBEMAP_FACE_NEGATIVE_X";
            case CU_CUBEMAP_FACE_POSITIVE_Y: return "CU_CUBEMAP_FACE_POSITIVE_Y";
            case CU_CUBEMAP_FACE_NEGATIVE_Y: return "CU_CUBEMAP_FACE_NEGATIVE_Y";
            case CU_CUBEMAP_FACE_POSITIVE_Z: return "CU_CUBEMAP_FACE_POSITIVE_Z";
            case CU_CUBEMAP_FACE_NEGATIVE_Z: return "CU_CUBEMAP_FACE_NEGATIVE_Z";
        }
        return "INVALID CUarray_cubemap_face: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUarray_cubemap_face()
    {
    }


}

