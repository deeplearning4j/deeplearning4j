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
 * Resource types
 */
public class CUresourcetype
{
    /**
     * Array resource
     */
    public static final int CU_RESOURCE_TYPE_ARRAY           = 0x00;

    /**
     * Mipmapped array resource
     */
    public static final int CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01;

    /**
     * Linear resource
     */
    public static final int CU_RESOURCE_TYPE_LINEAR          = 0x02;

    /**
     * Pitch 2D resource
     */
    public static final int CU_RESOURCE_TYPE_PITCH2D         = 0x03;

    /**
     * Returns the String identifying the given CUresourcetype
     *
     * @param n The CUresourcetype
     * @return The String identifying the given CUresourcetype
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_RESOURCE_TYPE_ARRAY : return "CU_RESOURCE_TYPE_ARRAY";
            case CU_RESOURCE_TYPE_MIPMAPPED_ARRAY : return "CU_RESOURCE_TYPE_MIPMAPPED_ARRAY";
            case CU_RESOURCE_TYPE_LINEAR : return "CU_RESOURCE_TYPE_LINEAR";
            case CU_RESOURCE_TYPE_PITCH2D : return "CU_RESOURCE_TYPE_PITCH2D";
        }
        return "INVALID CUresourcetype: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUresourcetype()
    {
    }


}
