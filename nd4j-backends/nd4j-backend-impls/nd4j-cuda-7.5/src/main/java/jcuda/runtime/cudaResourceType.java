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
 * CUDA resource types
 */
public class cudaResourceType
{
    /**
     * Array resource
     */
    public static final int cudaResourceTypeArray          = 0x00;

    /**
     * Mipmapped array resource
     */
    public static final int cudaResourceTypeMipmappedArray = 0x01;

    /**
     * Linear resource
     */
    public static final int cudaResourceTypeLinear         = 0x02;

    /**
     * Pitch 2D resource
     */
    public static final int cudaResourceTypePitch2D        = 0x03;

    /**
     * Returns the String identifying the given cudaResourceType
     *
     * @param k The cudaResourceType
     * @return The String identifying the given cudaResourceType
     */
    public static String stringFor(int k)
    {
        switch (k)
        {
            case cudaResourceTypeArray: return "cudaResourceTypeArray";
            case cudaResourceTypeMipmappedArray: return "cudaResourceTypeMipmappedArray";
            case cudaResourceTypeLinear: return "cudaResourceTypeLinear";
            case cudaResourceTypePitch2D: return "cudaResourceTypePitch2D";
        }
        return "INVALID cudaResourceType: "+k;
    }


    /**
     * Private constructor to prevent instantiation.
     */
    private cudaResourceType()
    {

    }
};
