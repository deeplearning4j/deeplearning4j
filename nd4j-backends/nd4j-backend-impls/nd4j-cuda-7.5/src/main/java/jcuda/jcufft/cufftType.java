/*
 * JCufft - Java bindings for CUFFT, the NVIDIA CUDA FFT library,
 * to be used with JCuda
 *
 * Copyright (c) 2008-2015 Marco Hutter - http://www.jcuda.org
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

package jcuda.jcufft;

/**
 * The type of a CUFFT operation
 */
public class cufftType
{
    /**
     * CUFFT transform type Real to complex (interleaved)
     */
    public static final int CUFFT_R2C = 0x2A;


    /**
     * CUFFT transform type Complex (interleaved) to real
     */
    public static final int CUFFT_C2R = 0x2C;


    /**
     * CUFFT transform type Complex to complex, interleaved
     */
    public static final int CUFFT_C2C = 0x29;

    /**
     * CUFFT transform type Double to Double-Complex
     */
    public static final int CUFFT_D2Z = 0x6a;

    /**
     * CUFFT transform type Double-Complex to Double
     */
    public static final int CUFFT_Z2D = 0x6c;

    /**
     * CUFFT transform type Double-Complex to Double-Complex
     */
    public static final int CUFFT_Z2Z = 0x69;


    /**
     * Returns the String identifying the given cufftType
     *
     * @param m The cufftType
     * @return The String identifying the given cufftType
     */
    public static String stringFor(int m)
    {
        switch (m)
        {
            case CUFFT_R2C : return "CUFFT_R2C";
            case CUFFT_C2R : return "CUFFT_C2R";
            case CUFFT_C2C : return "CUFFT_C2C";
            case CUFFT_D2Z : return "CUFFT_D2Z";
            case CUFFT_Z2D : return "CUFFT_Z2D";
            case CUFFT_Z2Z : return "CUFFT_Z2Z";
        }
        return "INVALID cufftType: " + m;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cufftType()
    {
    }

}
