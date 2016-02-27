/*
 * JCurand - Java bindings for CURAND, the NVIDIA CUDA random
 * number generation library, to be used with JCuda
 *
 * Copyright (c) 2010-2015 Marco Hutter - http://www.jcuda.org
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
package jcuda.jcurand;

/**
 * CURAND choice of direction vector set
 */
public class curandDirectionVectorSet
{
    /**
     * Specific set of 32-bit direction vectors generated from polynomials
     * recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
     */
    public static final int CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101;
    /**
     * Specific set of 32-bit direction vectors generated from polynomials
     * recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions,
     * and scrambled
     */
    public static final int CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102;
    /**
     * Specific set of 64-bit direction vectors generated from polynomials
     * recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
     */
    public static final int CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103;
    /**
     * Specific set of 64-bit direction vectors generated from polynomials
     * recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions,
     * and scrambled
     */
    public static final int CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104;

    /**
     * Private constructor to prevent instantiation
     */
    private curandDirectionVectorSet(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CURAND_DIRECTION_VECTORS_32_JOEKUO6: return "CURAND_DIRECTION_VECTORS_32_JOEKUO6";
            case CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6: return "CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6";
            case CURAND_DIRECTION_VECTORS_64_JOEKUO6: return "CURAND_DIRECTION_VECTORS_64_JOEKUO6";
            case CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6: return "CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6";
        }
        return "INVALID curandDirectionVectorSet";
    }
}

