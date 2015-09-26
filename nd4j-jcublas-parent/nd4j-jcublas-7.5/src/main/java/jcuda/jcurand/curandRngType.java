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
 * CURAND generator types
 */
public class curandRngType
{
    public static final int CURAND_RNG_TEST = 0;
    /**
     * Default pseudorandom generator
     */
    public static final int CURAND_RNG_PSEUDO_DEFAULT = 100;
    /**
     * XORWOW pseudorandom generator
     */
    public static final int CURAND_RNG_PSEUDO_XORWOW = 101;
    /**
     * MRG32k3a pseudorandom generator
     */
    public static final int CURAND_RNG_PSEUDO_MRG32K3A = 121;
    /**
     * Mersenne Twister MTGP32 pseudorandom generator
     */
    public static final int CURAND_RNG_PSEUDO_MTGP32 = 141;
    /**
     * Mersenne Twister MT19937 pseudorandom generator
     */
    public static final int CURAND_RNG_PSEUDO_MT19937 = 142;
    /**
     * PHILOX-4x32-10 pseudorandom generator
     */
    public static final int CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161;
    /**
     * Default quasirandom generator
     */
    public static final int CURAND_RNG_QUASI_DEFAULT = 200;
    /**
     * Sobol32 quasirandom generator
     */
    public static final int CURAND_RNG_QUASI_SOBOL32 = 201;
    /**
     * Scrambled Sobol32 quasirandom generator
     */
    public static final int CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202;
    /**
     * Sobol64 quasirandom generator
     */
    public static final int CURAND_RNG_QUASI_SOBOL64 = 203;
    /**
     * Scrambled Sobol64 quasirandom generator
     */
    public static final int CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204;

    /**
     * Private constructor to prevent instantiation
     */
    private curandRngType(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CURAND_RNG_TEST: return "CURAND_RNG_TEST";
            case CURAND_RNG_PSEUDO_DEFAULT: return "CURAND_RNG_PSEUDO_DEFAULT";
            case CURAND_RNG_PSEUDO_XORWOW: return "CURAND_RNG_PSEUDO_XORWOW";
            case CURAND_RNG_PSEUDO_MRG32K3A: return "CURAND_RNG_PSEUDO_MRG32K3A";
            case CURAND_RNG_PSEUDO_MTGP32: return "CURAND_RNG_PSEUDO_MTGP32";
            case CURAND_RNG_PSEUDO_MT19937: return "CURAND_RNG_PSEUDO_MT19937";
            case CURAND_RNG_QUASI_DEFAULT: return "CURAND_RNG_QUASI_DEFAULT";
            case CURAND_RNG_QUASI_SOBOL32: return "CURAND_RNG_QUASI_SOBOL32";
            case CURAND_RNG_QUASI_SCRAMBLED_SOBOL32: return "CURAND_RNG_QUASI_SCRAMBLED_SOBOL32";
            case CURAND_RNG_QUASI_SOBOL64: return "CURAND_RNG_QUASI_SOBOL64";
            case CURAND_RNG_QUASI_SCRAMBLED_SOBOL64: return "CURAND_RNG_QUASI_SCRAMBLED_SOBOL64";
        }
        return "INVALID curandRngType";
    }
}

