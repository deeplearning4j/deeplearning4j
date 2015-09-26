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

