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
 * Online compilation targets.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 *
 * @see JCudaDriver#cuModuleLoadDataEx
 */
public class CUjit_target
{

    /**
     * Compute device class 1.0
     */
    public static final int CU_TARGET_COMPUTE_10 = 10;

    /**
     * Compute device class 1.1
     */
    public static final int CU_TARGET_COMPUTE_11 = 11;

    /**
     * Compute device class 1.2
     */
    public static final int CU_TARGET_COMPUTE_12 = 12;

    /**
     * Compute device class 1.3
     */
    public static final int CU_TARGET_COMPUTE_13 = 13;

    /**
     * Compute device class 2.0
     */
    public static final int CU_TARGET_COMPUTE_20 = 20;

    /**
     * Compute device class 2.1
     */
    public static final int CU_TARGET_COMPUTE_21 = 21;

    /**
     * Compute device class 3.0
     */
    public static final int CU_TARGET_COMPUTE_30 = 30;

    /**
     * Compute device class 3.2
     */
    public static final int CU_TARGET_COMPUTE_32 = 32;

    /**
     * Compute device class 3.5
     */
    public static final int CU_TARGET_COMPUTE_35 = 35;

    /**
     * Compute device class 3.7
     */
    public static final int CU_TARGET_COMPUTE_37 = 37;

    /**
     * Compute device class 5.0
     */
    public static final int CU_TARGET_COMPUTE_50 = 50;

    /**
     * Compute device class 5.2
     */
    public static final int CU_TARGET_COMPUTE_52 = 52;

    /**
     * Returns the String identifying the given CUjit_target
     *
     * @param n The CUjit_target
     * @return The String identifying the given CUjit_target
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_TARGET_COMPUTE_10: return "CU_TARGET_COMPUTE_10";
            case CU_TARGET_COMPUTE_11: return "CU_TARGET_COMPUTE_11";
            case CU_TARGET_COMPUTE_12: return "CU_TARGET_COMPUTE_12";
            case CU_TARGET_COMPUTE_13: return "CU_TARGET_COMPUTE_13";
            case CU_TARGET_COMPUTE_20: return "CU_TARGET_COMPUTE_20";
            case CU_TARGET_COMPUTE_21: return "CU_TARGET_COMPUTE_21";
            case CU_TARGET_COMPUTE_30: return "CU_TARGET_COMPUTE_30";
            case CU_TARGET_COMPUTE_32: return "CU_TARGET_COMPUTE_32";
            case CU_TARGET_COMPUTE_35: return "CU_TARGET_COMPUTE_35";
            case CU_TARGET_COMPUTE_37: return "CU_TARGET_COMPUTE_37";
            case CU_TARGET_COMPUTE_50: return "CU_TARGET_COMPUTE_50";
            case CU_TARGET_COMPUTE_52: return "CU_TARGET_COMPUTE_52";
        }
        return "INVALID CUjit_target: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUjit_target()
    {
    }

}

