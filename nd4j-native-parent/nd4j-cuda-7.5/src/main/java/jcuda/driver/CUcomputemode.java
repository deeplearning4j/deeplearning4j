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
 * Compute Modes. <br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 * <br />
 * @see CUdevice_attribute#CU_DEVICE_ATTRIBUTE_COMPUTE_MODE
 */
public class CUcomputemode
{

    /**
     * Default compute mode (Multiple contexts allowed per device)
     */
    public static final int CU_COMPUTEMODE_DEFAULT    = 0;

    /**
     * Compute-exclusive-thread mode (Only one context used by a
     * single thread can be present on this device at a time)
     */
    public static final int CU_COMPUTEMODE_EXCLUSIVE  = 1;

    /**
     * Compute-prohibited mode (No contexts can be created on
     * this device at this time)
     */
    public static final int CU_COMPUTEMODE_PROHIBITED = 2;

    /**
     * Compute-exclusive-process mode (Only one context used by a
     * single process can be present on this device at a time)
     */
    public static final int CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3;


    /**
     * Returns the String identifying the given CUcomputemode
     *
     * @param n The CUcomputemode
     * @return The String identifying the given CUcomputemode
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_COMPUTEMODE_DEFAULT: return "CU_COMPUTEMODE_DEFAULT";
            case CU_COMPUTEMODE_EXCLUSIVE: return "CU_COMPUTEMODE_EXCLUSIVE";
            case CU_COMPUTEMODE_PROHIBITED: return "CU_COMPUTEMODE_PROHIBITED";
            case CU_COMPUTEMODE_EXCLUSIVE_PROCESS: return "CU_COMPUTEMODE_EXCLUSIVE_PROCESS";
        }
        return "INVALID CUcomputemode: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUcomputemode()
    {
    }


}


