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


