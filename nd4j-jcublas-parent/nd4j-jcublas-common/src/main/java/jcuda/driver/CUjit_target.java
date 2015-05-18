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

