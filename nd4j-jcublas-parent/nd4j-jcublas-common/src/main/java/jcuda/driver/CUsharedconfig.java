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
 * Shared memory configurations
 */
public class CUsharedconfig
{
    /**
     * Set default shared memory bank size 
     */
    public static final int CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE    = 0x00;
    
    /**
     *  Set shared memory bank width to four bytes 
     */
    public static final int CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE  = 0x01;
    
    /** 
     * Set shared memory bank width to eight bytes 
     */
    public static final int CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02;  

    /**
     * Returns the String identifying the given CUsharedconfig
     *
     * @param n The CUsharedconfig
     * @return The String identifying the given CUsharedconfig
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: return "CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE";
            case CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: return "CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE";
            case CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: return "CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE";
        }
        return "INVALID CUsharedconfig: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUsharedconfig()
    {
    }

}



