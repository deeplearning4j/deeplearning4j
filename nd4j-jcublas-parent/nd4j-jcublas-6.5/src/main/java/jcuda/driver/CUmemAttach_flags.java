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
 * CUDA Mem Attach Flags
 */
public class CUmemAttach_flags
{
    /** 
     * Memory can be accessed by any stream on any device 
     */
    public static final int CU_MEM_ATTACH_GLOBAL = 0x1; 
    
    /** 
     * Memory cannot be accessed by any stream on any device 
     */
    public static final int CU_MEM_ATTACH_HOST   = 0x2; 
    
    /** 
     * Memory can only be accessed by a single stream on the 
     * associated device 
     */
    public static final int CU_MEM_ATTACH_SINGLE = 0x4;

    /**
     * Returns the String identifying the given CUmemAttach_flags
     *
     * @param n The CUmemAttach_flags
     * @return The String identifying the given CUmemAttach_flags
     */
    public static String stringFor(int n)
    {
        if (n == 0)
        {
            return "INVALID CUmemAttach_flags: "+n;
        }
        String result = "";
        if ((n & CU_MEM_ATTACH_GLOBAL) != 0) result += "CU_MEM_ATTACH_GLOBAL ";
        if ((n & CU_MEM_ATTACH_HOST) != 0) result += "CU_MEM_ATTACH_HOST ";
        if ((n & CU_MEM_ATTACH_SINGLE) != 0) result += "CU_MEM_ATTACH_SINGLE ";
        return result;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUmemAttach_flags()
    {
    }
    
}
