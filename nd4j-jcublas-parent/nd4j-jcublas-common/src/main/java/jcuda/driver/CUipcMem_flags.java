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
 * Memory flags
 */
public class CUipcMem_flags
{
    /** 
     * Automatically enable peer access between remote devices as needed 
     */
    public static final int CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1; 

    /**
     * Returns the String identifying the given CUipcMem_flags
     *
     * @param n The CUipcMem_flags
     * @return The String identifying the given CUipcMem_flags
     */
    public static String stringFor(int n)
    {
        if (n == 0)
        {
            return "INVALID CUipcMem_flags: "+n;
        }
        String result = "";
        if ((n & CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS) != 0) result += "CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS";
        return result;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUipcMem_flags()
    {
    }
    
}
