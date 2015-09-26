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

package jcuda.runtime;

/**
 * Shared memory configurations
 */
public class cudaSharedMemConfig
{
    /**
     * Set default shared memory bank size 
     */
    public static final int cudaSharedMemBankSizeDefault   = 0;
   
    /**
     *  Set shared memory bank width to four bytes 
     */
    public static final int cudaSharedMemBankSizeFourByte = 1;

    /** 
     * Set shared memory bank width to eight bytes 
     */
    public static final int cudaSharedMemBankSizeEightByte     = 2;
    
    /**
     * Returns the String identifying the given cudaSharedMemConfig
     *
     * @param n The cudaSharedMemConfig
     * @return The String identifying the given cudaSharedMemConfig
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaSharedMemBankSizeDefault: return "cudaSharedMemBankSizeDefault";
            case cudaSharedMemBankSizeFourByte: return "cudaSharedMemBankSizeFourByte";
            case cudaSharedMemBankSizeEightByte: return "cudaSharedMemBankSizeEightByte";
        }
        return "INVALID cudaSharedMemConfig: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaSharedMemConfig()
    {
    }

}
