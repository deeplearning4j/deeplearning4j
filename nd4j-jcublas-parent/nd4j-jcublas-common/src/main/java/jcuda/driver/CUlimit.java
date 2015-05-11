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
 * Limits
 */
public class CUlimit
{
    /**
     * GPU thread stack size
     */
    public static final int CU_LIMIT_STACK_SIZE        = 0x00;
    
    /**
     * GPU printf FIFO size
     */
    public static final int CU_LIMIT_PRINTF_FIFO_SIZE  = 0x01;

    /**
     * GPU malloc heap size 
     */
    public static final int CU_LIMIT_MALLOC_HEAP_SIZE  = 0x02;  

    /**
     * GPU device runtime launch synchronize depth 
     */
    public static final int CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH           = 0x03;
    
    /** 
     * GPU device runtime pending launch count 
     */
    public static final int CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04;
    
    /**
     * Returns the String identifying the given CUlimit
     *
     * @param n The CUlimit
     * @return The String identifying the given CUlimit
     */
    public static String stringFor(int n)
    {
        switch (n) 
        {
            case CU_LIMIT_STACK_SIZE : return "CU_LIMIT_STACK_SIZE";
            case CU_LIMIT_PRINTF_FIFO_SIZE : return "CU_LIMIT_PRINTF_FIFO_SIZE";
            case CU_LIMIT_MALLOC_HEAP_SIZE : return "CU_LIMIT_MALLOC_HEAP_SIZE";
            case CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH : return "CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH";
            case CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT : return "CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT";
        }
        return "INVALID CUlimit: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUlimit()
    {
    }


}
