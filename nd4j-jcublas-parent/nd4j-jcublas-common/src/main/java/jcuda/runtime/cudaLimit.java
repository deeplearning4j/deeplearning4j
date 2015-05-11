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
 * CUDA limits.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 */
public class cudaLimit
{
    /**
     * GPU thread stack size
     */
    public static final int cudaLimitStackSize      = 0x00;
    
    /**
     * GPU printf/fprintf FIFO size
     */
    public static final int cudaLimitPrintfFifoSize = 0x01;
    
    /** 
     * GPU malloc heap size 
     */
    public static final int cudaLimitMallocHeapSize = 0x02;
    
    /**
     * GPU device runtime synchronize depth
     */
    public static final int cudaLimitDevRuntimeSyncDepth          = 0x03;
    
    /**
     * GPU device runtime pending launch count 
     */
    public static final int cudaLimitDevRuntimePendingLaunchCount = 0x04;
    
    /**
     * Returns the String identifying the given cudaLimit
     *
     * @param n The cudaLimit
     * @return The String identifying the given cudaLimit
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaLimitStackSize: return "cudaLimitStackSize";
            case cudaLimitPrintfFifoSize: return "cudaLimitPrintfFifoSize";
            case cudaLimitMallocHeapSize: return "cudaLimitMallocHeapSize";
            case cudaLimitDevRuntimeSyncDepth: return "cudaLimitDevRuntimeSyncDepth";
            case cudaLimitDevRuntimePendingLaunchCount: return "cudaLimitDevRuntimePendingLaunchCount";
        }
        return "INVALID cudaLimit: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaLimit()
    {
    }

};

