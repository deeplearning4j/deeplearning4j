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
 * Context creation flags.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see JCudaDriver#cuCtxCreate
 */
public class CUctx_flags
{

    /**
     * Automatic scheduling
     */
    public static final int CU_CTX_SCHED_AUTO  = 0x00;

    /**
     * Set spin as default scheduling
     */
    public static final int CU_CTX_SCHED_SPIN  = 0x01;

    /**
     * Set yield as default scheduling
     */
    public static final int CU_CTX_SCHED_YIELD = 0x02;

    /**
     * Use blocking synchronization
     */
    public static final int CU_CTX_BLOCKING_SYNC = 0x04;

    /**
     * Use blocking synchronization
     */
    public static final int CU_CTX_SCHED_BLOCKING_SYNC = 0x04;

    /**
     * Scheduling flags mask
     */
    public static final int CU_CTX_SCHED_MASK  = 0x07;

    /**
     * Support mapped pinned allocations
     */
    public static final int CU_CTX_MAP_HOST = 0x08;

    /**
     * Keep local memory allocation after launch
     */
    public static final int CU_CTX_LMEM_RESIZE_TO_MAX = 0x10;

    /**
     * Context creation flags mask
     */
    public static final int CU_CTX_FLAGS_MASK  = 0x3F;

    /**
     * Returns the String identifying the given CUctx_flags
     *
     * @param n The CUctx_flags
     * @return The String identifying the given CUctx_flags
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_CTX_SCHED_AUTO : return "CU_CTX_SCHED_AUTO";
            case CU_CTX_SCHED_SPIN : return "CU_CTX_SCHED_SPIN";
            case CU_CTX_SCHED_YIELD : return "CU_CTX_SCHED_YIELD";
            case CU_CTX_BLOCKING_SYNC: return "CU_CTX_BLOCKING_SYNC";
            case CU_CTX_MAP_HOST: return "CU_CTX_MAP_HOST";
            case CU_CTX_LMEM_RESIZE_TO_MAX: return "CU_CTX_LMEM_RESIZE_TO_MAX";
            case CU_CTX_FLAGS_MASK : return "[CU_CTX_FLAGS_MASK]";
            case CU_CTX_SCHED_MASK : return "[CU_CTX_SCHED_MASK]";
        }
        return "INVALID CUctx_flags: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUctx_flags()
    {
    }

}
