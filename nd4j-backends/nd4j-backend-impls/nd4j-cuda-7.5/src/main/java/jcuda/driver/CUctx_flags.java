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
 * Context creation flags.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see jcuda.driver.JCudaDriver#cuCtxCreate
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
