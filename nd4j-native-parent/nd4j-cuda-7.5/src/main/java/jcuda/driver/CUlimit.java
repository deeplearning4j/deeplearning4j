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
