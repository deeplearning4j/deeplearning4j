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
 * Function cache configurations.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 */
public class CUfunc_cache
{
    /**
     * No preference for shared memory or L1 (default)
     */
    public static final int CU_FUNC_CACHE_PREFER_NONE   = 0x00;

    /**
     * Prefer larger shared memory and smaller L1 cache
     */
    public static final int CU_FUNC_CACHE_PREFER_SHARED = 0x01;

    /**
     * Prefer larger L1 cache and smaller shared memory
     */
    public static final int CU_FUNC_CACHE_PREFER_L1     = 0x02;

    /**
     * Prefer equal sized L1 cache and shared memory
     */
    public static final int CU_FUNC_CACHE_PREFER_EQUAL   = 0x03;

    /**
     * Returns the String identifying the given CUfunc_cache
     *
     * @param n The CUfunc_cache
     * @return The String identifying the given CUfunc_cache
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_FUNC_CACHE_PREFER_NONE: return "CU_FUNC_CACHE_PREFER_NONE";
            case CU_FUNC_CACHE_PREFER_SHARED: return "CU_FUNC_CACHE_PREFER_SHARED";
            case CU_FUNC_CACHE_PREFER_L1: return "CU_FUNC_CACHE_PREFER_L1";
            case CU_FUNC_CACHE_PREFER_EQUAL: return "CU_FUNC_CACHE_PREFER_EQUAL";
        }
        return "INVALID CUfunc_cache: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUfunc_cache()
    {
    }

};

