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

package jcuda.runtime;

/**
 * CUDA function cache configurations
 */
public class cudaFuncCache
{
    /**
     * Default function cache configuration, no preference
     */
    public static final int cudaFuncCachePreferNone   = 0;


    /**
     * Prefer larger shared memory and smaller L1 cache
     */
    public static final int cudaFuncCachePreferShared = 1;


    /**
     * Prefer larger L1 cache and smaller shared memory
     */
    public static final int cudaFuncCachePreferL1     = 2;

    /**
     * Prefer equal size L1 cache and shared memory
     */
    public static final int cudaFuncCachePreferEqual  = 3;

    /**
     * Returns the String identifying the given cudaFuncCache
     *
     * @param n The cudaFuncCache
     * @return The String identifying the given cudaFuncCache
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaFuncCachePreferNone: return "cudaFuncCachePreferNone";
            case cudaFuncCachePreferShared: return "cudaFuncCachePreferShared";
            case cudaFuncCachePreferL1: return "cudaFuncCachePreferL1";
            case cudaFuncCachePreferEqual: return "cudaFuncCachePreferEqual";
        }
        return "INVALID cudaFuncCache: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaFuncCache()
    {
    }

}
