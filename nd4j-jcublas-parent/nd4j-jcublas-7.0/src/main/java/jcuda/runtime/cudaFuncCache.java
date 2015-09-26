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
