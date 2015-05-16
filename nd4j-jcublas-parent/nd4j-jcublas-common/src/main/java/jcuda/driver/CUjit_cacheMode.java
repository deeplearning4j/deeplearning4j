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
 * Caching modes for dlcm 
 */
public class CUjit_cacheMode
{
    /** 
     * Compile with no -dlcm flag specified 
     */
    public static final int CU_JIT_CACHE_OPTION_NONE = 0;

    /**
     * Compile with L1 cache disabled 
     */
    public static final int CU_JIT_CACHE_OPTION_CG = 1;
    
    /**
     * Compile with L1 cache enabled 
     */
    public static final int CU_JIT_CACHE_OPTION_CA = 2;

    /**
     * Returns the String identifying the given CUjit_cacheMode
     *
     * @param n The CUjit_cacheMode
     * @return The String identifying the given CUjit_cacheMode
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_JIT_CACHE_OPTION_NONE: return "CU_JIT_CACHE_OPTION_NONE";
            case CU_JIT_CACHE_OPTION_CG: return "CU_JIT_CACHE_OPTION_CG";
            case CU_JIT_CACHE_OPTION_CA: return "CU_JIT_CACHE_OPTION_CA";
        }
        return "INVALID CUjit_cacheMode: "+n;
    }
    
    /**
     * Private constructor to prevent instantation
     */
    private CUjit_cacheMode()
    {
        
    }
}
