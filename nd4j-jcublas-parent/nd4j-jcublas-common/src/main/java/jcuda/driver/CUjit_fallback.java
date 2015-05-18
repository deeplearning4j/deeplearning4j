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
 * Cubin matching fallback strategies.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 *
 * @see JCudaDriver#cuModuleLoadDataEx
 */
public class CUjit_fallback
{
    /**
     * Prefer to compile ptx if exact binary match not found
     */
    public static final int CU_PREFER_PTX = 0;

    /**
     * Prefer to fall back to compatible binary code if 
     * exact binary match not found
     */
    public static final int CU_PREFER_BINARY = 1;

    /**
     * Returns the String identifying the given CUjit_fallback
     *
     * @param n The CUjit_fallback
     * @return The String identifying the given CUjit_fallback
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_PREFER_PTX: return "CU_PREFER_PTX";
            case CU_PREFER_BINARY: return "CU_PREFER_BINARY";
        }
        return "INVALID CUjit_fallback: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUjit_fallback()
    {
    }

}
