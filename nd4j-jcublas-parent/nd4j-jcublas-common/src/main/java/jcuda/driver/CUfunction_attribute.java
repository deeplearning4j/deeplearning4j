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
 * Function properties.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see JCudaDriver#cuFuncGetAttribute
 */
public class CUfunction_attribute
{
    /**
     * The number of threads beyond which a launch of the function would fail.
     * This number depends on both the function and the device on which the
     * function is currently loaded.
     */
    public static final int CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0;

    /**
     * The size in bytes of statically-allocated shared memory required by
     * this function. This does not include dynamically-allocated shared
     * memory requested by the user at runtime.
     */
    public static final int CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1;

    /**
     * The size in bytes of user-allocated constant memory required by this
     * function.
     */
    public static final int CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2;

    /**
     * The size in bytes of thread local memory used by this function.
     */
    public static final int CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3;

    /**
     * The number of registers used by each thread of this function.
     */
    public static final int CU_FUNC_ATTRIBUTE_NUM_REGS = 4;

    /**
     * The PTX virtual architecture version for which the function was compiled.
     */
    public static final int CU_FUNC_ATTRIBUTE_PTX_VERSION = 5;

    /**
     * The binary version for which the function was compiled.
     */
    public static final int CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6;

    /**
     * The attribute to indicate whether the function has been compiled with 
     * user specified option "-Xptxas --dlcm=ca" set .
     */
    public static final int CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7;
    
    //public static final int CU_FUNC_ATTRIBUTE_MAX = 8;


    /**
     * Returns the String identifying the given CUfunction_attribute
     *
     * @param n The CUfunction_attribute
     * @return The String identifying the given CUfunction_attribute
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: return "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK";
            case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: return "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES";
            case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: return "CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES";
            case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: return "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES";
            case CU_FUNC_ATTRIBUTE_NUM_REGS: return "CU_FUNC_ATTRIBUTE_NUM_REGS";
            case CU_FUNC_ATTRIBUTE_PTX_VERSION: return "CU_FUNC_ATTRIBUTE_PTX_VERSION";
            case CU_FUNC_ATTRIBUTE_BINARY_VERSION: return "CU_FUNC_ATTRIBUTE_BINARY_VERSION";
            case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA: return "CU_FUNC_ATTRIBUTE_CACHE_MODE_CA";
        }
        return "INVALID CUfunction_attribute: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUfunction_attribute()
    {
    }

};

