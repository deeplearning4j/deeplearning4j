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

