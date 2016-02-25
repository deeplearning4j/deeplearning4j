/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda
 *
 * Copyright (c) 2008-2015 Marco Hutter - http://www.jcuda.org
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



package jcuda.jcublas;

/**
 * JCublas status return values.
 */
public class cublasStatus
{
    /** Operation completed successfully */
    public static final int CUBLAS_STATUS_SUCCESS          = 0x00000000;

    /** Library not initialized */
    public static final int CUBLAS_STATUS_NOT_INITIALIZED  = 0x00000001;

    /** Resource allocation failed */
    public static final int CUBLAS_STATUS_ALLOC_FAILED     = 0x00000003;

    /** Unsupported numerical value was passed to function */
    public static final int CUBLAS_STATUS_INVALID_VALUE    = 0x00000007;

    /**
     * Function requires an architectural feature absent from the
     * architecture of the device
     */
    public static final int CUBLAS_STATUS_ARCH_MISMATCH    = 0x00000008;

    /** Access to GPU memory space failed */
    public static final int CUBLAS_STATUS_MAPPING_ERROR    = 0x0000000B;

    /** GPU program failed to execute */
    public static final int CUBLAS_STATUS_EXECUTION_FAILED = 0x0000000D;

    /** An internal CUBLAS operation failed */
    public static final int CUBLAS_STATUS_INTERNAL_ERROR   = 0x0000000E;

    /** The functionality requested is not supported. */
    public static final int CUBLAS_STATUS_NOT_SUPPORTED    = 0x0000000F;

    /** JCublas status returns */
    public static final int JCUBLAS_STATUS_INTERNAL_ERROR      = 0x10000001;

    /**
     * Returns the String identifying the given cublasStatus
     *
     * @param n The cublasStatus
     * @return The String identifying the given cublasStatus
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUBLAS_STATUS_SUCCESS          : return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED  : return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED     : return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE    : return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH    : return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR    : return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED : return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR   : return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED   : return "CUBLAS_STATUS_NOT_SUPPORTED";
            case JCUBLAS_STATUS_INTERNAL_ERROR  : return "JCUBLAS_STATUS_INTERNAL_ERROR";
        }
        return "INVALID cublasStatus: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cublasStatus()
    {
    }

}
