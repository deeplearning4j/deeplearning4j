/*
 * JCusparse - Java bindings for CUSPARSE, the NVIDIA CUDA sparse
 * matrix library, to be used with JCuda
 *
 * Copyright (c) 2010-2015 Marco Hutter - http://www.jcuda.org
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
package jcuda.jcusparse;

/**
 * CUSPARSE status return values
 */
public class cusparseStatus
{
    /**
     * The operation completed successfully.
     */
    public static final int CUSPARSE_STATUS_SUCCESS = 0;

    /**
     * The CUSPARSE library was not initialized.
     */
    public static final int CUSPARSE_STATUS_NOT_INITIALIZED = 1;

    /**
     * Resource allocation failed inside the CUSPARSE library.
     */
    public static final int CUSPARSE_STATUS_ALLOC_FAILED = 2;

    /**
     * An unsupported value or parameter was passed to the function.
     */
    public static final int CUSPARSE_STATUS_INVALID_VALUE = 3;

    /**
     * The function requires a feature absent from the device architecture.
     */
    public static final int CUSPARSE_STATUS_ARCH_MISMATCH = 4;

    /**
     * An access to GPU memory space failed, which is usually caused
     * by a failure to bind a texture.
     */
    public static final int CUSPARSE_STATUS_MAPPING_ERROR = 5;

    /**
     * The GPU program failed to execute.
     */
    public static final int CUSPARSE_STATUS_EXECUTION_FAILED = 6;

    /**
     * An internal CUSPARSE operation failed.
     */
    public static final int CUSPARSE_STATUS_INTERNAL_ERROR = 7;

    /**
     * The matrix type is not supported by this function.
     */
    public static final int CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8;

    /**
     * An entry of the matrix is either structural zero or numerical
     * zero (singular block)
     */
    public static final int CUSPARSE_STATUS_ZERO_PIVOT = 9;

    /**
     * An internal JCusparse error occurred
     */
    public static final int JCUSPARSE_STATUS_INTERNAL_ERROR = -1;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseStatus(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
            case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
            case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
            case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
            case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
            case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
            case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
            case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
            case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT";
            case JCUSPARSE_STATUS_INTERNAL_ERROR: return "JCUSPARSE_STATUS_INTERNAL_ERROR";
        }
        return "INVALID cusparseStatus: "+n;
    }
}

