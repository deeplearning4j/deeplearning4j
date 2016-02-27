/*
 * JCusolver - Java bindings for CUSOLVER, the NVIDIA CUDA solver
 * library, to be used with JCuda
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
package jcusolver;

public class cusolverStatus
{
    /**
     * The operation completed successfully.
     */
    public static final int CUSOLVER_STATUS_SUCCESS = 0;

    /**
     * The cuSolver library was not initialized. This is usually caused by the
     * lack of a prior call, an error in the CUDA Runtime API called by the
     * cuSolver routine, or an error in the hardware setup.
     * To correct: call  cusolverCreate() prior to the function call; and
     * check that the hardware, an appropriate version of the driver, and the
     * cuSolver library are correctly installed.
     */
    public static final int CUSOLVER_STATUS_NOT_INITIALIZED = 1;

    /**
     * Resource allocation failed inside the cuSolver library. This is usually
     * caused by a  cudaMalloc() failure.
     * To correct: prior to the function call, deallocate previously allocated
     * memory as much as possible.
     */
    public static final int CUSOLVER_STATUS_ALLOC_FAILED = 2;

    /**
     * An unsupported value or parameter was passed to the function (a
     * negative vector size, for example).
     * To correct: ensure that all the parameters being passed have valid
     * values.
     */
    public static final int CUSOLVER_STATUS_INVALID_VALUE = 3;

    /**
     * The function requires a feature absent from the device architecture;
     * usually caused by the lack of support for atomic operations or double
     * precision.
     * To correct: compile and run the application on a device with compute
     * capability 2.0 or above.
     */
    public static final int CUSOLVER_STATUS_ARCH_MISMATCH = 4;

    /**
     * Mapping error
     */
    public static final int CUSOLVER_STATUS_MAPPING_ERROR = 5;

    /**
     * The GPU program failed to execute. This is often caused by a launch
     * failure of the kernel on the GPU, which can be caused by multiple
     * reasons.
     * To correct: check that the hardware, an appropriate version of the
     * driver, and the cuSolver library are correctly installed.
     */
    public static final int CUSOLVER_STATUS_EXECUTION_FAILED = 6;

    /**
     * An internal cuSolver operation failed. This error is usually caused by a
     * cudaMemcpyAsync() failure.
     * To correct: check that the hardware, an appropriate version of the
     * driver, and the cuSolver library are correctly installed. Also, check
     * that the memory passed as a parameter to the routine is not being
     * deallocated prior to the routine's completion.
     */
    public static final int CUSOLVER_STATUS_INTERNAL_ERROR = 7;

    /**
     * The matrix type is not supported by this function. This is usually caused
     * by passing an invalid matrix descriptor to the function.
     * To correct: check that the fields in  descrA were set correctly.
     */
    public static final int CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8;

    /**
     * Status not supported
     */
    public static final int CUSOLVER_STATUS_NOT_SUPPORTED = 9;

    /**
     * Zero pivot
     */
    public static final int CUSOLVER_STATUS_ZERO_PIVOT = 10;

    /**
     * Invalid license
     */
    public static final int CUSOLVER_STATUS_INVALID_LICENSE = 11;

    /**
     * Private constructor to prevent instantiation
     */
    private cusolverStatus(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
            case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
            case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
            case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
            case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
            case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
            case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
            case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
            case CUSOLVER_STATUS_ZERO_PIVOT: return "CUSOLVER_STATUS_ZERO_PIVOT";
            case CUSOLVER_STATUS_INVALID_LICENSE: return "CUSOLVER_STATUS_INVALID_LICENSE";
        }
        return "INVALID cusolverStatus: "+n;
    }
}

