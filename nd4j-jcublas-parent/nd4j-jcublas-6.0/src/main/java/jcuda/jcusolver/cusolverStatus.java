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
package jcuda.jcusolver;

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

