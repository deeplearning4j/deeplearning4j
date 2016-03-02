/*
 * JCurand - Java bindings for CURAND, the NVIDIA CUDA random
 * number generation library, to be used with JCuda
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
package jcuda.jcurand;

/**
 * CURAND function call status types
 */
public class curandStatus
{
    /**
     * No errors
     */
    public static final int CURAND_STATUS_SUCCESS = 0;
    /**
     * Header file and linked library version do not match
     */
    public static final int CURAND_STATUS_VERSION_MISMATCH = 100;
    /**
     * Generator not initialized
     */
    public static final int CURAND_STATUS_NOT_INITIALIZED = 101;
    /**
     * Memory allocation failed
     */
    public static final int CURAND_STATUS_ALLOCATION_FAILED = 102;
    /**
     * Generator is wrong type
     */
    public static final int CURAND_STATUS_TYPE_ERROR = 103;
    /**
     * Argument out of range
     */
    public static final int CURAND_STATUS_OUT_OF_RANGE = 104;
    /**
     * Length requested is not a multple of dimension
     */
    public static final int CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105;
    /**
     * GPU does not have double precision required by MRG32k3a
     */
    public static final int CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106;
    /**
     * Kernel launch failure
     */
    public static final int CURAND_STATUS_LAUNCH_FAILURE = 201;
    /**
     * Preexisting failure on library entry
     */
    public static final int CURAND_STATUS_PREEXISTING_FAILURE = 202;
    /**
     * Initialization of CUDA failed
     */
    public static final int CURAND_STATUS_INITIALIZATION_FAILED = 203;
    /**
     * Architecture mismatch, GPU does not support requested feature
     */
    public static final int CURAND_STATUS_ARCH_MISMATCH = 204;
    /**
     * Internal library error
     */
    public static final int CURAND_STATUS_INTERNAL_ERROR = 999;

    /**
     * Private constructor to prevent instantiation
     */
    private curandStatus(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
            case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";
            case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
            case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";
            case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
            case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
            case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
            case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
            case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
            case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
            case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
            case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
            case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
        }
        return "INVALID curandStatus";
    }
}

