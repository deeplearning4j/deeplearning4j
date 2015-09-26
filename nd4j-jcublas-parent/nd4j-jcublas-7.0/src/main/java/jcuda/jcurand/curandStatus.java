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

