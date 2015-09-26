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
