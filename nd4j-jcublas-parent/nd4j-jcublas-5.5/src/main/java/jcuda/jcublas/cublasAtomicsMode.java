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
 * The type indicates whether CUBLAS routines which has an alternate 
 * implementation using atomics can be used. The atomics mode can 
 * be set and queried using and routines, respectively. 
 *  
 * @see JCublas2#cublasSetAtomicsMode(cublasHandle, int)
 */
public class cublasAtomicsMode
{
    /**
     * Atomics are not allowed
     */
    public static final int CUBLAS_ATOMICS_NOT_ALLOWED = 0;

    /**
     * Atomics are allowed
     */
    public static final int CUBLAS_ATOMICS_ALLOWED = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cublasAtomicsMode(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUBLAS_ATOMICS_NOT_ALLOWED: return "CUBLAS_ATOMICS_NOT_ALLOWED";
            case CUBLAS_ATOMICS_ALLOWED: return "CUBLAS_ATOMICS_ALLOWED";
        }
        return "INVALID cublasAtomicsMode: "+n;
    }
}

