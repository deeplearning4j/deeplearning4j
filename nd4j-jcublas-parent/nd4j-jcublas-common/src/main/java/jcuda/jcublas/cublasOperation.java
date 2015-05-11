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
 * Indicates which operation needs to be performed with the
 * dense matrix.
 */
public class cublasOperation
{
    /**
     * The non-transpose operation is selected
     */
    public static final int CUBLAS_OP_N = 0;

    /**
     * The transpose operation is selected
     */
    public static final int CUBLAS_OP_T = 1;

    /**
     * The conjugate transpose operation is selected
     */
    public static final int CUBLAS_OP_C = 2;

    /**
     * Private constructor to prevent instantiation
     */
    private cublasOperation(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUBLAS_OP_N: return "CUBLAS_OP_N";
            case CUBLAS_OP_T: return "CUBLAS_OP_T";
            case CUBLAS_OP_C: return "CUBLAS_OP_C";
        }
        return "INVALID cublasOperation: "+n;
    }
}

