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

/** CUSOLVERRF matrix format */
public class cusolverRfMatrixFormat
{
    /**
     * default
     */
    public static final int CUSOLVERRF_MATRIX_FORMAT_CSR = 0;
    public static final int CUSOLVERRF_MATRIX_FORMAT_CSC = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cusolverRfMatrixFormat(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSOLVERRF_MATRIX_FORMAT_CSR: return "CUSOLVERRF_MATRIX_FORMAT_CSR";
            case CUSOLVERRF_MATRIX_FORMAT_CSC: return "CUSOLVERRF_MATRIX_FORMAT_CSC";
        }
        return "INVALID cusolverRfMatrixFormat: "+n;
    }
}

