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

/** CUSOLVERRF triangular solve algorithm */
public class cusolverRfTriangularSolve
{
    public static final int CUSOLVERRF_TRIANGULAR_SOLVE_ALG0 = 0;
    /**
     * default
     */
    public static final int CUSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1;
    public static final int CUSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2;
    public static final int CUSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3;

    /**
     * Private constructor to prevent instantiation
     */
    private cusolverRfTriangularSolve(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSOLVERRF_TRIANGULAR_SOLVE_ALG0: return "CUSOLVERRF_TRIANGULAR_SOLVE_ALG0";
            case CUSOLVERRF_TRIANGULAR_SOLVE_ALG1: return "CUSOLVERRF_TRIANGULAR_SOLVE_ALG1";
            case CUSOLVERRF_TRIANGULAR_SOLVE_ALG2: return "CUSOLVERRF_TRIANGULAR_SOLVE_ALG2";
            case CUSOLVERRF_TRIANGULAR_SOLVE_ALG3: return "CUSOLVERRF_TRIANGULAR_SOLVE_ALG3";
        }
        return "INVALID cusolverRfTriangularSolve: "+n;
    }
}

