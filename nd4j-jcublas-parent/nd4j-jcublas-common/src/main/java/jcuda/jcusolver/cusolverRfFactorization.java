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

/** CUSOLVERRF factorization algorithm */
public class cusolverRfFactorization
{
    /**
     * default
     */
    public static final int CUSOLVERRF_FACTORIZATION_ALG0 = 0;
    public static final int CUSOLVERRF_FACTORIZATION_ALG1 = 1;
    public static final int CUSOLVERRF_FACTORIZATION_ALG2 = 2;

    /**
     * Private constructor to prevent instantiation
     */
    private cusolverRfFactorization(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSOLVERRF_FACTORIZATION_ALG0: return "CUSOLVERRF_FACTORIZATION_ALG0";
            case CUSOLVERRF_FACTORIZATION_ALG1: return "CUSOLVERRF_FACTORIZATION_ALG1";
            case CUSOLVERRF_FACTORIZATION_ALG2: return "CUSOLVERRF_FACTORIZATION_ALG2";
        }
        return "INVALID cusolverRfFactorization: "+n;
    }
}

