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

/** CUSOLVERRF unit diagonal */
public class cusolverRfUnitDiagonal
{
    /**
     * default
     */
    public static final int CUSOLVERRF_UNIT_DIAGONAL_STORED_L = 0;
    public static final int CUSOLVERRF_UNIT_DIAGONAL_STORED_U = 1;
    public static final int CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2;
    public static final int CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3;

    /**
     * Private constructor to prevent instantiation
     */
    private cusolverRfUnitDiagonal(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSOLVERRF_UNIT_DIAGONAL_STORED_L: return "CUSOLVERRF_UNIT_DIAGONAL_STORED_L";
            case CUSOLVERRF_UNIT_DIAGONAL_STORED_U: return "CUSOLVERRF_UNIT_DIAGONAL_STORED_U";
            case CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L: return "CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L";
            case CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U: return "CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U";
        }
        return "INVALID cusolverRfUnitDiagonal: "+n;
    }
}

