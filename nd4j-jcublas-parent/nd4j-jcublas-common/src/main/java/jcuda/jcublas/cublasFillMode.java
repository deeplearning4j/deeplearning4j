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
 * Indicates which part (lower or upper) of the dense matrix was
 * filled and consequently should be used by the function
 */
public class cublasFillMode
{
    /**
     * The lower part of the matrix is filled
     */
    public static final int CUBLAS_FILL_MODE_LOWER = 0;

    /**
     * The upper part of the matrix is filled
     */
    public static final int CUBLAS_FILL_MODE_UPPER = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cublasFillMode(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUBLAS_FILL_MODE_LOWER: return "CUBLAS_FILL_MODE_LOWER";
            case CUBLAS_FILL_MODE_UPPER: return "CUBLAS_FILL_MODE_UPPER";
        }
        return "INVALID cublasFillMode: "+n;
    }
}

