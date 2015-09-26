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
 * Indicates whether the dense matrix is on the left or right side
 * in the matrix equation solved by a particular function.
 */
public class cublasSideMode
{
    /**
     * The matrix is on the left side in the equation
     */
    public static final int CUBLAS_SIDE_LEFT = 0;

    /**
     * The matrix is on the right side in the equation
     */
    public static final int CUBLAS_SIDE_RIGHT = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cublasSideMode(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUBLAS_SIDE_LEFT: return "CUBLAS_SIDE_LEFT";
            case CUBLAS_SIDE_RIGHT: return "CUBLAS_SIDE_RIGHT";
        }
        return "INVALID cublasSideMode: "+n;
    }
}

