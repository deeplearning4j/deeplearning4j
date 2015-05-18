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
package jcuda.jcusparse;

/**
 * Indicates if the base of the matrix indices is zero or one.
 */
public class cusparseIndexBase
{
    /**
     * The base index is zero.
     */
    public static final int CUSPARSE_INDEX_BASE_ZERO = 0;
    
    /**
     * The base index is one.
     */
    public static final int CUSPARSE_INDEX_BASE_ONE = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseIndexBase(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_INDEX_BASE_ZERO: return "CUSPARSE_INDEX_BASE_ZERO";
            case CUSPARSE_INDEX_BASE_ONE: return "CUSPARSE_INDEX_BASE_ONE";
        }
        return "INVALID cusparseIndexBase: "+n;
    }
}

