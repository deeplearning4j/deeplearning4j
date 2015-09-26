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
 * Indicates whether the elements of a dense matrix should be 
 * parsed by rows or by columns.
 */
public class cusparseDirection
{
    /**
     * The matrix should be parsed by rows.
     */
    public static final int CUSPARSE_DIRECTION_ROW = 0;
    
    /**
     * The matrix should be parsed by columns
     */
    public static final int CUSPARSE_DIRECTION_COLUMN = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseDirection(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_DIRECTION_ROW: return "CUSPARSE_DIRECTION_ROW";
            case CUSPARSE_DIRECTION_COLUMN: return "CUSPARSE_DIRECTION_COLUMN";
        }
        return "INVALID cusparseDirection: "+n;
    }
}

