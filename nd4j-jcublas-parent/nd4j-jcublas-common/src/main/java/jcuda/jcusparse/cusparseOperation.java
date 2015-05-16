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
 * Indicates which operations need to be performed with the 
 * sparse matrix.
 */
public class cusparseOperation
{
    /**
     * The non-transpose operation is selected.
     */
    public static final int CUSPARSE_OPERATION_NON_TRANSPOSE = 0;
    
    /**
     * The transpose operation is selected.
     */
    public static final int CUSPARSE_OPERATION_TRANSPOSE = 1;
    
    /**
     * The conjugate transpose operation is selected.
     */
    public static final int CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseOperation(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_OPERATION_NON_TRANSPOSE: return "CUSPARSE_OPERATION_NON_TRANSPOSE";
            case CUSPARSE_OPERATION_TRANSPOSE: return "CUSPARSE_OPERATION_TRANSPOSE";
            case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE: return "CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE";
        }
        return "INVALID cusparseOperation";
    }
}

