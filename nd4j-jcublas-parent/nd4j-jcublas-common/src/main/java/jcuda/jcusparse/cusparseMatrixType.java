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
 * Indicates the type of matrix stored in sparse storage.
 */
public class cusparseMatrixType
{
    /**
     * The matrix is general.
     */
    public static final int CUSPARSE_MATRIX_TYPE_GENERAL = 0;
    
    /**
     * The matrix is symmetric.
     */
    public static final int CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1;
    
    /**
     * The matrix is Hermitian.
     */
    public static final int CUSPARSE_MATRIX_TYPE_HERMITIAN = 2;
    
    /**
     * The matrix is triangular.
     */
    public static final int CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseMatrixType(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_MATRIX_TYPE_GENERAL: return "CUSPARSE_MATRIX_TYPE_GENERAL";
            case CUSPARSE_MATRIX_TYPE_SYMMETRIC: return "CUSPARSE_MATRIX_TYPE_SYMMETRIC";
            case CUSPARSE_MATRIX_TYPE_HERMITIAN: return "CUSPARSE_MATRIX_TYPE_HERMITIAN";
            case CUSPARSE_MATRIX_TYPE_TRIANGULAR: return "CUSPARSE_MATRIX_TYPE_TRIANGULAR";
        }
        return "INVALID cusparseMatrixType: "+n;
    }
}

