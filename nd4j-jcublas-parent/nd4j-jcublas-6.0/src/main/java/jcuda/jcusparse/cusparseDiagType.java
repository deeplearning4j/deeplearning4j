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
 * Indicates if the matrix diagonal entries are unity. 
 * The diagonal elements are always assumed to be present, but 
 * if CUSPARSE_DIAG_TYPE_UNIT is passed to an API routine, 
 * then the routine will assume that all diagonal entries are 
 * unity and will not read or modify those entries.
 */
public class cusparseDiagType
{
    /**
     * The matrix diagonal has non-unit elements.
     */
    public static final int CUSPARSE_DIAG_TYPE_NON_UNIT = 0;
    
    /**
     * The matrix diagonal has unit elements.
     */
    public static final int CUSPARSE_DIAG_TYPE_UNIT = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseDiagType(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_DIAG_TYPE_NON_UNIT: return "CUSPARSE_DIAG_TYPE_NON_UNIT";
            case CUSPARSE_DIAG_TYPE_UNIT: return "CUSPARSE_DIAG_TYPE_UNIT";
        }
        return "INVALID cusparseDiagType: "+n;
    }
}

