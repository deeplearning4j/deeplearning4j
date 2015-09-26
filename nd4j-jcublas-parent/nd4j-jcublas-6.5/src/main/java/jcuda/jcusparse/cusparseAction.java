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
 * Indicates whether the operation is performed only on indices 
 * or on data and indices.
 */
public class cusparseAction
{
    /**
     * The operation is performed only on indices.
     */
    public static final int CUSPARSE_ACTION_SYMBOLIC = 0;
    
    /**
     * The operation is performed on data and indices.
     */
    public static final int CUSPARSE_ACTION_NUMERIC = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseAction(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_ACTION_SYMBOLIC: return "CUSPARSE_ACTION_SYMBOLIC";
            case CUSPARSE_ACTION_NUMERIC: return "CUSPARSE_ACTION_NUMERIC";
        }
        return "INVALID cusparseAction: "+n;
    }
}

