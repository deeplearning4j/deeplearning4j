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
 * Partition modes
 */
public class cusparseHybPartition
{
    /**
     * Automatically decide how to split the data into regular/irregular part
     */
    public static final int CUSPARSE_HYB_PARTITION_AUTO = 0;
    /**
     * Store data into regular part up to a user specified threshold
     */
    public static final int CUSPARSE_HYB_PARTITION_USER = 1;
    /**
     * Store all data in the regular part
     */
    public static final int CUSPARSE_HYB_PARTITION_MAX = 2;

    /**
     * Private constructor to prevent instantiation
     */
    private cusparseHybPartition(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUSPARSE_HYB_PARTITION_AUTO: return "CUSPARSE_HYB_PARTITION_AUTO";
            case CUSPARSE_HYB_PARTITION_USER: return "CUSPARSE_HYB_PARTITION_USER";
            case CUSPARSE_HYB_PARTITION_MAX: return "CUSPARSE_HYB_PARTITION_MAX";
        }
        return "INVALID cusparseHybPartition: "+n;
    }
}

