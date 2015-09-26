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

package jcuda.driver;

/**
 * Texture reference filtering modes.
 *
 * @see JCudaDriver#cuTexRefSetFilterMode
 * @see JCudaDriver#cuTexRefGetFilterMode
 */
public class CUfilter_mode
{

    /**
     * Point filter mode
     */
    public static final int CU_TR_FILTER_MODE_POINT = 0;

    /**
     * Linear filter mode
     */
    public static final int CU_TR_FILTER_MODE_LINEAR = 1;


    /**
     * Returns the String identifying the given CUfilter_mode
     *
     * @param n The CUfilter_mode
     * @return The String identifying the given CUfilter_mode
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_TR_FILTER_MODE_POINT: return "CU_TR_FILTER_MODE_POINT";
            case CU_TR_FILTER_MODE_LINEAR: return "CU_TR_FILTER_MODE_LINEAR";
        }
        return "INVALID CUfilter_mode: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUfilter_mode()
    {
    }

}
