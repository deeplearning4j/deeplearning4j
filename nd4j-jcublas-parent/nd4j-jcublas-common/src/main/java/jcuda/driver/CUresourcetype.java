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
 * Resource types
 */
public class CUresourcetype
{
    /** 
     * Array resource 
     */
    public static final int CU_RESOURCE_TYPE_ARRAY           = 0x00; 
    
    /**
     * Mipmapped array resource 
     */
    public static final int CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01; 
    
    /**
     * Linear resource 
     */
    public static final int CU_RESOURCE_TYPE_LINEAR          = 0x02;
    
    /** 
     * Pitch 2D resource 
     */
    public static final int CU_RESOURCE_TYPE_PITCH2D         = 0x03; 
    
    /**
     * Returns the String identifying the given CUresourcetype
     *
     * @param n The CUresourcetype
     * @return The String identifying the given CUresourcetype
     */
    public static String stringFor(int n)
    {
        switch (n) 
        {
            case CU_RESOURCE_TYPE_ARRAY : return "CU_RESOURCE_TYPE_ARRAY";
            case CU_RESOURCE_TYPE_MIPMAPPED_ARRAY : return "CU_RESOURCE_TYPE_MIPMAPPED_ARRAY";
            case CU_RESOURCE_TYPE_LINEAR : return "CU_RESOURCE_TYPE_LINEAR";
            case CU_RESOURCE_TYPE_PITCH2D : return "CU_RESOURCE_TYPE_PITCH2D";
        }
        return "INVALID CUresourcetype: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUresourcetype()
    {
    }


}
