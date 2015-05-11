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
package jcuda.runtime;


/**
 * CUDA resource types
 */
public class cudaResourceType
{
    /**
     * Array resource 
     */
    public static final int cudaResourceTypeArray          = 0x00;
    
    /**
     * Mipmapped array resource 
     */
    public static final int cudaResourceTypeMipmappedArray = 0x01; 
    
    /**
     * Linear resource 
     */
    public static final int cudaResourceTypeLinear         = 0x02; 
    
    /**
     * Pitch 2D resource 
     */
    public static final int cudaResourceTypePitch2D        = 0x03;  
    
    /**
     * Returns the String identifying the given cudaResourceType
     *
     * @param k The cudaResourceType
     * @return The String identifying the given cudaResourceType
     */
    public static String stringFor(int k)
    {
        switch (k)
        {
            case cudaResourceTypeArray: return "cudaResourceTypeArray";
            case cudaResourceTypeMipmappedArray: return "cudaResourceTypeMipmappedArray";
            case cudaResourceTypeLinear: return "cudaResourceTypeLinear";
            case cudaResourceTypePitch2D: return "cudaResourceTypePitch2D";
        }
        return "INVALID cudaResourceType: "+k;
    }
    
    
    /**
     * Private constructor to prevent instantiation.
     */
    private cudaResourceType()
    {
        
    }
};
