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
 * CUDA Resource descriptor. 
 * 
 * NOTE: The structure of this class should be considered as being
 * preliminary. In the C version, the internal fields are stored 
 * as a union. Depending on whether the union storage had a reason,
 * the structure of this class might change accordingly.  
 */
public class CUDA_RESOURCE_DESC
{
    /**
     * Resource type 
     * 
     * @see CUresourcetype 
     */
    public int resType;                   

    /**
     * Flags (must be zero) 
     */
    public int flags;                       
    
    /**
     * CUDA array for {@link CUresourcetype#CU_RESOURCE_TYPE_ARRAY} 
     */
    public CUarray array_hArray = new CUarray();                   

    /** 
     * CUDA mipmapped array for {@link CUresourcetype#CU_RESOURCE_TYPE_MIPMAPPED_ARRAY}
     */
    public CUmipmappedArray mipmap_hMipmappedArray; 

    /** 
     * Device pointer for {@link CUresourcetype#CU_RESOURCE_TYPE_LINEAR} 
     */
    public CUdeviceptr linear_devPtr = new CUdeviceptr();    
    
    /**
     * Array format for {@link CUresourcetype#CU_RESOURCE_TYPE_LINEAR}
     * 
     *  @see CUarray_format 
     */
    public int linear_format;            
    
    /** 
     * Channels per array element for {@link CUresourcetype#CU_RESOURCE_TYPE_LINEAR}
     */
    public int linear_numChannels;         
    
    /** 
     * Size in bytes for {@link CUresourcetype#CU_RESOURCE_TYPE_LINEAR}
     */
    public long linear_sizeInBytes;               

    /**
     * Device pointer for {@link CUresourcetype#CU_RESOURCE_TYPE_PITCH2D}
     */
    public CUdeviceptr pitch2D_devPtr = new CUdeviceptr();              
    
    /** 
     * Array format for {@link CUresourcetype#CU_RESOURCE_TYPE_PITCH2D}
     * 
     * @see CUarray_format
     */
    public int pitch2D_format;   
    
    /**
     * Channels per array element for {@link CUresourcetype#CU_RESOURCE_TYPE_PITCH2D}
     */
    public int pitch2D_numChannels;         
    
    /** 
     * Width of the array in elements for {@link CUresourcetype#CU_RESOURCE_TYPE_PITCH2D}
     */
    public long pitch2D_width;                     
    
    /**
     * Height of the array in elements for {@link CUresourcetype#CU_RESOURCE_TYPE_PITCH2D}
     */
    public long pitch2D_height;                    
    
    /** 
     * Pitch between two rows in bytes for {@link CUresourcetype#CU_RESOURCE_TYPE_PITCH2D}
     */
    public long pitch2D_pitchInBytes;              

    /**
     * Creates a new, uninitialized CUDA_RESOURCE_DESC
     */
    public CUDA_RESOURCE_DESC()
    {
        
    }
    
    
    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "CUDA_RESOURCE_DESC["+createString(",")+"]";
    }

    /**
     * Creates and returns a formatted (aligned, multi-line) String
     * representation of this object
     *
     * @return A formatted String representation of this object
     */
    public String toFormattedString()
    {
        return "CUDA resource descriptor:\n    "+createString("\n    ");
    }

    /**
     * Creates and returns a string representation of this object,
     * using the given separator for the fields
     *
     * @param f Separator
     * @return A String representation of this object
     */
    private String createString(String f)
    {
        StringBuilder sb = new StringBuilder();
        switch (resType)
        {
            case CUresourcetype.CU_RESOURCE_TYPE_ARRAY:
                sb.append("hArray="+array_hArray+f);
                break;
                
            case  CUresourcetype.CU_RESOURCE_TYPE_MIPMAPPED_ARRAY:
                sb.append("hMipmappedArray="+mipmap_hMipmappedArray+f);
                break;

            case  CUresourcetype.CU_RESOURCE_TYPE_LINEAR:
                sb.append("devPtr="+linear_devPtr+f);
                sb.append("format="+CUarray_format.stringFor(linear_format)+f);
                sb.append("numChannels="+linear_numChannels+f);
                sb.append("sizeInBytes="+linear_sizeInBytes+f);
                break;
                
            case  CUresourcetype.CU_RESOURCE_TYPE_PITCH2D:
                sb.append("devPtr="+pitch2D_devPtr+f);
                sb.append("format="+CUarray_format.stringFor(pitch2D_format)+f);
                sb.append("numChannels="+pitch2D_numChannels+f);
                sb.append("width="+pitch2D_width+f);
                sb.append("height="+pitch2D_height+f);
                sb.append("pitchInBytes="+pitch2D_pitchInBytes+f);
                break;
                
            default:
                sb.append("INVALID");
        }
        return sb.toString();
    }

    
}
