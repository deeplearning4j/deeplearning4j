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
 * Texture descriptor
 */
public class CUDA_TEXTURE_DESC
{
    /**
     * Address modes 
     * 
     * @see CUaddress_mode 
     */
    public int addressMode[] = new int[3];
    
    /**
     * Filter mode 
     * 
     * @see CUfilter_mode 
     */
    public int filterMode;       
    
    /** 
     * Flags
     * 
     * @see JCudaDriver#CU_TRSF_READ_AS_INTEGER
     * @see JCudaDriver#CU_TRSF_NORMALIZED_COORDINATES
     */
    public int flags;             
    
    /** 
     * Maximum anisotropy ratio 
     */
    public int maxAnisotropy;    
    
    /**
     * Mipmap filter mode 
     * 
     * @see CUfilter_mode 
     */
    public int mipmapFilterMode; 
    
    /**
     * Mipmap level bias 
     */
    public float mipmapLevelBias;          
    
    /**
     * Mipmap minimum level clamp 
     */
    public float minMipmapLevelClamp;      
    
    /**
     * Mipmap maximum level clamp 
     */
    public float maxMipmapLevelClamp;      
    
    // private int _reserved[] = new int[16];
    
    /**
     * Creates a new, uninitialized CUDA_TEXTURE_DESC
     */
    public CUDA_TEXTURE_DESC()
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
        return "CUDA_TEXTURE_DESC["+createString(",")+"]";
    }

    /**
     * Creates and returns a formatted (aligned, multi-line) String
     * representation of this object
     *
     * @return A formatted String representation of this object
     */
    public String toFormattedString()
    {
        return "CUDA texture descriptor:\n    "+createString("\n    ");
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
        sb.append("addressMode="+"("+
            CUaddress_mode.stringFor(addressMode[0])+","+
            CUaddress_mode.stringFor(addressMode[1])+","+
            CUaddress_mode.stringFor(addressMode[2])+")"+f);
        sb.append("filterMode="+CUfilter_mode.stringFor(filterMode)+f);
        String flagsString = "";
        if ((flags & JCudaDriver.CU_TRSF_READ_AS_INTEGER) != 0)
        {
            flagsString += "CU_TRSF_READ_AS_INTEGER";
        }
        if ((flags & JCudaDriver.CU_TRSF_NORMALIZED_COORDINATES) != 0)
        {
            flagsString += "CU_TRSF_NORMALIZED_COORDINATES";
        }
        sb.append("flags="+flags+"("+flagsString+")");
        sb.append("maxAnisotropy="+maxAnisotropy);
        sb.append("mipmapFilterMode="+CUfilter_mode.stringFor(mipmapFilterMode)+f);
        sb.append("mipmapLevelBias="+mipmapLevelBias+f);
        sb.append("minMipmapLevelClamp="+minMipmapLevelClamp+f);
        sb.append("maxMipmapLevelClamp="+maxMipmapLevelClamp+f);
        return sb.toString();
    }

    
}
