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
 * CUDA texture descriptor
 */
public class cudaTextureDesc
{
    /**
     * Texture address mode for up to 3 dimensions
     * 
     * @see cudaTextureAddressMode
     */
    public int addressMode[] = new int[3];
    
    /**
     * Texture filter mode
     * 
     * @see cudaTextureFilterMode
     */
    public int filterMode;

    /**
     * Texture read mode
     * 
     * @see cudaTextureReadMode
     */
    public int readMode;
    
    /**
     * Perform sRGB->linear conversion during texture read
     */
    public int sRGB;
    
    /**
     * Indicates whether texture reads are normalized or not
     */
    public int normalizedCoords;
    
    /**
     * Limit to the anisotropy ratio
     */
    public int maxAnisotropy;
    
    /**
     * Mipmap filter mode
     * 
     * @see cudaTextureFilterMode 
     */
    public int mipmapFilterMode;
    
    /**
     * Offset applied to the supplied mipmap level
     */
    public float mipmapLevelBias;

    /**
     * Lower end of the mipmap level range to clamp access to
     */
    public float minMipmapLevelClamp;

    /**
     * Upper end of the mipmap level range to clamp access to
     */
    public float maxMipmapLevelClamp;

    /**
     * Creates a new, uninitialized cudaTextureDesc
     */
    public cudaTextureDesc()
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
        return "cudaTextureDesc["+createString(",")+"]";
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
        sb.append("addressMode=["+
            cudaTextureAddressMode.stringFor(addressMode[0])+","+
            cudaTextureAddressMode.stringFor(addressMode[1])+","+
            cudaTextureAddressMode.stringFor(addressMode[2])+"]"+f);
        sb.append("filterMode="+cudaTextureFilterMode.stringFor(filterMode)+f);
        sb.append("readMode="+cudaTextureReadMode.stringFor(readMode)+f);
        sb.append("sRGB=["+sRGB+f);
        sb.append("normalizedCoords="+normalizedCoords+f);
        sb.append("maxAnisotropy="+maxAnisotropy+f);
        sb.append("mipmapFilterMode="+cudaTextureFilterMode.stringFor(mipmapFilterMode)+f);
        sb.append("mipmapLevelBias="+mipmapLevelBias+f);
        sb.append("minMipmapLevelClamp="+minMipmapLevelClamp+f);
        sb.append("maxMipmapLevelClamp="+maxMipmapLevelClamp+f);
        return sb.toString();
    }
    
}
