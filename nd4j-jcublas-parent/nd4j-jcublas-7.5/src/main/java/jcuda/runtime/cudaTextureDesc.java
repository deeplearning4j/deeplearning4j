/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2015 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
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
