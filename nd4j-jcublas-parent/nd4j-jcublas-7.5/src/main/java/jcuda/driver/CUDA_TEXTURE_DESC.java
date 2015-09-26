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
