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

import jcuda.Pointer;

/**
 * CUDA resource descriptor
 *
 * NOTE: The structure of this class should be considered as being
 * preliminary. In the C version, the internal fields are stored
 * as a union. Depending on whether the union storage had a reason,
 * the structure of this class might change accordingly.
 */
public class cudaResourceDesc
{
    /**
     * Resource type
     *
     * @see cudaResourceType
     */
    public int resType;

    /**
     * CUDA array for {@link cudaResourceType#cudaResourceTypeArray}
     */
    public cudaArray array_array = new cudaArray();

    /**
     * CUDA mipmapped array for {@link cudaResourceType#cudaResourceTypeMipmappedArray}
     */
    public cudaMipmappedArray mipmap_mipmap = new cudaMipmappedArray();

    /**
     * Device pointer for {@link cudaResourceType#cudaResourceTypeLinear}
     */
    public Pointer linear_devPtr = new Pointer();

    /**
     * Channel descriptor for {@link cudaResourceType#cudaResourceTypeLinear}
     */
    public cudaChannelFormatDesc linear_desc = new cudaChannelFormatDesc();

    /**
     * Size in bytes for {@link cudaResourceType#cudaResourceTypeLinear}
     */
    public long linear_sizeInBytes;

    /**
     * Device pointer for {@link cudaResourceType#cudaResourceTypePitch2D}
     */
    public Pointer pitch2D_devPtr = new Pointer();

    /**
     * Channel descriptor for {@link cudaResourceType#cudaResourceTypePitch2D}
     */
    public cudaChannelFormatDesc pitch2D_desc = new cudaChannelFormatDesc();

    /**
     * Width of the array in elements for {@link cudaResourceType#cudaResourceTypePitch2D}
     */
    public long pitch2D_width;

    /**
     * Height of the array in elements for {@link cudaResourceType#cudaResourceTypePitch2D}
     */
    public long pitch2D_height;

    /**
     * Pitch between two rows in bytes for {@link cudaResourceType#cudaResourceTypePitch2D}
     */
    public long pitch2D_pitchInBytes;

    /**
     * Creates a new, uninitialized cudaResourceDesc
     */
    public cudaResourceDesc()
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
        return "cudaResourceDesc["+createString(",")+"]";
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
            case cudaResourceType.cudaResourceTypeArray:
                sb.append("array="+array_array+f);
                break;

            case  cudaResourceType.cudaResourceTypeMipmappedArray:
                sb.append("mipmap="+mipmap_mipmap+f);
                break;

            case  cudaResourceType.cudaResourceTypeLinear:
                sb.append("devPtr="+linear_devPtr+f);
                sb.append("format="+linear_desc+f);
                sb.append("sizeInBytes="+linear_sizeInBytes+f);
                break;

            case  cudaResourceType.cudaResourceTypePitch2D:
                sb.append("devPtr="+pitch2D_devPtr+f);
                sb.append("format="+pitch2D_desc+f);
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

