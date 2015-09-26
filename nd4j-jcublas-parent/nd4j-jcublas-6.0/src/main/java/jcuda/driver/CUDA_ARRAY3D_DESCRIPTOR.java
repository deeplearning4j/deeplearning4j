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
 * Java port of a CUDA_ARRAY3D_DESCRIPTOR.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see JCudaDriver#cuArray3DCreate
 * @see JCudaDriver#cuArray3DGetDescriptor
 *
 */
public class CUDA_ARRAY3D_DESCRIPTOR
{
    /**
     * Width is the width of the CUDA array (in elements);
     */
    public long Width;

    /**
     * Height is the height of the CUDA array (in elements); the CUDA array is
     * one-dimensional if height and depth are 0, two-dimensional if depth is
     * 0, and three-dimensional otherwise;
     */
    public long Height;

    /**
     * Depth is the depth of the CUDA array (in elements); the CUDA array is
     * one-dimensional if height and depth are 0, two-dimensional if depth is
     * 0, and three-dimensional otherwise;
     */
    public long Depth;

    /**
     * Format specifies the format of the elements; CUarray_format
     *
     * @see CUarray_format
     */
    public int Format;

    /**
     * NumChannels specifies the number of packed components per CUDA array element.; it may be 1, 2 or 4
     */
    public int NumChannels;

    /**
     * The flags for the array creation. See 
     * {@link JCudaDriver#cuArray3DCreate(CUarray, CUDA_ARRAY3D_DESCRIPTOR)}.
     */
    public int Flags;

    /**
     * Creates a new, uninitialized CUDA_ARRAY3D_DESCRIPTOR
     */
    public CUDA_ARRAY3D_DESCRIPTOR()
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
        return "CUDA_ARRAY3D_DESCRIPTOR["+
            "Width="+Width+","+
            "Height="+Height+","+
            "Depth="+Depth+","+
            "CUarray_format_Format="+Format+","+
            "NumChannels="+NumChannels+"]";
    }
}
