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

import jcuda.Pointer;

/**
 * Java port of a cudaPitchedPtr
 *
 * @see JCuda#cudaMalloc3D(cudaPitchedPtr, cudaExtent)
 * @see JCuda#cudaMemcpy3D(cudaMemcpy3DParms)
 * @see JCuda#cudaMemset3D(cudaPitchedPtr, int, cudaExtent)
 */
public class cudaPitchedPtr
{
    /**
     * Pointer to allocated memory.
     */
    public Pointer ptr = new Pointer();

    /**
     * The pitch of the pointer, in bytes
     */
    public long pitch;

    /**
     * xsize and ysize, the logical width and height of the, are equivalent to the
     * width and height extent parameters provided by the programmer during allocation
     */
    public long xsize;

    /**
     * xsize and ysize, the logical width and height of the, are equivalent to the
     * width and height extent parameters provided by the programmer during allocation
     */
    public long ysize;

    /**
     * Creates a new, uninitialized cudaPitchedPtr
     */
    public cudaPitchedPtr()
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
        return "cudaPitchedPtr["+
            "ptr="+ptr+","+
            "pitch="+pitch+","+
            "xsize="+xsize+","+
            "ysize="+ysize+"]";
    }

}
