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

import jcuda.NativePointerObject;
import jcuda.runtime.cudaStream_t;

/**
 * Java port of a CUstream.
 *
 * @see JCudaDriver#cuStreamCreate
 * @see JCudaDriver#cuStreamQuery
 * @see JCudaDriver#cuStreamSynchronize
 * @see JCudaDriver#cuStreamDestroy
 */
public class CUstream extends NativePointerObject
{
    /**
     * Creates a new, uninitialized CUstream
     */
    public CUstream()
    {
    }
    
    /**
     * Creates a CUstream for the given {@link cudaStream_t}. This
     * corresponds to casting a cudaStream_t to a CUstream.
     * 
     * @param stream The other stream
     */
    public CUstream(cudaStream_t stream)
    {
        super(stream);
    }
    
    /**
     * Create a CUstream that is a constant with the given 
     * value. This is used for emulating the stream
     * handling constants, {@link JCudaDriver#CU_STREAM_LEGACY} 
     * and {@link JCudaDriver#CU_STREAM_PER_THREAD}
     * 
     * @param value The pointer value
     */
    CUstream(long value)
    {
        super(value);
    }

    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "CUstream["+
            "nativePointer=0x"+Long.toHexString(getNativePointer())+"]";
    }

}
