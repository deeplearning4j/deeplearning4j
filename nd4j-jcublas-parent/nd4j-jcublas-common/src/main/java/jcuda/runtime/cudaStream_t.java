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

import jcuda.NativePointerObject;
import jcuda.driver.CUstream;

/**
 * Java port of a cudaStream_t.
 *
 * @see JCuda#cudaStreamCreate
 * @see JCuda#cudaStreamQuery
 * @see JCuda#cudaStreamSynchronize
 * @see JCuda#cudaStreamDestroy
 */
public class cudaStream_t extends NativePointerObject
{
    /**
     * Creates a new, uninitialized cudaStream_t
     */
    public cudaStream_t()
    {
    }
    
    /**
     * Creates a cudaStream_t for the given {@link CUstream}. This
     * corresponds to casting a CUstream to a cudaStream_t.
     * 
     * @param stream The other stream
     */
    public cudaStream_t(CUstream stream)
    {
        super(stream);
    }
    
    /**
     * Create a cudaStream_t that is a constant with the given 
     * value. This is used for emulating the stream
     * handling constants, {@link JCuda#cudaStreamLegacy} 
     * and {@link JCuda#cudaStreamPerThread()}
     * 
     * @param value The pointer value
     */
    cudaStream_t(long value)
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
        return "cudaStream_t["+
            "nativePointer="+getNativePointer()+"]";
    }

}
