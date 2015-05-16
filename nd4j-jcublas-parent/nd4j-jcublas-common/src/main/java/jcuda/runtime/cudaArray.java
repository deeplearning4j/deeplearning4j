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
import jcuda.driver.CUarray;

/**
 * Java port of a cudaArray
 *
 * @see JCuda#cudaMallocArray
 * @see JCuda#cudaMalloc3DArray
 * @see JCuda#cudaFreeArray
 */
public class cudaArray extends NativePointerObject
{
    /**
     * Creates a new, uninitialized cudaArray
     */
    public cudaArray()
    {
    }
    
    /**
     * Creates a cudaArray for the given {@link CUarray}. This
     * corresponds to casting a CUarray to a cudaArray.
     * 
     * @param array The other array
     */
    public cudaArray(CUarray array)
    {
        super(array);
    }

    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "cudaArray["+
            "nativePointer=0x"+Long.toHexString(getNativePointer())+"]";
    }
}
