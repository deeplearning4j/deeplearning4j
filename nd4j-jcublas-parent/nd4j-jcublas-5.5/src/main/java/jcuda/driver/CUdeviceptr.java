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

import jcuda.Pointer;

/**
 * Java port of a CUdeviceptr.
 */
public class CUdeviceptr extends Pointer
{
    /**
     * Creates a new (null) device pointer
     */
    public CUdeviceptr()
    {
    }

    /**
     * Copy constructor
     *
     * @param other The other pointer
     */
    protected CUdeviceptr(CUdeviceptr other)
    {
        super(other);
    }

    /**
     * Creates a copy of the given pointer, with an
     * additional byte offset
     *
     * @param other The other pointer
     * @param byteOffset The additional byte offset
     */
    protected CUdeviceptr(CUdeviceptr other, long byteOffset)
    {
        super(other, byteOffset);
    }

    @Override
    public CUdeviceptr withByteOffset(long byteOffset)
    {
        return new CUdeviceptr(this, byteOffset);
    }

    /**
     * Returns a String representation of this object.
     *
     * @return A String representation of this object.
     */
    @Override
    public String toString()
    {
        return "CUdeviceptr["+
            "nativePointer=0x"+Long.toHexString(getNativePointer())+","+
            "byteOffset="+getByteOffset()+"]";
    }


}
