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

/**
 * Java port of a CUmodule.
 *
 * @see jcuda.driver.JCudaDriver#cuModuleLoadData
 * @see jcuda.driver.JCudaDriver#cuModuleLoadFatBinary
 * @see jcuda.driver.JCudaDriver#cuModuleUnload
 * @see jcuda.driver.JCudaDriver#cuModuleGetFunction
 * @see jcuda.driver.JCudaDriver#cuModuleGetGlobal
 * @see jcuda.driver.JCudaDriver#cuModuleGetTexRef
 */
public class CUmodule extends NativePointerObject
{
    /**
     * Creates a new, uninitialized CUmodule
     */
    public CUmodule()
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
        return "CUmodule["+
            "nativePointer=0x"+Long.toHexString(getNativePointer())+"]";
    }

}
