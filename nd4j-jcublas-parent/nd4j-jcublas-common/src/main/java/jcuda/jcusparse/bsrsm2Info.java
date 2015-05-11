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
package jcuda.jcusparse;

import jcuda.NativePointerObject;

/**
 * Java port of a bsrsm2Info
 */
public class bsrsm2Info extends NativePointerObject
{
    /**
     * Creates a new, uninitialized bsrsm2Info
     */
    public bsrsm2Info()
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
        return "bsrsm2Info["+
            "nativePointer=0x"+Long.toHexString(getNativePointer())+"]";
    }
}


