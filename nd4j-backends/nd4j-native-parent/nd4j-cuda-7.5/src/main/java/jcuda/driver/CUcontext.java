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

import jcuda.NativePointerObject;

/**
 * Java port of a CUcontext.
 *
 * @see jcuda.driver.JCudaDriver#cuCtxCreate
 * @see jcuda.driver.JCudaDriver#cuCtxAttach
 * @see jcuda.driver.JCudaDriver#cuCtxDetach
 * @see jcuda.driver.JCudaDriver#cuCtxDestroy
 * @see jcuda.driver.JCudaDriver#cuCtxPushCurrent
 * @see jcuda.driver.JCudaDriver#cuCtxPopCurrent
 */
public class CUcontext extends NativePointerObject
{

    /**
     * Creates a new, uninitialized CUcontext
     */
    public CUcontext()
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
        return "CUcontext["+
            "nativePointer=0x"+Long.toHexString(getNativePointer())+"]";
    }

}
