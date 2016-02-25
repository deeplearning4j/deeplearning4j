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

package jcuda;

/**
 * Base class for all classes that store a native pointer
 */
public abstract class NativePointerObject
{
    /**
     * The native pointer, written by native methods
     */
    private long nativePointer;

    /**
     * Creates a new NativePointerObject with a <code>null</code> pointer.
     */
    protected NativePointerObject()
    {
        nativePointer = 0;
    }

    /**
     * Creates a new Pointer with the given native pointer value
     */
    protected NativePointerObject(long nativePointer)
    {
        this.nativePointer = nativePointer;
    }

    /**
     * Creates a new Pointer with the samme native pointer as the
     * given one
     *
     * @param other The other NativePointerObject
     */
    protected NativePointerObject(NativePointerObject other)
    {
        this.nativePointer = other.nativePointer;
    }

    /**
     * Obtain the native pointer value.
     *
     * @return The native pointer value
     */
    public long getNativePointer()
    {
        return nativePointer;
    }

    @Override
    public String toString()
    {
        return "NativePointerObject[nativePointer=" + nativePointer + "]";
    }

    @Override
    public int hashCode()
    {
        final int prime = 31;
        int result = 1;
        result = prime * result + (int)(nativePointer ^ (nativePointer >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj)
    {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        NativePointerObject other = (NativePointerObject)obj;
        if (nativePointer != other.nativePointer)
            return false;
        return true;
    }



}
