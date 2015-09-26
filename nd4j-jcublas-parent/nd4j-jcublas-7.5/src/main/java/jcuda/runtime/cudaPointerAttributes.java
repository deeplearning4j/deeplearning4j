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

package jcuda.runtime;

import jcuda.Pointer;

/**
 * CUDA pointer attributes
 */
public class cudaPointerAttributes
{
    /**
     * The physical location of the memory, ::cudaMemoryTypeHost or
     * ::cudaMemoryTypeDevice.
     */
    public int memoryType;

    /**
     * The device against which the memory was allocated or registered.
     * If the memory type is ::cudaMemoryTypeDevice then this identifies
     * the device on which the memory referred physically resides.  If
     * the memory type is ::cudaMemoryTypeHost then this identifies the
     * device which was current when the memory was allocated or registered
     * (and if that device is deinitialized then this allocation will vanish
     * with that device's state).
     */
    public int device;

    /**
     * The address which may be dereferenced on the current device to access
     * the memory or NULL if no such address exists.
     */
    public Pointer devicePointer = new Pointer();

    /**
     * The address which may be dereferenced on the host to access the
     * memory or NULL if no such address exists.
     */
    public Pointer hostPointer = new Pointer();

    /**
     * Indicates if this pointer points to managed memory
     */
    public int isManaged;

    /**
     * Creates a new, uninitialized cudaPointerAttributes
     */
    public cudaPointerAttributes()
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
        return "cudaPointerAttributes["+
            "memoryType="+cudaMemoryType.stringFor(memoryType)+","+
            "device="+device+","+
            "devicePointer="+devicePointer+","+
            "hostPointer="+hostPointer+"," +
            "isManaged="+isManaged+"]";
    }

};