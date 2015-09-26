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