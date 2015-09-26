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


/**
 * Java port of a cudaIpcMemHandle.
 */
public class cudaIpcMemHandle
{
    private static final int CUDA_IPC_HANDLE_SIZE = 64;
    
    // This field is only used on the native side. Nobody
    // knows whether CUDA already uses this data, but
    // presumably, it does
    @SuppressWarnings("unused")
    private byte reserved[] = new byte[CUDA_IPC_HANDLE_SIZE];
    
    /**
     * Creates a new, uninitialized cudaIpcMemHandle
     */
    public cudaIpcMemHandle()
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
        return super.toString();
    }

}
