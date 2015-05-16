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
 * Memcpy kinds.
 *
 * @see JCuda#cudaMemcpy
 * @see cudaMemcpy3DParms
 */
public class cudaMemcpyKind
{

    /**
     * Host   -> Host
     */
    public static final int cudaMemcpyHostToHost = 0;

    /**
     * Host   -> Device
     */
    public static final int cudaMemcpyHostToDevice = 1;

    /**
     * Device -> Host
     */
    public static final int cudaMemcpyDeviceToHost = 2;

    /**
     * Device -> Device
     */
    public static final int cudaMemcpyDeviceToDevice = 3;

    /**
     * Default based unified virtual address space 
     */
    public static final int cudaMemcpyDefault = 4;
        
    /**
     * Returns the String identifying the given cudaMemcpyKind
     *
     * @param k The cudaMemcpyKind
     * @return The String identifying the given cudaMemcpyKind
     */
    public static String stringFor(int k)
    {
        switch (k)
        {
            case cudaMemcpyHostToHost: return "cudaMemcpyHostToHost";
            case cudaMemcpyHostToDevice: return "cudaMemcpyHostToDevice";
            case cudaMemcpyDeviceToHost: return "cudaMemcpyDeviceToHost";
            case cudaMemcpyDeviceToDevice: return "cudaMemcpyDeviceToDevice";
            case cudaMemcpyDefault: return "cudaMemcpyDefault";
        }
        return "INVALID cudaMemcpyKind: "+k;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaMemcpyKind()
    {
    }

}
