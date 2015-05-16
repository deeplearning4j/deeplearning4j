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
 * CUDA memory types
 */
public class cudaMemoryType
{
    /**
     *  Host memory 
     */
    public static final int cudaMemoryTypeHost   = 1;

    /**
     * Device memory
     */
    public static final int cudaMemoryTypeDevice = 2;

    /**
     * Returns the String identifying the given cudaMemoryType
     *
     * @param k The cudaMemoryType
     * @return The String identifying the given cudaMemoryType
     */
    public static String stringFor(int k)
    {
        switch (k)
        {
            case cudaMemoryTypeHost: return "cudaMemoryTypeHost";
            case cudaMemoryTypeDevice: return "cudaMemoryTypeDevice";
        }
        return "INVALID cudaMemoryType: "+k;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaMemoryType()
    {
    }

}
