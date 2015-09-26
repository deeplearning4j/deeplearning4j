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
 * CUDA Profiler Output modes
 */
public class cudaOutputMode
{
    /**
     * Output mode Key-Value pair format.
     */
    public static final int cudaKeyValuePair    = 0x00;
    
    /**
     * Output mode Comma separated values format.
     */
    public static final int cudaCSV             = 0x01;

    /**
     * Returns the String identifying the given cudaOutputMode
     *
     * @param n The cudaOutputMode
     * @return The String identifying the given cudaOutputMode
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaKeyValuePair: return "cudaKeyValuePair";
            case cudaCSV: return "cudaCSV";
        }
        return "INVALID cudaOutputMode: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaOutputMode()
    {
    }

}

