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
 * CUDA device compute modes.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 *
 * @see cudaDeviceProp#computeMode
 */
public class cudaComputeMode
{
    /**
     * Default compute mode (Multiple threads can use {@link JCuda#cudaSetDevice(int)} with this device)
     */
    public static final int cudaComputeModeDefault    =   0;

    /**
     * Compute-exclusive mode (Only one thread will be able to use {@link JCuda#cudaSetDevice(int)} with this device)
     */
    public static final int cudaComputeModeExclusive  =   1;

    /**
     * Compute-prohibited mode (No threads can use {@link JCuda#cudaSetDevice(int)} with this device)
     */
    public static final int cudaComputeModeProhibited =   2;

    /** 
     * Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device) 
     */
    public static final int cudaComputeModeExclusiveProcess = 3;
    
    /**
     * Returns the String identifying the given cudaComputeMode
     *
     * @param n The cudaComputeMode
     * @return The String identifying the given cudaComputeMode
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaComputeModeDefault: return "cudaComputeModeDefault";
            case cudaComputeModeExclusive: return "cudaComputeModeExclusive";
            case cudaComputeModeProhibited: return "cudaComputeModeProhibited";
            case cudaComputeModeExclusiveProcess: return "cudaComputeModeExclusiveProcess";
        }
        return "INVALID cudaComputeMode: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaComputeMode()
    {
    }

};

