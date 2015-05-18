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
 * CUDA devices corresponding to the current OpenGL context
 */
public class cudaGLDeviceList
{
    /** 
     * The CUDA devices for all GPUs used by the current OpenGL context 
     */
    public static final int cudaGLDeviceListAll           = 1;
    
    /** The CUDA devices for the GPUs used by the current OpenGL context 
     * in its currently rendering frame 
     */
    public static final int cudaGLDeviceListCurrentFrame  = 2;
    
    /** The CUDA devices for the GPUs to be used by the current OpenGL 
     * context in the next frame  
     */    
    public static final int cudaGLDeviceListNextFrame     = 3;  

    /**
     * Returns the String identifying the given cudaGLDeviceList
     *
     * @param n The cudaGLDeviceList
     * @return The String identifying the given cudaGLDeviceList
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaGLDeviceListAll: return "cudaGLDeviceListAll";
            case cudaGLDeviceListCurrentFrame: return "cudaGLDeviceListCurrentFrame";
            case cudaGLDeviceListNextFrame: return "cudaGLDeviceListNextFrame";
        }
        return "INVALID cudaGLDeviceList: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaGLDeviceList()
    {
    }
    
}
