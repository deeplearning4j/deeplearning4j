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
package jcuda.driver;

/**
 * CUDA devices corresponding to an OpenGL device
 */
public class CUGLDeviceList 
{
    /** 
     * The CUDA devices for all GPUs used by the current OpenGL context 
     */
    public static final int CU_GL_DEVICE_LIST_ALL            = 0x01;

    /**
     * The CUDA devices for the GPUs used by the current OpenGL context 
     * in its currently rendering frame 
     */
    public static final int CU_GL_DEVICE_LIST_CURRENT_FRAME  = 0x02; 
    
    /** 
     * The CUDA devices for the GPUs to be used by the current OpenGL 
     * context in the next frame 
     */
    public static final int CU_GL_DEVICE_LIST_NEXT_FRAME     = 0x03;
    
    /**
     * Returns the String identifying the given CUGLDeviceList
     *
     * @param n The CUGLDeviceList
     * @return The String identifying the given CUGLDeviceList
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_GL_DEVICE_LIST_ALL: return "CU_GL_DEVICE_LIST_ALL";
            case CU_GL_DEVICE_LIST_CURRENT_FRAME: return "CU_GL_DEVICE_LIST_CURRENT_FRAME";
            case CU_GL_DEVICE_LIST_NEXT_FRAME: return "CU_GL_DEVICE_LIST_NEXT_FRAME";
        }
        return "INVALID CUfunction_attribute: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUGLDeviceList()
    {
    }

    
} 