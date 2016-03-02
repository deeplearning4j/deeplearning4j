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