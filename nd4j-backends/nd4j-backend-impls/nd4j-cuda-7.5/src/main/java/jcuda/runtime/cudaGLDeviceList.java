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
