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
 * Memory types.
 *
 * @see JCudaDriver#cuMemcpyHtoD
 * @see JCudaDriver#cuMemcpyDtoH
 * @see JCudaDriver#cuMemcpyDtoD
 * @see JCudaDriver#cuMemcpyDtoA
 * @see JCudaDriver#cuMemcpyAtoD
 * @see JCudaDriver#cuMemcpyAtoH
 * @see JCudaDriver#cuMemcpyHtoA
 * @see JCudaDriver#cuMemcpyAtoA
 * @see JCudaDriver#cuMemcpy2D
 * @see JCudaDriver#cuMemcpy2DAsync
 * @see JCudaDriver#cuMemcpy3D
 * @see JCudaDriver#cuMemcpy3DAsync
 */
public class CUmemorytype
{
    /**
     * Host memory
     */
    public static final int CU_MEMORYTYPE_HOST = 0x01;

    /**
     * Device memory
     */
    public static final int CU_MEMORYTYPE_DEVICE = 0x02;

    /**
     * Array memory
     */
    public static final int CU_MEMORYTYPE_ARRAY = 0x03;

    /**
     * Unified device or host memory
     */
    public static final int CU_MEMORYTYPE_UNIFIED = 0x04;

    /**
     * Returns the String identifying the given CUmemorytype
     *
     * @param n The CUmemorytype
     * @return The String identifying the given CUmemorytype
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_MEMORYTYPE_HOST: return "CU_MEMORYTYPE_HOST";
            case CU_MEMORYTYPE_DEVICE: return "CU_MEMORYTYPE_DEVICE";
            case CU_MEMORYTYPE_ARRAY: return "CU_MEMORYTYPE_ARRAY";
            case CU_MEMORYTYPE_UNIFIED: return "CU_MEMORYTYPE_UNIFIED";
        }
        return "INVALID CUmemorytype: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUmemorytype()
    {
    }

}
