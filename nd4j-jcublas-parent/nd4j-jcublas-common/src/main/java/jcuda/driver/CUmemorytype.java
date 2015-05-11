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
