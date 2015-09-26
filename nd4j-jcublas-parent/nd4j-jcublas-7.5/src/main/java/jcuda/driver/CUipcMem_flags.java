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
 * Memory flags
 */
public class CUipcMem_flags
{
    /**
     * Automatically enable peer access between remote devices as needed
     */
    public static final int CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1;

    /**
     * Returns the String identifying the given CUipcMem_flags
     *
     * @param n The CUipcMem_flags
     * @return The String identifying the given CUipcMem_flags
     */
    public static String stringFor(int n)
    {
        if (n == 0)
        {
            return "INVALID CUipcMem_flags: "+n;
        }
        String result = "";
        if ((n & CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS) != 0) result += "CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS";
        return result;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUipcMem_flags()
    {
    }

}
