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
 * Device code formats
 */
public class CUjitInputType
{
    /**
     * Compiled device-class-specific device code<br />
     * Applicable options: none
     */
    public static final int CU_JIT_INPUT_CUBIN = 0;

    /**
     * PTX source code<br />
     * Applicable options: PTX compiler options
     */
    public static final int CU_JIT_INPUT_PTX = 1;

    /**
     * Bundle of multiple cubins and/or PTX of some device code<br />
     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
     */
    public static final int CU_JIT_INPUT_FATBINARY = 2;

    /**
     * Host object with embedded device code<br />
     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
     */
    public static final int CU_JIT_INPUT_OBJECT = 3;

    /**
     * Archive of host objects with embedded device code<br />
     * Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY
     */
    public static final int CU_JIT_INPUT_LIBRARY = 4;

    /**
     * Returns the String identifying the given CUjitInputType
     *
     * @param n The CUjitInputType
     * @return The String identifying the given CUjitInputType
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_JIT_INPUT_CUBIN: return "CU_JIT_INPUT_CUBIN";
            case CU_JIT_INPUT_PTX: return "CU_JIT_INPUT_PTX";
            case CU_JIT_INPUT_FATBINARY: return "CU_JIT_INPUT_FATBINARY";
            case CU_JIT_INPUT_OBJECT: return "CU_JIT_INPUT_OBJECT";
            case CU_JIT_INPUT_LIBRARY: return "CU_JIT_INPUT_LIBRARY";
        }
        return "INVALID CUjitInputType: "+n;
    }

    /**
     * Private constructor to prevent instantation
     */
    private CUjitInputType()
    {

    }


}