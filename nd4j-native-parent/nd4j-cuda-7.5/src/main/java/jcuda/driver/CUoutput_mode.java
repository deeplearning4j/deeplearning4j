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
 * Profiler Output Modes.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.
 *
 * @see JCudaDriver#cuProfilerInitialize
 */
public class CUoutput_mode
{
    /**
     * Output mode Key-Value pair format.
     */
    public static final int CU_OUT_KEY_VALUE_PAIR  = 0x00;

    /**
     * Output mode Comma separated values format.
     */
    public static final int CU_OUT_CSV             = 0x01;


    /**
     * Returns the String identifying the given CUoutput_mode
     *
     * @param n The CUoutput_mode
     * @return The String identifying the given CUoutput_mode
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CU_OUT_KEY_VALUE_PAIR: return "CU_OUT_KEY_VALUE_PAIR";
            case CU_OUT_CSV: return "CU_OUT_CSV";
        }
        return "INVALID CUoutput_mode: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private CUoutput_mode()
    {
    }

}
