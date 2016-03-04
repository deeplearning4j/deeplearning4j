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
 * Surface format modes
 *
 * @see surfaceReference
 */
public class cudaSurfaceFormatMode
{
    public static final int cudaFormatModeForced = 0;
    public static final int cudaFormatModeAuto = 1;

    /**
     * Returns the String identifying the given cudaSurfaceFormatMode
     *
     * @param m The cudaSurfaceFormatMode
     * @return The String identifying the given cudaSurfaceFormatMode
     */
    public static String stringFor(int m)
    {
        switch (m)
        {
            case cudaFormatModeForced: return "cudaFormatModeForced";
            case cudaFormatModeAuto: return "cudaFormatModeAuto";
        }
        return "INVALID cudaSurfaceFormatMode: " + m;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaSurfaceFormatMode()
    {
    }

};
