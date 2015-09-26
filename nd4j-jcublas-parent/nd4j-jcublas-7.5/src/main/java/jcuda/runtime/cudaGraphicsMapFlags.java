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
 * CUDA graphics interop map flags
 */
public class cudaGraphicsMapFlags
{
    /**
     * Default; Assume resource can be read/written
     */
    public static final int cudaGraphicsMapFlagsNone         = 0;

    /**
     * CUDA will not write to this resource
     */
    public static final int cudaGraphicsMapFlagsReadOnly     = 1;

    /**
     * CUDA will only write to and will not read from this resource
     */
    public static final int cudaGraphicsMapFlagsWriteDiscard = 2;


    /**
     * Returns the String identifying the given cudaGraphicsMapFlags
     *
     * @param n The cudaGraphicsMapFlags
     * @return The String identifying the given cudaGraphicsMapFlags
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaGraphicsMapFlagsNone: return "cudaGraphicsMapFlagsNone";
            case cudaGraphicsMapFlagsReadOnly: return "cudaGraphicsMapFlagsReadOnly";
            case cudaGraphicsMapFlagsWriteDiscard: return "cudaGraphicsMapFlagsWriteDiscard";
        }
        return "INVALID cudaGraphicsMapFlags: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaGraphicsMapFlags()
    {
    }

}
