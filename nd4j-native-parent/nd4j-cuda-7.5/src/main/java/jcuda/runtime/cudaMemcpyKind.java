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
 * Memcpy kinds.
 *
 * @see JCuda#cudaMemcpy
 * @see jcuda.runtime.cudaMemcpy3DParms
 */
public class cudaMemcpyKind
{

    /**
     * Host   -> Host
     */
    public static final int cudaMemcpyHostToHost = 0;

    /**
     * Host   -> Device
     */
    public static final int cudaMemcpyHostToDevice = 1;

    /**
     * Device -> Host
     */
    public static final int cudaMemcpyDeviceToHost = 2;

    /**
     * Device -> Device
     */
    public static final int cudaMemcpyDeviceToDevice = 3;

    /**
     * Default based unified virtual address space
     */
    public static final int cudaMemcpyDefault = 4;

    /**
     * Returns the String identifying the given cudaMemcpyKind
     *
     * @param k The cudaMemcpyKind
     * @return The String identifying the given cudaMemcpyKind
     */
    public static String stringFor(int k)
    {
        switch (k)
        {
            case cudaMemcpyHostToHost: return "cudaMemcpyHostToHost";
            case cudaMemcpyHostToDevice: return "cudaMemcpyHostToDevice";
            case cudaMemcpyDeviceToHost: return "cudaMemcpyDeviceToHost";
            case cudaMemcpyDeviceToDevice: return "cudaMemcpyDeviceToDevice";
            case cudaMemcpyDefault: return "cudaMemcpyDefault";
        }
        return "INVALID cudaMemcpyKind: "+k;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaMemcpyKind()
    {
    }

}
