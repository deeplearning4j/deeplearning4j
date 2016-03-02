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
 * CUDA device compute modes.<br />
 * <br />
 * Most comments are taken from the CUDA reference manual.<br />
 *
 * @see cudaDeviceProp#computeMode
 */
public class cudaComputeMode
{
    /**
     * Default compute mode (Multiple threads can use {@link JCuda#cudaSetDevice(int)} with this device)
     */
    public static final int cudaComputeModeDefault    =   0;

    /**
     * Compute-exclusive mode (Only one thread will be able to use {@link JCuda#cudaSetDevice(int)} with this device)
     */
    public static final int cudaComputeModeExclusive  =   1;

    /**
     * Compute-prohibited mode (No threads can use {@link JCuda#cudaSetDevice(int)} with this device)
     */
    public static final int cudaComputeModeProhibited =   2;

    /**
     * Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device)
     */
    public static final int cudaComputeModeExclusiveProcess = 3;

    /**
     * Returns the String identifying the given cudaComputeMode
     *
     * @param n The cudaComputeMode
     * @return The String identifying the given cudaComputeMode
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case cudaComputeModeDefault: return "cudaComputeModeDefault";
            case cudaComputeModeExclusive: return "cudaComputeModeExclusive";
            case cudaComputeModeProhibited: return "cudaComputeModeProhibited";
            case cudaComputeModeExclusiveProcess: return "cudaComputeModeExclusiveProcess";
        }
        return "INVALID cudaComputeMode: "+n;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cudaComputeMode()
    {
    }

};

