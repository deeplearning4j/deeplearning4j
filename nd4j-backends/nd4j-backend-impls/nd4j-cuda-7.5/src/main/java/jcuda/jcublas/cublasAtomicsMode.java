/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda
 *
 * Copyright (c) 2010-2015 Marco Hutter - http://www.jcuda.org
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
package jcuda.jcublas;

/**
 * The type indicates whether CUBLAS routines which has an alternate
 * implementation using atomics can be used. The atomics mode can
 * be set and queried using and routines, respectively.
 *
 * @see JCublas2#cublasSetAtomicsMode(cublasHandle, int)
 */
public class cublasAtomicsMode
{
    /**
     * Atomics are not allowed
     */
    public static final int CUBLAS_ATOMICS_NOT_ALLOWED = 0;

    /**
     * Atomics are allowed
     */
    public static final int CUBLAS_ATOMICS_ALLOWED = 1;

    /**
     * Private constructor to prevent instantiation
     */
    private cublasAtomicsMode(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUBLAS_ATOMICS_NOT_ALLOWED: return "CUBLAS_ATOMICS_NOT_ALLOWED";
            case CUBLAS_ATOMICS_ALLOWED: return "CUBLAS_ATOMICS_ALLOWED";
        }
        return "INVALID cublasAtomicsMode: "+n;
    }
}

