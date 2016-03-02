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
 * An enumerant to specify the data precision. It is used when the 
 * data reference does not carry the type itself (e.g void *)<br>
 * <br>
 * For example, it is used in the routine {@link JCublas#cublasSgemmEx}. 
 */
public class cublasDataType
{
    /**
     * The data type is 32-bit floating-point
     */
    public static final int CUBLAS_DATA_FLOAT    = 0;
    
    /**
     * The data type is 64-bit floating-point
     */
    public static final int CUBLAS_DATA_DOUBLE   = 1;

    /**
     * The data type is 16-bit floating-point
     */
    public static final int CUBLAS_DATA_HALF     = 2;

    /**
     * The data type is 8-bit signed integer
     */
    public static final int CUBLAS_DATA_INT8     = 3;
    
    /**
     * Private constructor to prevent instantiation
     */
    private cublasDataType(){}

    /**
     * Returns a string representation of the given constant
     *
     * @return A string representation of the given constant
     */
    public static String stringFor(int n)
    {
        switch (n)
        {
            case CUBLAS_DATA_FLOAT: return "CUBLAS_DATA_FLOAT";
            case CUBLAS_DATA_DOUBLE: return "CUBLAS_DATA_DOUBLE";
            case CUBLAS_DATA_HALF: return "CUBLAS_DATA_HALF";
            case CUBLAS_DATA_INT8: return "CUBLAS_DATA_INT8";
        }
        return "INVALID cublasDataType: "+n;
    }
}

