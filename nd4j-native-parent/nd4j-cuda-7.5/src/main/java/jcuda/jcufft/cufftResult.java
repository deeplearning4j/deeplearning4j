/*
 * JCufft - Java bindings for CUFFT, the NVIDIA CUDA FFT library,
 * to be used with JCuda
 *
 * Copyright (c) 2008-2015 Marco Hutter - http://www.jcuda.org
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

package jcuda.jcufft;

/**
 * The result of a CUFFT operation
 */
public class cufftResult
{
    /**
     * Any CUFFT operation is successful.
     */
    public static final int CUFFT_SUCCESS = 0;

    /**
     * CUFFT is passed an invalid plan handle.
     */
    public static final int CUFFT_INVALID_PLAN = 1;

    /**
     * CUFFT failed to allocate GPU memory.
     */
    public static final int CUFFT_ALLOC_FAILED = 2;

    /**
     * The user requests an unsupported type.
     */
    public static final int CUFFT_INVALID_TYPE = 3;

    /**
     * The user specifies a bad memory pointer.
     */
    public static final int CUFFT_INVALID_VALUE = 4;

    /**
     * Used for all internal driver errors.
     */
    public static final int CUFFT_INTERNAL_ERROR = 5;

    /**
     * CUFFT failed to execute an FFT on the GPU.
     */
    public static final int CUFFT_EXEC_FAILED = 6;

    /**
     * The CUFFT library failed to initialize.
     */
    public static final int CUFFT_SETUP_FAILED = 7;

    /**
     * The user specifies an unsupported FFT size.
     */
    public static final int CUFFT_INVALID_SIZE = 8;

    /**
     * Unaligned data
     */
    public static final int CUFFT_UNALIGNED_DATA = 9;

    /**
     * Incomplete parameter list
     */
    public static final int CUFFT_INCOMPLETE_PARAMETER_LIST = 0xA;

    /**
     * Invalid device
     */
    public static final int CUFFT_INVALID_DEVICE = 0xB;

    /**
     * Parse error
     */
    public static final int CUFFT_PARSE_ERROR = 0xC;

    /**
     * No workspace
     */
    public static final int CUFFT_NO_WORKSPACE = 0xD;

    /**
     * Not implemented
     */
    public static final int CUFFT_NOT_IMPLEMENTED = 0xE;

    /**
     * License error
     */
    public static final int CUFFT_LICENSE_ERROR = 0x0F;

    /**
     * An internal JCufft error occurred
     */
    public static final int JCUFFT_INTERNAL_ERROR = 0xFF;

    /**
     * Returns the String identifying the given cufftResult
     *
     * @param m The cufftResult
     * @return The String identifying the given cufftResult
     */
    public static String stringFor(int m)
    {
        switch (m)
        {
            case CUFFT_SUCCESS : return "CUFFT_SUCCESS";
            case CUFFT_INVALID_PLAN : return "CUFFT_INVALID_PLAN";
            case CUFFT_ALLOC_FAILED : return "CUFFT_ALLOC_FAILED";
            case CUFFT_INVALID_TYPE : return "CUFFT_INVALID_TYPE";
            case CUFFT_INVALID_VALUE : return "CUFFT_INVALID_VALUE";
            case CUFFT_INTERNAL_ERROR : return "CUFFT_INTERNAL_ERROR";
            case CUFFT_EXEC_FAILED : return "CUFFT_EXEC_FAILED";
            case CUFFT_SETUP_FAILED : return "CUFFT_SETUP_FAILED";
            case CUFFT_INVALID_SIZE : return "CUFFT_INVALID_SIZE";
            case CUFFT_UNALIGNED_DATA : return "CUFFT_UNALIGNED_DATA";
            case CUFFT_INCOMPLETE_PARAMETER_LIST : return "CUFFT_INCOMPLETE_PARAMETER_LIST";
            case CUFFT_INVALID_DEVICE : return "CUFFT_INVALID_DEVICE";
            case CUFFT_PARSE_ERROR : return "CUFFT_PARSE_ERROR";
            case CUFFT_NO_WORKSPACE : return "CUFFT_NO_WORKSPACE";
            case CUFFT_NOT_IMPLEMENTED : return "CUFFT_NOT_IMPLEMENTED";
            case CUFFT_LICENSE_ERROR : return "CUFFT_LICENSE_ERROR";
            case JCUFFT_INTERNAL_ERROR : return "JCUFFT_INTERNAL_ERROR";
        }
        return "INVALID cufftResult: " + m;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private cufftResult()
    {
    }

}
