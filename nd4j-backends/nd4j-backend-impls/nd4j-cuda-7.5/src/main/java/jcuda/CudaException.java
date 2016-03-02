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

package jcuda;

/**
 * An exception that may be thrown due to a CUDA error. <br />
 * <br />
 * For the JCuda runtime API, exceptions may be enabled or disabled using
 * {@link jcuda.runtime.JCuda#setExceptionsEnabled(boolean) JCuda#setExceptionsEnabled(boolean)}.
 * If exceptions are enabled, the CUDA binding methods will throw a
 * CudaException if the CUDA function did not return cudaError.cudaSuccess.<br />
 * <br />
 * For the JCuda driver API, exceptions may be enabled or disabled using
 * {@link jcuda.driver.JCudaDriver#setExceptionsEnabled(boolean) JCudaDriver#setExceptionsEnabled(boolean)}.
 * If exceptions are enabled, the CUDA binding methods will throw a
 * CudaException if the CUDA function did not return CUresult.CUDA_SUCCESS.<br />
 */
public class CudaException extends RuntimeException
{
    /**
     * The serial version UID
     */
    private static final long serialVersionUID = 1587809813906124159L;

    /**
     * Creates a new CudaException with the given error message.
     *
     * @param message The error message for this CudaException
     */
    public CudaException(String message)
    {
        super(message);
    }

    /**
     * Creates a new CudaException with the given error message
     * and the given Throwable as the cause.
     *
     * @param message The error message for this CudaException
     * @param cause The reason for this CudaException
     */
    public CudaException(String message, Throwable cause)
    {
        super(message, cause);
    }
}
