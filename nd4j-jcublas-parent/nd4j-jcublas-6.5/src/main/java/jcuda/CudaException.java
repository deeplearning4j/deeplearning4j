/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
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
