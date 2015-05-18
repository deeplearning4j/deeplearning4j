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

package jcuda.driver;

/**
 * Interface for emulating a CUDA stream callback.
 * 
 * @see JCudaDriver#cuStreamAddCallback(CUstream, CUstreamCallback, Object, int)
 */
public interface CUstreamCallback
{
    /**
     * The function that will be called
     * 
     * @param hStream The stream the callback was added to, as passed to 
     * {@link JCudaDriver#cuStreamAddCallback(CUstream, CUstreamCallback, Object, int)}. 
     * May be NULL.
     * @param status CUDA_SUCCESS or any persistent error on the stream.
     * @param userData User parameter provided at registration.
     */
    void call(CUstream hStream, int status, Object userData);
}

