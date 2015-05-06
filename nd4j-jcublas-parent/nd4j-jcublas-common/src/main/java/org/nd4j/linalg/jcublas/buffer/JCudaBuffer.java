/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.jcublas.buffer;

import java.nio.ByteBuffer;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * A Jcuda buffer
 *
 * @author Adam Gibson
 */
public interface JCudaBuffer extends DataBuffer {

	/**
	 * Set the underlying host buffer, very fast method of copying on RAM side and not using cublas (SLOW)
	 */
	void setHostBuffer(ByteBuffer hostBuffer);
	
	/**
	 * Get the underlying host bytebuffer
	 * @return
	 */
	ByteBuffer getHostBuffer();
	
    /**
     * THe pointer for the buffer
     *
     * @return the pointer for this buffer
     */
    Pointer getHostPointer();
    
    /**
     * Get a device pointer for this buffer
     * @return
     */
    Pointer getDevicePointer();

    /**
     * Frees the device pointer if it exists
     * @return
     */
    boolean freeDevicePointer();
    
    /**
     * copies all the allocated device memory to the host memory
     */
    void copyToHost();
    
    /**
     * Sets the data for this pointer
     * from the data in this pointer
     *
     * @param pointer the pointer to set
     */
    void set(Pointer pointer);





}
