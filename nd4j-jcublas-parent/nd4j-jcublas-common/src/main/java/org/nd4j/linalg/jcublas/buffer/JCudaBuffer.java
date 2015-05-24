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

package org.nd4j.linalg.jcublas.buffer;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Map;

import com.google.common.collect.Table;
import jcuda.Pointer;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * A Jcuda buffer
 *
 * @author Adam Gibson
 */
public interface JCudaBuffer extends DataBuffer {

	/**
	 * Set the underlying host buffer, very fast method of copying on RAM side and not using cublas (SLOW)
     * @param hostBuffer
     */
	void setHostBuffer(ByteBuffer hostBuffer);
	
	/**
	 * Get the underlying host bytebuffer
	 * @return
	 */
	Buffer getHostBuffer();
	
    /**
     * THe pointer for the buffer
     *
     * @return the pointer for this buffer
     */
    Pointer getHostPointer();

    /**
     * Get the host pointer with the given offset
     * note that this will automatically
     * multiply the specified offset
     * by the element size
     * @param offset the offset (NOT MULTIPLIED BY ELEMENT SIZE) to index in to the pointer at
     * @return the pointer at the given byte offset
     */
    Pointer getHostPointer(int offset);



    /**
     * Get the device pointer with the given offset and stride
     * @param stride the stride for the device pointer
     * @param offset the offset for the device pointer
     * @return the device pointer with the given
     * offset and stride
     * @param length the length of the pointer
     */
    Pointer getDevicePointer(int stride, int offset,int length);

    /**
     * Sets the data for this pointer
     * from the data in this pointer
     *
     * @param pointer the pointer to set
     */
    void set(Pointer pointer);


    boolean freeDevicePointer(int offset);

    void copyToHost(int offset);

    /**
     * Pointer to context map.
     * Contains the device pointer information
     * mapping thread name to offset
     * @return the pointer info containing allocated poitners
     */
    Table<String, Integer, BaseCudaDataBuffer.DevicePointerInfo> getPointersToContexts();

    /**
     * Returns true if the buffer has
     * already been copied to the device
     * @return true if the buffer
     * has already been copied to the device
     * false otherwise
     */
    boolean copied(String name);


    /**
     * Returns true if the data for this
     * thread name has already been copied
     * @param name the name of the thread to
     *             check for whether it's been copied or not
     *
     */
    void setCopied(String name);


}
