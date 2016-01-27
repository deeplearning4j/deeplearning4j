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

import com.google.common.collect.Table;
import jcuda.Pointer;
import org.apache.commons.lang3.tuple.Triple;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.context.CudaContext;

import java.nio.Buffer;
import java.nio.ByteBuffer;

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
     * @param arr the array to get the device pointer for
     * @param stride the stride for the device pointer
     * @param offset the offset for the device pointer
     * @return the device pointer with the given
     * offset and stride
     * @param length the length of the pointer
     */
    Pointer getDevicePointer(INDArray arr, int stride, int offset, int length);

    /**
     * Get the device pointer with the given offset and stride
     * @param stride the stride for the device pointer
     * @param offset the offset for the device pointer
     * @return the device pointer with the given
     * offset and stride
     * @param length the length of the pointer
     */
    Pointer getDevicePointer(int stride, int offset, int length);

    /**
     * Sets the data for this pointer
     * from the data in this pointer
     *
     * @param pointer the pointer to set
     */
    void set(Pointer pointer);


    /**
     * Frees the pointer
     * @param offset the offset to free
     * @param length the length to free
     * @return true if the pointer was freed,
     * false other wise
     */
    boolean freeDevicePointer(int offset, int length, int stride);


    /**
     * Copy to the host synchronizing
     * using the given context
     * @param context the context to synchronize
     * @param offset the offset to copy
     * @param length the length to copy
     * @param stride
     */
    void copyToHost(CudaContext context, int offset, int length, int stride);

    /**
     * Copies the buffer to the host
     * @param offset the offset for the buffer (one buffer may have multiple pointers
     * @param length the length of the pointer (one buffer may have different lengths)
     */
    void copyToHost(int offset, int length);

    /**
     * Pointer to context map.
     * Contains the device pointer information
     * mapping thread name to offset
     * @return the pointer info containing allocated pointers
     */
    Table<String, Triple<Integer, Integer, Integer>, DevicePointerInfo> getPointersToContexts();

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
