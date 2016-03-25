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
	 * Get the underlying host bytebuffer
	 * @return
	 */
    @Deprecated
	Buffer getHostBuffer();
	
    /**
     * THe pointer for the buffer
     *
     * @return the pointer for this buffer
     */
    @Deprecated
    Pointer getHostPointer();

    /**
     * Get the host pointer with the given offset
     * note that this will automatically
     * multiply the specified offset
     * by the element size
     * @param offset the offset (NOT MULTIPLIED BY ELEMENT SIZE) to index in to the pointer at
     * @return the pointer at the given byte offset
     */
    @Deprecated
    Pointer getHostPointer(int offset);
}
