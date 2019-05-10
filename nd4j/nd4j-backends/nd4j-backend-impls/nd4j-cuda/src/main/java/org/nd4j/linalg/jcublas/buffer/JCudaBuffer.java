/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.jcublas.buffer;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;

import java.nio.Buffer;

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
    Pointer getHostPointer(long offset);
}
