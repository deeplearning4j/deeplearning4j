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

package org.nd4j.aeron.ipc;

import org.agrona.DirectBuffer;
import org.agrona.concurrent.UnsafeBuffer;
import org.nd4j.linalg.primitives.Pair;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.binary.BinarySerde;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * NDArray Serialization and
 * de serialization class for
 * aeron.
 *
 * This is a low level class
 * specifically meant for speed.
 *
 * @author Adam Gibson
 */
public class AeronNDArraySerde extends BinarySerde {


    /**
     * Get the direct byte buffer from the given direct buffer
     * @param directBuffer
     * @return
     */
    public static ByteBuffer getDirectByteBuffer(DirectBuffer directBuffer) {
        return directBuffer.byteBuffer() == null
                        ? ByteBuffer.allocateDirect(directBuffer.capacity()).put(directBuffer.byteArray())
                        : directBuffer.byteBuffer();
    }

    /**
     * Convert an ndarray to an unsafe buffer
     * for use by aeron
     * @param arr the array to convert
     * @return the unsafebuffer representation of this array
     */
    public static UnsafeBuffer toBuffer(INDArray arr) {
        return new UnsafeBuffer(toByteBuffer(arr));

    }



    /**
     * Create an ndarray
     * from the unsafe buffer.
     * Note that if you are interacting with a buffer that specifies
     * an {@link org.nd4j.aeron.ipc.NDArrayMessage.MessageType}
     * then you must pass in an offset + 4.
     * Adding 4 to the offset will cause the inter
     * @param buffer the buffer to create the array from
     * @return the ndarray derived from this buffer
     */
    public static Pair<INDArray, ByteBuffer> toArrayAndByteBuffer(DirectBuffer buffer, int offset) {
        return toArrayAndByteBuffer(getDirectByteBuffer(buffer), offset);
    }


    /**
     * Create an ndarray
     * from the unsafe buffer
     * @param buffer the buffer to create the array from
     * @return the ndarray derived from this buffer
     */
    public static INDArray toArray(DirectBuffer buffer, int offset) {
        return toArrayAndByteBuffer(buffer, offset).getLeft();
    }

    /**
     * Create an ndarray
     * from the unsafe buffer
     * @param buffer the buffer to create the array from
     * @return the ndarray derived from this buffer
     */
    public static INDArray toArray(DirectBuffer buffer) {
        return toArray(buffer, 0);
    }



}
