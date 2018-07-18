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

package org.nd4j.jita.allocator.utils;

import lombok.NonNull;
import org.bytedeco.javacpp.LongPointer;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;


/**
 * @author raver119@gmail.com
 */
public class AllocationUtils {

    public static long getRequiredMemory(@NonNull AllocationShape shape) {
        return shape.getLength() * getElementSize(shape);
    }

    public static int getElementSize(@NonNull AllocationShape shape) {
        if (shape.getElementSize() > 0)
            return shape.getElementSize();
        else
            return (shape.getDataType() == DataBuffer.Type.FLOAT ? 4
                            : shape.getDataType() == DataBuffer.Type.DOUBLE ? 8 : 2);
    }

    /**
     * This method returns AllocationShape for specific array, that takes in account its real shape: offset, length, etc
     *
     * @param array
     * @return
     */
    public static AllocationShape buildAllocationShape(INDArray array) {
        AllocationShape shape = new AllocationShape();
        shape.setStride(array.elementWiseStride());
        shape.setOffset(array.originalOffset());
        shape.setDataType(array.data().dataType());
        shape.setLength(array.length());
        shape.setDataType(array.data().dataType());

        return shape;
    }

    /**
     * This method returns AllocationShape for the whole DataBuffer.
     *
     * @param buffer
     * @return
     */
    public static AllocationShape buildAllocationShape(DataBuffer buffer) {
        AllocationShape shape = new AllocationShape();
        shape.setStride(1);
        shape.setOffset(buffer.originalOffset());
        shape.setDataType(buffer.dataType());
        shape.setLength(buffer.length());

        return shape;
    }

    /**
     * This method returns AllocationShape for specific buffer, that takes in account its real shape: offset, length, etc
     *
     * @param buffer
     * @return
     */
    public static AllocationShape buildAllocationShape(JCudaBuffer buffer) {
        AllocationShape shape = new AllocationShape();
        shape.setStride(1);
        shape.setOffset(buffer.originalOffset());
        shape.setDataType(buffer.dataType());
        shape.setLength(buffer.length());

        return shape;
    }

    /**
     * This method returns byte offset based on AllocationShape
     *
     * @return
     */
    public static long getByteOffset(AllocationShape shape) {
        return shape.getOffset() * getElementSize(shape);
    }


    public static DataBuffer getPointersBuffer(long[] pointers) {
        CudaDoubleDataBuffer tempX = new CudaDoubleDataBuffer(pointers.length);
        AtomicAllocator.getInstance().memcpyBlocking(tempX, new LongPointer(pointers), pointers.length * 8, 0);
        return tempX;
    }
}
