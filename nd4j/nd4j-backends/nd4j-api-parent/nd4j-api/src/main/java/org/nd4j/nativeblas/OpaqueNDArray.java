/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.nativeblas;

import org.bytedeco.javacpp.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;

public class OpaqueNDArray extends Pointer {

    public OpaqueNDArray(Pointer p) { super(p); }
    public static OpaqueNDArray create(
            OpaqueDataBuffer shapeInfo,
            OpaqueDataBuffer buffer,
            OpaqueDataBuffer specialBuffer,
            long offset) {
        return Nd4j.getNativeOps().create(shapeInfo, buffer, specialBuffer, offset);
    }


    public DataType dataType() {
        return ArrayOptionsHelper.dataType(extras());
    }

    public long extras() {
        return Shape.extras(shapeInfo());
    }

    public static long getOpaqueNDArrayOffset(OpaqueNDArray array) {
        return Nd4j.getNativeOps().getOpaqueNDArrayOffset(array);
    }


    public static long[] getOpaqueNDArrayShapeInfo(OpaqueNDArray array) {
        LongPointer ret =  Nd4j.getNativeOps().getOpaqueNDArrayShapeInfo(array);
        long len = Nd4j.getNativeOps().getShapeInfoLength(array);
        ret.capacity(len);
        long[] retArr = new long[(int) len];
        ret.get(retArr);
        return retArr;
    }

    public static Pointer getOpaqueNDArrayBuffer(OpaqueNDArray array) {
        return Nd4j.getNativeOps().getOpaqueNDArrayBuffer(array);
    }

    public static Pointer getOpaqueNDArraySpecialBuffer(OpaqueNDArray array) {
        return Nd4j.getNativeOps().getOpaqueNDArraySpecialBuffer(array);
    }

    public static long getOpaqueNDArrayLength(OpaqueNDArray array) {
        return Nd4j.getNativeOps().getOpaqueNDArrayLength(array);
    }
    public static  void deleteNDArray(OpaqueNDArray array) {
        Nd4j.getNativeOps().deleteNDArray(array);
    }

    public static void delete(OpaqueNDArray array) {
        if (array != null && !array.isNull()) {
            deleteNDArray(array);
            array.setNull();
        }
    }

    @Override
    public void close() {
        delete(this);
    }

    /**
     * Convert an INDArray to an OpaqueNDArray
     * @param array The INDArray to convert
     * @return The corresponding OpaqueNDArray
     */
    public static OpaqueNDArray fromINDArray(INDArray array) {
        if (array == null) {
            return null;
        }

        DataBuffer buffer = array.data();
        DataBuffer shapeInfo = array.shapeInfoDataBuffer();

        return create(
                shapeInfo.opaqueBuffer(),
                array.isEmpty() ? null : buffer.opaqueBuffer(),
                array.isEmpty() ? null : array.data().opaqueBuffer(),
                array.offset()
        );
    }

    /**
     * Convert an OpaqueNDArray to an INDArray
     * @param opaqueArray The OpaqueNDArray to convert
     * @return The corresponding INDArray
     */
    public static INDArray toINDArray(OpaqueNDArray opaqueArray) {
        if (opaqueArray == null || opaqueArray.isNull()) {
            return null;
        }

        long offset = opaqueArray.getOffset();
        long[] shapeInfoPtr = opaqueArray.shapeInfo();
        Pointer bufferPtr = opaqueArray.buffer();
        Pointer specialBufferPtr = opaqueArray.specialBuffer();

        long length = opaqueArray.length();

        // Extract shape information
        long[] shape = Shape.shape(shapeInfoPtr);
        long[] stride = Shape.stride(shapeInfoPtr);
        char order = Shape.order(shapeInfoPtr);
        long ews = Shape.elementWiseStride(shapeInfoPtr);
        long extras = Shape.extras(shapeInfoPtr);

        // Create LongShapeDescriptor
        LongShapeDescriptor descriptor = LongShapeDescriptor.builder()
                .shape(shape)
                .stride(stride)
                .offset(offset)
                .ews(ews)
                .order(order)
                .extras(extras)
                .build();

        // Create DataBuffer from the OpaqueNDArray's buffer
        DataType dataType = ArrayOptionsHelper.dataType(extras);
        DataBuffer buffer = Nd4j.createBuffer(bufferPtr,specialBufferPtr,length,dataType);
        // Create INDArray using the descriptor and buffer
        return Nd4j.create(buffer, descriptor);
    }

    // Convenience methods
    public long getOffset() {
        return getOpaqueNDArrayOffset(this);
    }


    public long[] shapeInfo() {
        return getOpaqueNDArrayShapeInfo(this);
    }



    public Pointer buffer() {
        return getOpaqueNDArrayBuffer(this);
    }

    public Pointer specialBuffer() {
        return getOpaqueNDArraySpecialBuffer(this);
    }

    public long length() {
        return getOpaqueNDArrayLength(this);
    }
}