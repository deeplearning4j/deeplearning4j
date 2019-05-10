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

package org.nd4j.linalg.api.shape;

import lombok.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
public class LongShapeDescriptor {

    @Getter
    private char order;

    private long offset;

    private long ews;

    private long hashShape = 0;
    private long hashStride = 0;

    @Getter
    private long[] shape;

    @Getter
    private long[] stride;

    @Getter @Setter
    private long extras;

    public LongShapeDescriptor(long[] shape, long[] stride, long offset, long ews, char order, long extras) {
        /*
        if (shape != null) {
            hashShape = shape[0];
            for (int i = 1; i < shape.length; i++)
                hashShape = 31 * hashShape + shape[i];
        }
        
        if (stride != null) {
            hashStride = stride[0];
            for (int i = 1; i < stride.length; i++)
                hashStride = 31 * hashStride + stride[i];
        }
        */
        this.shape = Arrays.copyOf(shape, shape.length);
        this.stride = Arrays.copyOf(stride, stride.length);

        this.offset = offset;
        this.ews = ews;
        this.order = order;

        this.extras = extras;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        LongShapeDescriptor that = (LongShapeDescriptor) o;

        if (extras != that.extras)
            return false;
        if (order != that.order)
            return false;
        if (offset != that.offset)
            return false;
        if (ews != that.ews)
            return false;
        if (!Arrays.equals(shape, that.shape))
            return false;
        return Arrays.equals(stride, that.stride);

    }

    public DataType dataType() {
        return ArrayOptionsHelper.dataType(extras);
    }

    @Override
    public int hashCode() {
        int result = (int) order;

        result = 31 * result + longHashCode(offset);
        result = 31 * result + longHashCode(ews);
        result = 31 * result + longHashCode(extras);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(stride);
        return result;
    }

    @Override
    public String toString() {

        StringBuilder builder = new StringBuilder();

        builder.append(shape.length).append(",").append(Arrays.toString(shape)).append(",")
                        .append(Arrays.toString(stride)).append(",").append(extras).append(",").append(ews).append(",")
                        .append(order);

        String result = builder.toString().replaceAll("\\]", "").replaceAll("\\[", "");
        result = "[" + result + "]";

        return result;
    }

    private int longHashCode(long v) {
        // impl from j8
        return (int)(v ^ (v >>> 32));
    }

    public static LongShapeDescriptor fromShapeDescriptor(@NonNull ShapeDescriptor descriptor) {
        return new LongShapeDescriptor(ArrayUtil.toLongArray(descriptor.getShape()), ArrayUtil.toLongArray(descriptor.getStride()), descriptor.getOffset(), descriptor.getEws(), descriptor.getOrder(), descriptor.getExtras());
    }

    public static LongShapeDescriptor fromShape(int[] shape, @NonNull DataType dataType) {
        return fromShape(ArrayUtil.toLongArray(shape), dataType);
    }

    public static LongShapeDescriptor fromShape(long[] shape, @NonNull DataType dataType) {
        return fromShape(shape, Nd4j.getStrides(shape, Nd4j.order()), 1, Nd4j.order(), dataType, false);
    }

    public static LongShapeDescriptor fromShape(@NonNull long[] shape, @NonNull long[] strides, long ews, char order, @NonNull DataType dataType, boolean empty){
        long extras = 0L;
        extras = ArrayOptionsHelper.setOptionBit(extras, dataType);
        if (empty)
            extras = ArrayOptionsHelper.setOptionBit(extras, ArrayType.EMPTY);

        return new LongShapeDescriptor(shape, strides, 0, ews, order, extras);
    }

    public static LongShapeDescriptor fromShape(long[] shape, long extras){
        return new LongShapeDescriptor(shape, Nd4j.getStrides(shape, Nd4j.order()), 0, 1, Nd4j.order(), extras);
    }

    /**
     * Return a new LongShapeDescriptor with the same shape, strides, order etc but with the specified datatype instead
     * @param dataType Datatype of the returned descriptor
     */
    public LongShapeDescriptor asDataType(DataType dataType){
        long extras = 0L;
        extras = ArrayOptionsHelper.setOptionBit(extras, dataType);
        if(isEmpty()){
            extras = ArrayOptionsHelper.setOptionBit(extras, ArrayType.EMPTY);
        }
        return new LongShapeDescriptor(shape, stride, offset, ews, order, extras);
    }

    public boolean isEmpty(){
        return ArrayOptionsHelper.hasBitSet(extras, ArrayOptionsHelper.ATYPE_EMPTY_BIT);
    }
}
