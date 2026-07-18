/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.vulkan;

import com.google.flatbuffers.FlatBufferBuilder;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.graph.FlatArray;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.BaseNDArrayProxy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.any.Assign;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;

import java.util.List;

/**
 * Vulkan-owned NDArray identity.
 *
 * <p>This mirrors CUDA's {@code JCublasNDArray} boundary: array construction is
 * owned by the device backend while storage remains represented by the common
 * {@link DataBuffer}/{@code OpaqueDataBuffer} contract. Host-facing operations
 * first commit pending work and synchronize the primary side of the Vulkan
 * dual-buffer.</p>
 */
public class VulkanNDArray extends BaseNDArray {

    public VulkanNDArray() {
    }

    public VulkanNDArray(double[][] data) {
        super(data);
    }

    public VulkanNDArray(double[][] data, char ordering) {
        super(data, ordering);
    }

    public VulkanNDArray(float[][] data) {
        super(data);
    }

    public VulkanNDArray(float[][] data, char ordering) {
        super(data, ordering);
    }

    public VulkanNDArray(int[] shape, DataBuffer buffer) {
        super(shape, buffer);
    }

    public VulkanNDArray(DataBuffer buffer, LongShapeDescriptor descriptor) {
        super(buffer, descriptor);
    }

    public VulkanNDArray(LongShapeDescriptor descriptor) {
        super(descriptor);
    }

    public VulkanNDArray(DataBuffer data) {
        super(data);
    }

    public VulkanNDArray(DataBuffer data, int[] shape) {
        super(data, shape);
    }

    public VulkanNDArray(DataBuffer data, long[] shape) {
        super(data, shape);
    }

    public VulkanNDArray(DataBuffer data, int[] shape, long offset) {
        super(data, shape, offset);
    }

    public VulkanNDArray(DataBuffer data, int[] shape, long offset, char ordering) {
        super(data, shape, offset, ordering);
    }

    public VulkanNDArray(DataBuffer data, int[] shape, int[] stride, long offset, char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    public VulkanNDArray(DataBuffer data, int[] shape, int[] stride) {
        super(data, shape, stride);
    }

    public VulkanNDArray(DataBuffer data, long[] shape, long[] stride, long offset,
                         char ordering, DataType dataType) {
        super(data, shape, stride, offset, ordering, dataType);
    }

    public VulkanNDArray(DataBuffer data, long[] shape, long[] stride, long offset,
                         long ews, char ordering, DataType dataType) {
        super(data, shape, stride, offset, ews, ordering, dataType);
    }

    public VulkanNDArray(DataBuffer data, long[] shape, long[] stride, char ordering,
                         DataType dataType) {
        super(data, shape, stride, ordering, dataType);
    }

    public VulkanNDArray(DataBuffer data, long[] shape, long[] stride, long offset,
                         long ews, char ordering, DataType dataType, boolean isView) {
        super(data, shape, stride, offset, ews, ordering, dataType, isView);
    }

    public VulkanNDArray(int[] shape) {
        super(shape);
    }

    public VulkanNDArray(long[] shape) {
        super(shape);
    }

    public VulkanNDArray(int[] shape, char ordering) {
        super(shape, ordering);
    }

    public VulkanNDArray(int[] shape, long offset, char ordering) {
        super(shape, offset, ordering);
    }

    public VulkanNDArray(long[] shape, long offset, char ordering) {
        super(shape, offset, ordering);
    }

    public VulkanNDArray(int[] shape, int[] stride) {
        super(shape, stride);
    }

    public VulkanNDArray(int[] shape, int[] stride, long offset) {
        super(shape, stride, offset);
    }

    public VulkanNDArray(int[] shape, int[] stride, long offset, char ordering) {
        super(shape, stride, offset, ordering);
    }

    public VulkanNDArray(int[] shape, int[] stride, long offset, char ordering,
                         boolean initialize) {
        super(shape, stride, offset, ordering, initialize);
    }

    public VulkanNDArray(long[] shape, long[] stride, long offset, char ordering,
                         boolean initialize) {
        super(shape, stride, offset, ordering, initialize);
    }

    public VulkanNDArray(int rows, int columns) {
        super(rows, columns);
    }

    public VulkanNDArray(int rows, int columns, char ordering) {
        super(rows, columns, ordering);
    }

    public VulkanNDArray(List<INDArray> slices, int[] shape) {
        super(slices, shape);
    }

    public VulkanNDArray(List<INDArray> slices, long[] shape) {
        super(slices, shape);
    }

    public VulkanNDArray(List<INDArray> slices, int[] shape, char ordering) {
        super(slices, shape, ordering);
    }

    public VulkanNDArray(List<INDArray> slices, long[] shape, char ordering) {
        super(slices, shape, ordering);
    }

    public VulkanNDArray(List<INDArray> slices, int[] shape, int[] stride) {
        super(slices, shape, stride);
    }

    public VulkanNDArray(List<INDArray> slices, int[] shape, int[] stride,
                         char ordering) {
        super(slices, shape, stride, ordering);
    }

    public VulkanNDArray(float[] data) {
        super(data);
    }

    public VulkanNDArray(float[] data, char ordering) {
        super(data, ordering);
    }

    public VulkanNDArray(float[] data, int[] shape) {
        super(data, shape);
    }

    public VulkanNDArray(float[] data, int[] shape, char ordering) {
        super(data, shape, ordering);
    }

    public VulkanNDArray(float[] data, long[] shape, char ordering) {
        super(data, shape, ordering);
    }

    public VulkanNDArray(float[] data, int[] shape, long offset) {
        super(data, shape, offset);
    }

    public VulkanNDArray(float[] data, int[] shape, long offset, char ordering) {
        super(data, shape, offset, ordering);
    }

    public VulkanNDArray(float[] data, int[] shape, int[] stride) {
        super(data, shape, stride);
    }

    public VulkanNDArray(float[] data, int[] shape, int[] stride, char ordering) {
        super(data, shape, stride, ordering);
    }

    public VulkanNDArray(float[] data, int[] shape, int[] stride, long offset) {
        super(data, shape, stride, offset);
    }

    public VulkanNDArray(float[] data, int[] shape, int[] stride, long offset,
                         char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    public VulkanNDArray(float[] data, long[] shape, long[] stride, long offset,
                         char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    public VulkanNDArray(double[] data, int[] shape, char ordering) {
        super(data, shape, ordering);
    }

    public VulkanNDArray(double[] data, long[] shape, char ordering) {
        super(data, shape, ordering);
    }

    public VulkanNDArray(double[] data, int[] shape, int[] stride, long offset) {
        super(data, shape, stride, offset);
    }

    public VulkanNDArray(double[] data, int[] shape, int[] stride, long offset,
                         char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    public VulkanNDArray(double[] data, long[] shape, long[] stride, long offset,
                         char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    public VulkanNDArray(int[] data, int[] shape, int[] strides) {
        super(data, shape, strides);
    }

    public VulkanNDArray(DataType dataType, long[] shape, long[] paddings,
                         long[] paddingOffsets, char ordering, MemoryWorkspace workspace) {
        super(dataType, shape, paddings, paddingOffsets, ordering, workspace);
    }

    public VulkanNDArray(DataType dataType, long[] shape, long[] strides, int offset,
                         char ordering, MemoryWorkspace workspace) {
        super(dataType, shape, strides, offset, ordering, false, workspace);
    }

    static VulkanNDArray wrapReplica(
            VulkanDataBuffer dataBuffer, INDArray source, DataBuffer shapeInformation) {
        return wrapOwned(dataBuffer, shapeInformation, source.offset());
    }

    private static VulkanNDArray wrapOwned(
            DataBuffer dataBuffer, DataBuffer shapeInformation, long offset) {
        VulkanNDArray result = new VulkanNDArray();
        result.data = dataBuffer;
        result.offset = offset;
        result.setShapeInfoDataBuffer(shapeInformation);
        return result;
    }

    private void synchronizeHostView() {
        VulkanRuntime.getInstance().executioner().commit();
        if (data != null && data.opaqueBuffer() != null) {
            data.opaqueBuffer().syncToPrimary();
        }
    }

    @Override
    public String getString(long index) {
        Preconditions.checkState(isS(), "getString is available only for string arrays");
        Preconditions.checkState(data instanceof VulkanDataBuffer,
                "Vulkan string arrays require VulkanDataBuffer storage");
        return ((VulkanDataBuffer) data).getString(index);
    }

    @Override
    protected int stringBuffer(FlatBufferBuilder builder, DataBuffer buffer) {
        Preconditions.checkArgument(buffer.dataType() == DataType.UTF8,
                "This method can be called on UTF8 buffers only");
        VulkanRuntime.getInstance().executioner().commit();
        buffer.opaqueBuffer().syncToPrimary();
        byte[] bytes = new byte[Math.toIntExact(buffer.length())];
        ((BytePointer) buffer.pointer()).get(bytes);
        return FlatArray.createBufferVector(builder, bytes);
    }

    @Override
    public LongShapeDescriptor shapeDescriptor() {
        return LongShapeDescriptor.fromShape(shape(), stride(), -1, ordering(), dataType(), isEmpty());
    }

    @Override
    public INDArray unsafeDuplication() {
        return unsafeDuplication(true);
    }

    @Override
    public INDArray unsafeDuplication(boolean blocking) {
        VulkanRuntime runtime = VulkanRuntime.getInstance();
        INDArray copy = runtime.affinityManager().replicateToDevice(
                runtime.affinityManager().getDeviceForCurrentThread(), this);
        if (blocking) {
            runtime.executioner().commit();
        }
        return copy;
    }

    @Override
    public INDArray dup() {
        return dup(ordering());
    }

    @Override
    public INDArray dup(char order) {
        Shape.assertValidOrder(order);
        VulkanRuntime runtime = VulkanRuntime.getInstance();
        VulkanAffinityManager affinity = runtime.affinityManager();
        int destinationDevice = affinity.getDeviceForCurrentThread();

        INDArray copySource = this;
        boolean temporarySource = false;
        if (affinity.getDeviceForArray(this) != destinationDevice) {
            copySource = affinity.replicateToDevice(destinationDevice, this);
            temporarySource = true;
        }

        if (!copySource.isView() && copySource.ordering() == order) {
            return temporarySource
                    ? copySource
                    : affinity.replicateToDevice(destinationDevice, copySource);
        }

        if (copySource.isS()) {
            if (copySource.ordering() == order) {
                return temporarySource
                        ? copySource
                        : affinity.replicateToDevice(destinationDevice, copySource);
            }
            if (temporarySource) {
                copySource.close();
            }
            throw new UnsupportedOperationException(
                    "Changing the ordering of a Vulkan string array requires a string device emitter");
        }

        MemoryWorkspace workspace = runtime.memoryManager().getCurrentWorkspace();
        DataBuffer copyBuffer = runtime.dataBufferFactory().create(
                dataType(), length(), false, workspace);
        long[] copyShape = shape().clone();
        long[] copyStrides = order == 'f'
                ? ArrayUtil.calcStridesFortran(copyShape)
                : ArrayUtil.calcStrides(copyShape);
        long elementWiseStride = Shape.elementWiseStride(copyShape, copyStrides, order == 'f');
        DataBuffer copyShapeInfo = runtime.executioner().createShapeInfo(
                copyShape, copyStrides, elementWiseStride, order, dataType(), isEmpty(), false);
        VulkanNDArray result = wrapOwned(copyBuffer, copyShapeInfo, 0L);

        try {
            runtime.executioner().exec(new Assign(copySource, result));
            if (temporarySource) {
                runtime.executioner().commit();
            }
            return result;
        } catch (RuntimeException | Error failure) {
            result.close();
            throw failure;
        } finally {
            if (temporarySource) {
                copySource.close();
            }
        }
    }

    @Override
    public boolean equals(Object other) {
        synchronizeHostView();
        return super.equals(other);
    }

    @Override
    public String toString() {
        synchronizeHostView();
        return super.toString();
    }

    private Object writeReplace() throws java.io.ObjectStreamException {
        synchronizeHostView();
        return new BaseNDArrayProxy(this);
    }
}
