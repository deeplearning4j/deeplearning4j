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

import org.bytedeco.javacpp.Pointer;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.DataTypeEx;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.Flatten;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.vulkan.ops.executioner.VulkanExecutioner;
import org.nd4j.nativeblas.BaseNativeNDArrayFactory;
import org.nd4j.nativeblas.OpaqueNDArray;
import org.nd4j.nativeblas.OpaqueNDArrayArr;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Vulkan array factory.
 *
 * <p>The factory follows the CUDA backend boundary: it extends the shared native
 * factory and constructs a backend-owned NDArray type. It does not inherit the
 * CPU factory and it never substitutes CPU BLAS implementations for missing
 * Vulkan functionality.</p>
 */
public class VulkanNDArrayFactory extends BaseNativeNDArrayFactory {
    private final VulkanRuntime runtime;
    private final VulkanDataBufferFactory dataBufferFactory;
    private final VulkanExecutioner executioner;

    public VulkanNDArrayFactory() {
        this.runtime = VulkanRuntime.getInstance();
        this.dataBufferFactory = runtime.dataBufferFactory();
        this.executioner = runtime.executioner();
    }

    public VulkanNDArrayFactory(DataType dtype, Character order) {
        super(dtype, order);
        this.runtime = VulkanRuntime.getInstance();
        this.dataBufferFactory = runtime.dataBufferFactory();
        this.executioner = runtime.executioner();
    }

    public VulkanNDArrayFactory(DataType dtype, char order) {
        super(dtype, order);
        this.runtime = VulkanRuntime.getInstance();
        this.dataBufferFactory = runtime.dataBufferFactory();
        this.executioner = runtime.executioner();
    }

    private DataBuffer createTypedBuffer(double[] data, DataType dataType, MemoryWorkspace workspace) {
        VulkanDataBuffer buffer = (VulkanDataBuffer) dataBufferFactory.create(
                dataType, data.length, false, workspace);
        buffer.put(data);
        return buffer;
    }

    private DataBuffer createTypedBuffer(float[] data, DataType dataType, MemoryWorkspace workspace) {
        VulkanDataBuffer buffer = (VulkanDataBuffer) dataBufferFactory.create(
                dataType, data.length, false, workspace);
        buffer.put(data);
        return buffer;
    }

    private DataBuffer createTypedBuffer(long[] data, DataType dataType, MemoryWorkspace workspace) {
        VulkanDataBuffer buffer = (VulkanDataBuffer) dataBufferFactory.create(
                dataType, data.length, false, workspace);
        buffer.put(data);
        return buffer;
    }

    private DataBuffer createTypedBuffer(int[] data, DataType dataType, MemoryWorkspace workspace) {
        VulkanDataBuffer buffer = (VulkanDataBuffer) dataBufferFactory.create(
                dataType, data.length, false, workspace);
        buffer.put(data);
        return buffer;
    }

    private DataBuffer createTypedBuffer(short[] data, DataType dataType, MemoryWorkspace workspace) {
        VulkanDataBuffer buffer = (VulkanDataBuffer) dataBufferFactory.create(
                dataType, data.length, false, workspace);
        buffer.put(data);
        return buffer;
    }

    private DataBuffer createTypedBuffer(byte[] data, DataType dataType, MemoryWorkspace workspace) {
        VulkanDataBuffer buffer = (VulkanDataBuffer) dataBufferFactory.create(
                dataType, data.length, false, workspace);
        buffer.put(data);
        return buffer;
    }

    private DataBuffer createTypedBuffer(boolean[] data, DataType dataType, MemoryWorkspace workspace) {
        VulkanDataBuffer buffer = (VulkanDataBuffer) dataBufferFactory.create(
                dataType, data.length, false, workspace);
        buffer.put(data);
        return buffer;
    }

    private static int[] strides(int[] shape, char ordering) {
        return ordering == FORTRAN
                ? ArrayUtil.calcStridesFortran(shape)
                : ArrayUtil.calcStrides(shape);
    }

    private int[] strides(int[] shape) {
        return strides(shape, order());
    }

    private static long[] strides(long[] shape, char ordering) {
        for (long dimension : shape) {
            if (dimension == 0L) {
                return new long[shape.length];
            }
        }
        return ordering == FORTRAN
                ? ArrayUtil.calcStridesFortran(shape)
                : ArrayUtil.calcStrides(shape);
    }

    private long[] strides(long[] shape) {
        return strides(shape, order());
    }

    @Override
    public void createBlas() {
        // The legacy factory contract requires a Blas metadata object. Actual
        // level operations are dispatched through Vulkan custom-op emitters.
        blas = new VulkanBlas();
    }

    @Override
    public void createLevel1() {
        level1 = new VulkanLevel1();
    }

    @Override
    public void createLevel2() {
        level2 = new VulkanLevel2();
    }

    @Override
    public void createLevel3() {
        level3 = new VulkanLevel3();
    }

    @Override
    public void createLapack() {
        lapack = new VulkanLapack();
    }

    @Override
    public INDArray createFromDescriptor(DataBuffer shapeInformation) {
        VulkanNDArray array = new VulkanNDArray();
        array.setShapeInfoDataBuffer(shapeInformation);
        long[] shapeInfo = array.shapeInfoJava();
        DataType dataType = Shape.dataType(shapeInfo);
        long length = Shape.isEmpty(shapeInfo) ? 0 : Shape.length(shapeInfo);
        array.setData(dataBufferFactory.create(dataType, length, false));
        return array;
    }

    @Override
    public INDArray create(int[] shape, DataBuffer buffer) {
        return new VulkanNDArray(shape, buffer);
    }

    @Override
    public INDArray create(DataBuffer buffer, LongShapeDescriptor descriptor) {
        return new VulkanNDArray(buffer, descriptor);
    }

    @Override
    public INDArray create(double[][] data) {
        return new VulkanNDArray(data);
    }

    @Override
    public INDArray create(double[][] data, char ordering) {
        return new VulkanNDArray(data, ordering);
    }

    @Override
    public INDArray create(float[][] data) {
        return new VulkanNDArray(data);
    }

    @Override
    public INDArray create(float[][] data, char ordering) {
        return new VulkanNDArray(data, ordering);
    }

    @Override
    public INDArray create(DataBuffer data) {
        return new VulkanNDArray(data);
    }

    @Override
    public INDArray create(DataBuffer data, long rows, long columns, int[] stride, long offset) {
        return new VulkanNDArray(data, new long[]{rows, columns}, ArrayUtil.toLongArray(stride),
                offset, order(), data.dataType());
    }

    @Override
    public INDArray create(int[] shape, char ordering) {
        return new VulkanNDArray(shape, ordering);
    }

    @Override
    public INDArray createUninitialized(int[] shape, char ordering) {
        return new VulkanNDArray(shape, strides(shape, ordering), 0, ordering, false);
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape, int[] stride, long offset, char ordering) {
        return new VulkanNDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(float[] data, int[] shape, long offset, Character ordering) {
        return new VulkanNDArray(data, shape, offset, ordering);
    }

    @Override
    public INDArray create(float[] data, long rows, long columns, int[] stride, long offset, char ordering) {
        return new VulkanNDArray(data, new long[]{rows, columns}, ArrayUtil.toLongArray(stride),
                offset, ordering);
    }

    @Override
    public INDArray create(double[] data, int[] shape, char ordering) {
        return new VulkanNDArray(data, shape, ordering);
    }

    @Override
    public INDArray create(double[] data, long[] shape, char ordering) {
        return new VulkanNDArray(data, shape, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long offset, Character ordering) {
        return new VulkanNDArray(data, shape, strides(shape, ordering), offset, ordering);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long offset, Character ordering) {
        return new VulkanNDArray(data, shape, strides(shape, ordering), offset, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, char ordering) {
        return new VulkanNDArray(data, shape, ordering);
    }

    @Override
    public INDArray create(LongShapeDescriptor descriptor) {
        return new VulkanNDArray(descriptor);
    }

    @Override
    public INDArray create(Collection<String> strings, long[] shape, char order) {
        DataBuffer buffer = dataBufferFactory.createBuffer(strings.toArray(new String[0]));
        return new VulkanNDArray(
                buffer, shape, strides(shape, order), order, DataType.UTF8);
    }

    @Override
    public INDArray createUninitialized(DataType dataType, long[] shape, long[] strides,
                                        char ordering, MemoryWorkspace workspace) {
        return new VulkanNDArray(dataType, shape, strides, 0, ordering, workspace);
    }

    @Override
    public INDArray create(DataBuffer dataBuffer, DataBuffer descriptor) {
        VulkanNDArray array = new VulkanNDArray();
        array.setShapeInfoDataBuffer(descriptor);
        array.setData(dataBuffer);
        return array;
    }

    @Override
    public INDArray create(List<INDArray> arrays, int[] shape, char ordering) {
        return new VulkanNDArray(arrays, shape, ordering);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, long offset) {
        return new VulkanNDArray(data, shape, stride, offset);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, long offset, char ordering) {
        return new VulkanNDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, long offset) {
        return new VulkanNDArray(data, shape, stride, offset);
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, long offset, char ordering) {
        return new VulkanNDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape) {
        return new VulkanNDArray(data, shape);
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape, int[] stride, long offset) {
        return new VulkanNDArray(data, ArrayUtil.toLongArray(shape), ArrayUtil.toLongArray(stride),
                offset, order(), data.dataType());
    }

    @Override
    public INDArray create(List<INDArray> arrays, int[] shape) {
        return new VulkanNDArray(arrays, shape);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, char ordering, long offset) {
        return new VulkanNDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, long offset, char ordering) {
        return new VulkanNDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, long offset, char ordering) {
        return new VulkanNDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, char ordering,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, workspace);
        return new VulkanNDArray(buffer, shape, stride, ordering, dataType);
    }

    @Override
    public INDArray create(DataBuffer buffer, int[] shape, long offset) {
        return new VulkanNDArray(buffer, shape, offset);
    }

    @Override
    public INDArray create(float[] data, int[] shape, long offset) {
        return new VulkanNDArray(data, shape, offset);
    }

    @Override
    public INDArray create(double[] data, int[] shape, long offset) {
        return new VulkanNDArray(data, shape, strides(shape), offset);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, long offset) {
        return new VulkanNDArray(data, shape, stride, offset, order());
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride,
                           char ordering, DataType dataType) {
        DataBuffer buffer = createTypedBuffer(data, dataType, null);
        return new VulkanNDArray(buffer, shape, stride, ordering, dataType);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, long offset) {
        return new VulkanNDArray(data, shape, stride, offset, order());
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, workspace);
        return new VulkanNDArray(buffer, shape, stride, order(), dataType);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, workspace);
        return new VulkanNDArray(buffer, shape, stride, order(), dataType);
    }

    @Override
    public INDArray create(long[] data, long[] shape, long[] stride,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, null);
        return new VulkanNDArray(buffer, shape, stride, order(), dataType);
    }

    @Override
    public INDArray create(int[] data, long[] shape, long[] stride,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, null);
        return new VulkanNDArray(buffer, shape, stride, order(), dataType);
    }

    @Override
    public INDArray create(short[] data, long[] shape, long[] stride,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, null);
        return new VulkanNDArray(buffer, shape, stride, order(), dataType);
    }

    @Override
    public INDArray create(byte[] data, long[] shape, long[] stride,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, null);
        return new VulkanNDArray(buffer, shape, stride, order(), dataType);
    }

    @Override
    public INDArray create(boolean[] data, long[] shape, long[] stride,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, null);
        return new VulkanNDArray(buffer, shape, stride, order(), dataType);
    }

    @Override
    public INDArray create(long[] data, long[] shape, long[] stride, char ordering,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, null);
        return new VulkanNDArray(buffer, shape, stride, ordering, dataType);
    }

    @Override
    public INDArray create(int[] data, long[] shape, long[] stride, char ordering,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, null);
        return new VulkanNDArray(buffer, shape, stride, ordering, dataType);
    }

    @Override
    public INDArray create(short[] data, long[] shape, long[] stride, char ordering,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, null);
        return new VulkanNDArray(buffer, shape, stride, ordering, dataType);
    }

    @Override
    public INDArray create(byte[] data, long[] shape, long[] stride, char ordering,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, null);
        return new VulkanNDArray(buffer, shape, stride, ordering, dataType);
    }

    @Override
    public INDArray create(boolean[] data, long[] shape, long[] stride, char ordering,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, null);
        return new VulkanNDArray(buffer, shape, stride, ordering, dataType);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, char ordering,
                           DataType dataType, MemoryWorkspace workspace) {
        DataBuffer buffer = createTypedBuffer(data, dataType, workspace);
        return new VulkanNDArray(buffer, shape, stride, ordering, dataType);
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape) {
        return new VulkanNDArray(data, shape);
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride, long offset) {
        return new VulkanNDArray(data, shape, stride, offset, order(), data.dataType());
    }

    @Override
    public INDArray create(List<INDArray> arrays, long[] shape) {
        return new VulkanNDArray(arrays, shape);
    }

    @Override
    public INDArray create(long rows, long columns, long[] stride, long offset) {
        return new VulkanNDArray(new long[]{rows, columns}, stride, offset, order(), true);
    }

    @Override
    public INDArray create(long[] shape, char ordering) {
        return new VulkanNDArray(shape, 0, ordering);
    }

    @Override
    public INDArray create(DataType dataType, long[] shape,
                           char ordering, MemoryWorkspace workspace) {
        return create(dataType, shape, strides(shape, ordering), ordering, workspace);
    }

    @Override
    public INDArray create(DataType dataType, long[] shape, long[] strides,
                           char ordering, MemoryWorkspace workspace) {
        DataBuffer buffer = dataBufferFactory.create(
                dataType, Shape.lengthOf(shape), true, workspace);
        return new VulkanNDArray(buffer, shape, strides, ordering, dataType);
    }

    @Override
    public INDArray createUninitialized(long[] shape, char ordering) {
        return new VulkanNDArray(shape, strides(shape, ordering), 0, ordering, false);
    }

    @Override
    public INDArray createUninitialized(DataType dataType, long[] shape,
                                        char ordering, MemoryWorkspace workspace) {
        DataBuffer buffer = dataBufferFactory.create(
                dataType, Shape.lengthOf(shape), false, workspace);
        return new VulkanNDArray(buffer, shape, strides(shape, ordering), ordering, dataType);
    }

    @Override
    public INDArray createUninitializedDetached(DataType dataType, char ordering, long... shape) {
        DataBuffer buffer = dataBufferFactory.create(
                dataType, Shape.lengthOf(shape), false, null);
        return new VulkanNDArray(buffer, shape, strides(shape, ordering), ordering, dataType);
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride,
                           long offset, char ordering) {
        return new VulkanNDArray(data, shape, stride, offset, ordering, data.dataType());
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride,
                           long offset, long ews, char ordering) {
        return new VulkanNDArray(data, shape, stride, offset, ews, ordering, data.dataType());
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride,
                           long offset, long ews, char ordering, boolean isView) {
        return new VulkanNDArray(data, shape, stride, offset, ews, ordering, data.dataType(), isView);
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride,
                           long offset, char ordering, DataType dataType) {
        return new VulkanNDArray(data, shape, stride, offset, ordering, dataType);
    }

    @Override
    public INDArray create(List<INDArray> arrays, long[] shape, char ordering) {
        return new VulkanNDArray(arrays, shape, ordering);
    }

    @Override
    public INDArray create(DataType dataType, long[] shape, long[] paddings,
                           long[] paddingOffsets, char ordering, MemoryWorkspace workspace) {
        return new VulkanNDArray(dataType, shape, paddings, paddingOffsets, ordering, workspace);
    }

    @Override
    public INDArray empty(DataType dataType) {
        long extras = org.nd4j.linalg.api.shape.options.ArrayOptionsHelper.setOptionBit(
                0L, org.nd4j.linalg.api.shape.options.ArrayType.EMPTY);
        extras = org.nd4j.linalg.api.shape.options.ArrayOptionsHelper.setOptionBit(extras, dataType);
        DataBuffer shapeInfo = executioner.createShapeInfo(
                new long[0], new long[0], 1, 'c', dataType, extras);
        return createFromDescriptor(shapeInfo);
    }

    private static DataType rawType(DataTypeEx type) {
        switch (type) {
            case FLOAT8: return DataType.FLOAT8;
            case INT8: return DataType.BYTE;
            case UINT8: return DataType.UBYTE;
            case FLOAT16: return DataType.HALF;
            case INT16: return DataType.SHORT;
            case UINT16: return DataType.UINT16;
            case FLOAT: return DataType.FLOAT;
            case DOUBLE: return DataType.DOUBLE;
            default:
                throw new UnsupportedOperationException(
                        "Vulkan compressed buffers require a compression provider: " + type);
        }
    }

    @Override
    public INDArray convertDataEx(DataTypeEx typeSrc, INDArray source, DataTypeEx typeDst) {
        if (source.isView()) {
            throw new IllegalArgumentException(
                    "Compressed conversion requires an owning Vulkan array; call dup() first");
        }
        DataBuffer converted = convertDataEx(typeSrc, source.data(), typeDst);
        source.setData(converted);
        source.markAsCompressed(false);
        return source;
    }

    @Override
    public DataBuffer convertDataEx(DataTypeEx typeSrc, DataBuffer source, DataTypeEx typeDst) {
        DataBuffer target = dataBufferFactory.create(rawType(typeDst), source.length(), false);
        convertDataEx(typeSrc, source, typeDst, target);
        return target;
    }

    @Override
    public void convertDataEx(DataTypeEx typeSrc, DataBuffer source,
                              DataTypeEx typeDst, DataBuffer target) {
        convertDataEx(typeSrc, source.addressPointer(), typeDst,
                target.addressPointer(), target.length());
    }

    @Override
    public void convertDataEx(DataTypeEx typeSrc, Pointer source,
                              DataTypeEx typeDst, Pointer target, long length) {
        invokeNative("Vulkan type conversion", () ->
                runtime.nativeOps().convertTypes(null, typeSrc.ordinal(), source,
                        length, typeDst.ordinal(), target));
    }

    @Override
    public void convertDataEx(DataTypeEx typeSrc, Pointer source,
                              DataTypeEx typeDst, DataBuffer buffer) {
        convertDataEx(typeSrc, source, typeDst, buffer.addressPointer(), buffer.length());
    }

    @Override
    public INDArray toFlattened(char ordering, Collection<INDArray> arrays) {
        if (arrays == null || arrays.isEmpty()) {
            throw new IllegalArgumentException("At least one array is required for flatten");
        }
        return executioner.exec(
                new Flatten(ordering, arrays.toArray(new INDArray[0])))[0];
    }

    @Override
    public INDArray concat(int dimension, INDArray... arrays) {
        if (arrays == null || arrays.length == 0) {
            throw new IllegalArgumentException("At least one array is required for concat");
        }
        if (arrays.length == 1) {
            return arrays[0];
        }
        return executioner.exec(new Concat(dimension, arrays))[0];
    }

    @Override
    public INDArray specialConcat(int dimension, INDArray... arrays) {
        return concat(dimension, arrays);
    }

    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, int[] indexes) {
        return pullRows(source, sourceDimension, ArrayUtil.toLongArray(indexes), order());
    }

    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, long[] indexes) {
        return pullRows(source, sourceDimension, indexes, order());
    }

    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, int[] indexes, char ordering) {
        return pullRows(source, sourceDimension, ArrayUtil.toLongArray(indexes), ordering);
    }

    private INDArray pullRows(
            INDArray source, int sourceDimension, long[] indexes, char ordering) {
        int axis = normalizeAxis(source, sourceDimension);
        long[] outputShape = pullRowsOutputShape(source, axis, indexes);
        validatePullRowsIndexes(source, axis, indexes);

        int deviceId = runtime.affinityManager().getDeviceForArray(source);
        int previousDevice = runtime.affinityManager().getDeviceForCurrentThread();
        INDArray destination = null;
        try {
            if (previousDevice != deviceId) {
                runtime.affinityManager().setDeviceForCurrentThread(deviceId);
            }
            destination = createUninitialized(
                    source.dataType(), outputShape, ordering, null);
            executePullRows(source, destination, axis, indexes);
            return destination;
        } catch (RuntimeException | Error failure) {
            if (destination != null) {
                destination.close();
            }
            throw failure;
        } finally {
            if (previousDevice != deviceId) {
                runtime.affinityManager().setDeviceForCurrentThread(previousDevice);
            }
        }
    }

    @Override
    public INDArray pullRows(
            INDArray source, INDArray destination, int sourceDimension, int[] indexes) {
        long[] longIndexes = ArrayUtil.toLongArray(indexes);
        int axis = normalizeAxis(source, sourceDimension);
        long[] expectedShape = pullRowsOutputShape(source, axis, longIndexes);
        validatePullRowsIndexes(source, axis, longIndexes);

        if (destination == null) {
            return pullRows(source, axis, longIndexes, order());
        }
        if (source.dataType() != destination.dataType()) {
            throw new IllegalArgumentException(
                    "Source and destination data types must match");
        }
        if (!Arrays.equals(expectedShape, destination.shape())) {
            throw new IllegalArgumentException(
                    "Expected pullRows destination shape "
                            + Arrays.toString(expectedShape) + ", got "
                            + Arrays.toString(destination.shape()));
        }

        int deviceId = runtime.affinityManager().getDeviceForArray(source);
        int destinationDevice = runtime.affinityManager().getDeviceForArray(destination);
        if (destinationDevice != deviceId) {
            throw new IllegalArgumentException(
                    "Vulkan pullRows source and destination must belong to the same device");
        }

        int previousDevice = runtime.affinityManager().getDeviceForCurrentThread();
        try {
            if (previousDevice != deviceId) {
                runtime.affinityManager().setDeviceForCurrentThread(deviceId);
            }
            executePullRows(source, destination, axis, longIndexes);
            return destination;
        } finally {
            if (previousDevice != deviceId) {
                runtime.affinityManager().setDeviceForCurrentThread(previousDevice);
            }
        }
    }

    private void executePullRows(
            INDArray source, INDArray destination, int axis, long[] indexes) {
        INDArray indexArray = createLongMetadata(
                indexes, new long[]{indexes.length});
        try (OpaqueNDArray sourceOpaque =
                     OpaqueNDArray.fromINDArrayUncached(runtime, source);
             OpaqueNDArray destinationOpaque =
                     OpaqueNDArray.fromINDArrayUncached(runtime, destination);
             OpaqueNDArray indexOpaque =
                     OpaqueNDArray.fromINDArrayUncached(runtime, indexArray)) {
            invokeNative("Vulkan pullRows", () -> runtime.nativeOps().pullRows(
                    null, sourceOpaque, destinationOpaque, indexes.length,
                    indexOpaque, axis));
        } finally {
            indexArray.close();
        }
    }

    private static long[] pullRowsOutputShape(
            INDArray source, int axis, long[] indexes) {
        if (source == null) {
            throw new IllegalArgumentException("Source array cannot be null");
        }
        if (indexes == null || indexes.length == 0) {
            throw new IllegalArgumentException("Indexes cannot be null or empty");
        }
        if (source.rank() == 1) {
            return new long[]{indexes.length};
        }
        if (source.rank() != 2) {
            throw new UnsupportedOperationException(
                    "Vulkan pullRows follows the established rank-1/2 NativeOps contract");
        }
        return axis == 1
                ? new long[]{indexes.length, source.size(1)}
                : new long[]{source.size(0), indexes.length};
    }

    private static void validatePullRowsIndexes(
            INDArray source, int axis, long[] indexes) {
        long tadLength = source.rank() == 1 ? 1 : source.size(axis);
        long tadCount = tadLength == 0 ? 0 : source.length() / tadLength;
        for (long index : indexes) {
            if (index < 0 || index >= tadCount) {
                throw new IllegalArgumentException(
                        "pullRows index " + index + " is outside [0," + tadCount + ")");
            }
        }
    }

    private static int normalizeAxis(INDArray source, int sourceDimension) {
        if (source == null) {
            throw new IllegalArgumentException("Source array cannot be null");
        }
        int rank = source.rank();
        int axis = sourceDimension < 0 ? sourceDimension + rank : sourceDimension;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException(
                    "Invalid source dimension " + sourceDimension + " for rank " + rank);
        }
        return axis;
    }

    @Override
    public void shuffle(INDArray array, Random random, long... dimensions) {
        shuffle(Collections.singletonList(array), random,
                Collections.singletonList(dimensions));
    }

    @Override
    public void shuffle(List<INDArray> arrays, Random random, List<long[]> dimensions) {
        if (arrays == null || arrays.isEmpty()) {
            throw new IllegalArgumentException("No input arrays were provided for shuffle");
        }
        if (random == null) {
            throw new IllegalArgumentException("Shuffle random generator cannot be null");
        }
        if (dimensions == null || dimensions.isEmpty()) {
            throw new IllegalArgumentException("Shuffle dimensions cannot be null or empty");
        }
        if (dimensions.size() > 1 && dimensions.size() != arrays.size()) {
            throw new IllegalArgumentException(
                    "Shuffle requires either one shared dimension vector or one vector per array");
        }

        List<long[]> normalizedDimensions =
                normalizeShuffleDimensions(arrays, dimensions);
        long itemCount = indexedItemCount(arrays.get(0), normalizedDimensions.get(0));
        if (itemCount <= 0 || itemCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Shuffle indexed item count must be in [1,"
                            + Integer.MAX_VALUE + "], got " + itemCount);
        }
        for (int index = 1; index < arrays.size(); ++index) {
            long candidate = indexedItemCount(
                    arrays.get(index), normalizedDimensions.get(index));
            if (candidate != itemCount) {
                throw new IllegalArgumentException(
                        "All shuffled arrays must expose the same indexed item count");
            }
        }

        int deviceId = requireCommonVulkanDevice(arrays, "shuffle");
        int previousDevice = runtime.affinityManager().getDeviceForCurrentThread();
        INDArray mapArray = null;
        try {
            if (previousDevice != deviceId) {
                runtime.affinityManager().setDeviceForCurrentThread(deviceId);
            }

            int[] map = ArrayUtil.buildInterleavedVector(random, (int) itemCount);
            mapArray = createIntMetadata(map, new long[]{map.length});
            try (OpaqueNDArray mapOpaque =
                         OpaqueNDArray.fromINDArrayUncached(runtime, mapArray)) {
                if (hasDistinctDimensions(normalizedDimensions)) {
                    for (int index = 0; index < arrays.size(); ++index) {
                        INDArray dimensionArray = createLongMetadata(
                                normalizedDimensions.get(index),
                                new long[]{normalizedDimensions.get(index).length});
                        try {
                            executeShuffle(
                                    new INDArray[]{arrays.get(index)},
                                    dimensionArray, mapOpaque);
                        } finally {
                            dimensionArray.close();
                        }
                    }
                } else {
                    long[] sharedDimensions = normalizedDimensions.get(0);
                    INDArray dimensionArray = createLongMetadata(
                            sharedDimensions, new long[]{sharedDimensions.length});
                    try {
                        executeShuffle(
                                arrays.toArray(new INDArray[0]),
                                dimensionArray, mapOpaque);
                    } finally {
                        dimensionArray.close();
                    }
                }
            }
        } finally {
            if (mapArray != null) {
                mapArray.close();
            }
            if (previousDevice != deviceId) {
                runtime.affinityManager().setDeviceForCurrentThread(previousDevice);
            }
        }
    }

    @Override
    public void shuffle(Collection<INDArray> arrays, Random random, long... dimensions) {
        if (arrays == null) {
            throw new IllegalArgumentException("Shuffle arrays cannot be null");
        }
        shuffle(new ArrayList<>(arrays), random,
                Collections.singletonList(dimensions));
    }

    private void executeShuffle(
            INDArray[] arrays, INDArray dimensions, OpaqueNDArray shuffleMap) {
        try (OpaqueNDArrayArr arrayPointers =
                     OpaqueNDArrayArr.createFrom(runtime, arrays);
             OpaqueNDArray dimensionOpaque =
                     OpaqueNDArray.fromINDArrayUncached(runtime, dimensions)) {
            invokeNative("Vulkan shuffle", () -> runtime.nativeOps().shuffle(
                    null, arrayPointers, arrayPointers, arrays.length,
                    dimensionOpaque, shuffleMap));
        }
    }

    private void invokeNative(String operation, Runnable invocation) {
        runtime.nativeOps().clearLastError();
        invocation.run();
        int errorCode = runtime.nativeOps().lastErrorCode();
        if (errorCode != 0) {
            String message = runtime.nativeOps().lastErrorMessage();
            runtime.nativeOps().clearLastError();
            throw new IllegalStateException(
                    operation + " failed"
                            + (message == null || message.isEmpty() ? "" : ": " + message));
        }
    }

    private int requireCommonVulkanDevice(
            List<INDArray> arrays, String operation) {
        INDArray first = arrays.get(0);
        if (first == null) {
            throw new IllegalArgumentException(
                    "Vulkan " + operation + " received a null array at index 0");
        }
        int deviceId = runtime.affinityManager().getDeviceForArray(first);
        for (int index = 1; index < arrays.size(); ++index) {
            INDArray array = arrays.get(index);
            if (array == null) {
                throw new IllegalArgumentException(
                        "Vulkan " + operation + " received a null array at index " + index);
            }
            int candidateDevice = runtime.affinityManager().getDeviceForArray(array);
            if (candidateDevice != deviceId) {
                throw new IllegalArgumentException(
                        "Vulkan " + operation
                                + " requires every array to belong to the same device");
            }
        }
        return deviceId;
    }

    private static List<long[]> normalizeShuffleDimensions(
            List<INDArray> arrays, List<long[]> dimensions) {
        List<long[]> normalized = new ArrayList<>(arrays.size());
        for (int arrayIndex = 0; arrayIndex < arrays.size(); ++arrayIndex) {
            INDArray array = arrays.get(arrayIndex);
            if (array == null) {
                throw new IllegalArgumentException(
                        "Shuffle received a null array at index " + arrayIndex);
            }
            long[] requested = dimensions.size() == 1
                    ? dimensions.get(0)
                    : dimensions.get(arrayIndex);
            normalized.add(normalizeDimensions(array, requested));
        }
        return normalized;
    }

    private static long[] normalizeDimensions(
            INDArray array, long[] dimensions) {
        if (dimensions == null || dimensions.length == 0) {
            throw new IllegalArgumentException(
                    "Each shuffled array requires at least one dimension");
        }

        long[] normalized = new long[dimensions.length];
        boolean[] seen = new boolean[array.rank()];
        for (int index = 0; index < dimensions.length; ++index) {
            long requested = dimensions[index];
            long axis = requested < 0 ? requested + array.rank() : requested;
            if (axis < 0 || axis >= array.rank()) {
                throw new IllegalArgumentException(
                        "Shuffle dimension " + requested
                                + " is invalid for rank " + array.rank());
            }
            int axisIndex = (int) axis;
            if (seen[axisIndex]) {
                throw new IllegalArgumentException(
                        "Shuffle dimensions cannot contain duplicates");
            }
            seen[axisIndex] = true;
            normalized[index] = axis;
        }
        return normalized;
    }

    private static long indexedItemCount(
            INDArray array, long[] dimensions) {
        if (array.rank() == 1) {
            return array.length();
        }

        long tadLength = 1L;
        try {
            for (long dimension : dimensions) {
                tadLength = Math.multiplyExact(
                        tadLength, array.size((int) dimension));
            }
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException(
                    "Shuffle TAD length exceeds the supported integer range", overflow);
        }
        if (tadLength <= 0 || array.length() % tadLength != 0) {
            throw new IllegalArgumentException(
                    "Shuffle dimensions do not define an integral TAD partition");
        }
        return array.length() / tadLength;
    }

    private static boolean hasDistinctDimensions(
            List<long[]> dimensions) {
        long[] shared = dimensions.get(0);
        for (int index = 1; index < dimensions.size(); ++index) {
            if (!Arrays.equals(shared, dimensions.get(index))) {
                return true;
            }
        }
        return false;
    }

    private INDArray createLongMetadata(long[] values, long[] shape) {
        return create(
                values, shape, strides(shape, 'c'), 'c', DataType.LONG, null);
    }

    private INDArray createIntMetadata(int[] values, long[] shape) {
        return create(
                values, shape, strides(shape, 'c'), 'c', DataType.INT, null);
    }

    @Override
    public INDArray sort(INDArray x, boolean descending) {
        if (x == null || x.isScalar() || x.isEmpty()) {
            return x;
        }
        try (OpaqueNDArray opaque = OpaqueNDArray.fromINDArrayUncached(runtime, x)) {
            invokeNative("Vulkan global sort",
                    () -> runtime.nativeOps().sort(null, opaque, descending));
        }
        return x;
    }

    @Override
    public INDArray sort(INDArray x, boolean descending, long... dimension) {
        if (x == null || x.isScalar() || x.isEmpty()) {
            return x;
        }
        if (dimension == null || dimension.length == 0) {
            return sort(x, descending);
        }
        long[] axes = dimension.clone();
        Arrays.sort(axes);
        try (OpaqueNDArray opaque = OpaqueNDArray.fromINDArrayUncached(runtime, x)) {
            invokeNative("Vulkan dimension sort",
                    () -> runtime.nativeOps().sortTad(null, opaque, axes, axes.length,
                            null, null, descending));
        }
        return x;
    }

    @Override
    public INDArray sortCooIndices(INDArray x) {
        // Vulkan NDArrays are dense by construction; a COO index request is
        // already canonical and therefore has no data movement to perform.
        return x;
    }
}
