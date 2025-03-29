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

package org.nd4j.linalg.jcublas;


import com.google.flatbuffers.FlatBufferBuilder;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.graph.FlatArray;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.BaseNDArrayProxy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.JvmShapeInfo;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.CudaLongDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaUtf8Buffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.api.memory.MemcpyDirection;
import org.nd4j.linalg.workspace.WorkspaceUtils;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;

/**
 *
 *
 * Created by mjk on 8/23/14.
 *
 * @author mjk
 * @author Adam Gibson
 * @author raver119@gmail.com
 */
@Slf4j
public class JCublasNDArray extends BaseNDArray {


    public JCublasNDArray(DataBuffer buffer, CudaLongDataBuffer shapeInfo, long[] javaShapeInfo) {
        this.data = buffer;
        setShapeInfoDataBuffer(shapeInfo);
        this.jvmShapeInfo = new JvmShapeInfo(javaShapeInfo);
    }

    public JCublasNDArray(double[][] data) {
        super(data);
    }

    public JCublasNDArray(double[][] data, char ordering) {
        super(data, ordering);
    }

    public JCublasNDArray(int[] shape, DataBuffer buffer) {
        super(shape, buffer);
    }

    /**
     * Create this JCublasNDArray with the given data and shape and 0 offset
     *
     * @param data     the data to use
     * @param shape    the shape of the JCublasNDArray
     * @param ordering
     */
    public JCublasNDArray(float[] data, int[] shape, char ordering) {
        super(data, shape, ordering);
    }

    /**
     * @param data     the data to use
     * @param shape    the shape of the JCublasNDArray
     * @param offset   the desired offset
     * @param ordering the ordering of the JCublasNDArray
     */
    public JCublasNDArray(float[] data, int[] shape, long offset, char ordering) {
        super(data, shape, offset, ordering);
    }


    public JCublasNDArray(DataBuffer buffer, LongShapeDescriptor longShapeDescriptor) {
        super(buffer, longShapeDescriptor);
    }

    /**
     * Construct an JCublasNDArray of the specified shape
     * with an empty data array
     *
     * @param shape    the shape of the JCublasNDArray
     * @param stride   the stride of the JCublasNDArray
     * @param offset   the desired offset
     * @param ordering the ordering of the JCublasNDArray
     */
    public JCublasNDArray(int[] shape, int[] stride, long offset, char ordering) {
        super(shape, stride, offset, ordering);
    }

    /**
     * Construct an JCublasNDArray of the specified shape, with optional initialization
     *
     * @param shape    the shape of the JCublasNDArray
     * @param stride   the stride of the JCublasNDArray
     * @param offset   the desired offset
     * @param ordering the ordering of the JCublasNDArray
     * @param initialize Whether to initialize the INDArray. If true: initialize. If false: don't.
     */
    public JCublasNDArray(int[] shape, int[] stride, long offset, char ordering, boolean initialize) {
        super(shape, stride, offset, ordering, initialize);
    }

    public JCublasNDArray(long[] shape, long[] stride, long offset, char ordering, boolean initialize) {
        super(shape, stride, offset, ordering, initialize);
    }

    /**
     * Create the JCublasNDArray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape    the shape of the JCublasNDArray
     * @param stride   the stride of the JCublasNDArray
     * @param ordering the ordering of the JCublasNDArray
     */
    public JCublasNDArray(int[] shape, int[] stride, char ordering) {

        super(shape, stride, ordering);

    }

    public JCublasNDArray(int[] shape, long offset, char ordering) {
        super(shape, offset, ordering);
    }

    public JCublasNDArray(long[] shape, long offset, char ordering) {
        super(shape, offset, ordering);
    }

    public JCublasNDArray(int[] shape) {
        super(shape);
    }

    public JCublasNDArray(long[] shape) {
        super(shape);
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     * @param ordering
     */
    public JCublasNDArray(int newRows, int newColumns, char ordering) {
        super(newRows, newColumns, ordering);

    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices   the slices to merge
     * @param shape    the shape of the JCublasNDArray
     * @param ordering
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape, char ordering) {
        super(slices, shape, ordering);
    }

    public JCublasNDArray(List<INDArray> slices, long[] shape, char ordering) {
        super(slices, shape, ordering);
    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices   the slices to merge
     * @param shape    the shape of the JCublasNDArray
     * @param stride
     * @param ordering
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape, int[] stride, char ordering) {
        super(slices, shape, stride, ordering);

    }

    public JCublasNDArray(float[] data, int[] shape, int[] stride, char ordering) {
        super(data, shape, stride, ordering);

    }

    public JCublasNDArray(float[] data, int[] shape, int[] stride, long offset, char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    public JCublasNDArray(float[] data, long[] shape, long[] stride, long offset, char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    public JCublasNDArray(double[] data, long[] shape, long[] stride, long offset, char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    public JCublasNDArray(int[] data, int[] shape, int[] strides) {
        super(data, shape, strides);
    }
    public JCublasNDArray(DataBuffer data, int[] shape) {
        super(data, shape);
    }

    public JCublasNDArray(DataBuffer data, long[] shape) {
        super(data, shape);
    }

    public JCublasNDArray(DataBuffer buffer, int[] shape, long offset) {
        super(buffer, shape, offset);
    }

    /**
     * Create this JCublasNDArray with the given data and shape and 0 offset
     *
     * @param data  the data to use
     * @param shape the shape of the JCublasNDArray
     */
    public JCublasNDArray(float[] data, int[] shape) {
        super(data, shape);
    }

    public JCublasNDArray(float[] data, int[] shape, long offset) {

        super(data, shape, offset);

    }

    /**
     * Construct an JCublasNDArray of the specified shape
     * with an empty data array
     *
     * @param shape  the shape of the JCublasNDArray
     * @param stride the stride of the JCublasNDArray
     * @param offset the desired offset
     */
    public JCublasNDArray(int[] shape, int[] stride, long offset) {

        super(shape, stride, offset);
    }

    /**
     * Create the JCublasNDArray with
     * the specified shape and stride and an offset of 0
     *
     * @param shape  the shape of the JCublasNDArray
     * @param stride the stride of the JCublasNDArray
     */
    public JCublasNDArray(int[] shape, int[] stride) {
        super(shape, stride);
    }

    public JCublasNDArray(int[] shape, long offset) {
        super(shape, offset);
    }

    public JCublasNDArray(int[] shape, char ordering) {
        super(shape, ordering);
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public JCublasNDArray(int newRows, int newColumns) {
        super(newRows, newColumns);
    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the JCublasNDArray
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape) {
        super(slices, shape);
    }

    public JCublasNDArray(List<INDArray> slices, long[] shape) {
        super(slices, shape);
    }

    /**
     * Create an JCublasNDArray from the specified slices.
     * This will go through and merge all of the
     * data from each slice in to one JCublasNDArray
     * which will then take the specified shape
     *
     * @param slices the slices to merge
     * @param shape  the shape of the JCublasNDArray
     * @param stride
     */
    public JCublasNDArray(List<INDArray> slices, int[] shape, int[] stride) {
        super(slices, shape, stride);

    }

    public JCublasNDArray(float[] data, int[] shape, int[] stride) {
        super(data, shape, stride);
    }


    public JCublasNDArray(float[] data, int[] shape, int[] stride, long offset) {
        super(data, shape, stride, offset);
    }

    public JCublasNDArray(float[] data) {
        super(data);
    }


    public JCublasNDArray(JCublasNDArray doubleMatrix) {
        this(new long[] {doubleMatrix.rows(), doubleMatrix.columns()});
        this.data = dup().data();
    }

    public JCublasNDArray(double[] data, int[] shape, int[] stride, long offset) {
        super(data, shape, stride, offset);
    }

    public JCublasNDArray(float[][] floats) {
        super(floats);
    }

    public JCublasNDArray(float[][] data, char ordering) {
        super(data, ordering);
    }

    public JCublasNDArray(DataBuffer buffer, int[] shape, long offset, char ordering) {
        super(buffer, shape, offset, ordering);
    }

    public JCublasNDArray() {}

    public JCublasNDArray(DataBuffer buffer) {
        super(buffer);
    }

    public JCublasNDArray(DataBuffer buffer, int[] shape, int[] stride, long offset, char ordering) {
        super(buffer, shape, stride, offset, ordering);
    }

    public JCublasNDArray(DataBuffer buffer, long[] shape, long[] stride, long offset, char ordering, DataType dataType) {
        super(buffer, shape, stride, offset, ordering, dataType);
    }

    public JCublasNDArray(DataBuffer buffer, long[] shape, long[] stride, long offset, long ews, char ordering, DataType dataType) {
        super(buffer, shape, stride, offset, ews, ordering, dataType);
    }

    public JCublasNDArray(DataBuffer buffer, long[] shape, long[] stride, char ordering, DataType dataType) {
        super(buffer, shape, stride, ordering, dataType);
    }

    public JCublasNDArray(float[] data, char order) {
        super(data, order);
    }

    public JCublasNDArray(DataBuffer buffer, int[] shape, int[] strides) {
        super(buffer, shape, strides);
    }

    public JCublasNDArray(double[] data, int[] shape, char ordering) {
        super(data, shape, ordering);
    }

    public JCublasNDArray(double[] data, long[] shape, char ordering) {
        super(data, shape, ordering);
    }

    public JCublasNDArray(float[] data, long[] shape, char ordering) {
        super(data, shape, ordering);
    }

    public JCublasNDArray(double[] data, int[] shape, int[] stride, long offset, char ordering) {
        super(data, shape, stride, offset, ordering);
    }

    public JCublasNDArray(DataType dataType, long[] shape, long[] paddings, long[] paddingOffsets, char ordering,
            MemoryWorkspace workspace){
        super(dataType, shape, paddings, paddingOffsets, ordering, workspace);
    }

    public JCublasNDArray(DataBuffer data, long[] newShape, long[] newStride, long offset, long ews, char ordering, DataType dataType, boolean isView) {
        super(data, newShape, newStride, offset, ews, ordering, dataType, isView);
    }

    public JCublasNDArray(LongShapeDescriptor longShapeDescriptor) {
        super(longShapeDescriptor);
    }

    public JCublasNDArray(DataType dataType, long[] shape, long[] strides, int i, char ordering, MemoryWorkspace currentWorkspace) {
        super(dataType,shape,strides,currentWorkspace);
    }

    @Override
    public INDArray dup() {
        if (this.isCompressed() && this.ordering() == Nd4j.order().charValue()) {
            INDArray ret = Nd4j.createArrayFromShapeBuffer(data().dup(), this.shapeInfoDataBuffer());
            ret.markAsCompressed(true);
            return ret;
        }
        /*
            Special case for cuda: if we have not a view, and shapes do match - we
        */

        val res = super.dup();
        Nd4j.getExecutioner().commit();
        return res;
    }

    @Override
    public INDArray dup(char order) {
        if (this.isCompressed() && this.ordering() == order) {
            INDArray ret = Nd4j.createArrayFromShapeBuffer(data().dup(), this.shapeInfoDataBuffer());
            ret.markAsCompressed(true);
            return ret;
        }

      return super.dup(order);
    }

    @Override
    public boolean equals(Object o) {
        return super.equals(o);
    }

    /**
     * Generate string representation of the matrix.
     */
    @Override
    public String toString() {
        if (!isS())
            AtomicAllocator.getInstance().synchronizeHostData(this);
        return super.toString();
    }

    /**
     *
     * PLEASE NOTE: Never use this method, unless you 100% have to
     *
     * @param buffer
     */
    public void setShapeInfoDataBuffer(DataBuffer buffer) {
        this.jvmShapeInfo = new JvmShapeInfo(buffer.asLong());
        this.shapeInfoDataBuffer = buffer;
    }

    private Object writeReplace() throws java.io.ObjectStreamException {
        return new BaseNDArrayProxy(this);
    }

    @Override
    public INDArray permutei(long... rearrange) {
        Nd4j.getExecutioner().push();

        return super.permutei(rearrange);
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
        WorkspaceUtils.assertValidArray(this, "Cannot duplicate array");
        DataBuffer rb = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? Nd4j.getDataBufferFactory().createSame(this.data, false) : Nd4j.getDataBufferFactory().createSame(this.data, false, Nd4j.getMemoryManager().getCurrentWorkspace());

        INDArray ret = Nd4j.createArrayFromShapeBuffer(rb, this.shapeInfoDataBuffer());


        if (blocking)
            Nd4j.getExecutioner().push();



        AtomicAllocator allocator = AtomicAllocator.getInstance();
        val context = (CudaContext) allocator.getDeviceContext();

        AllocationPoint srcPoint = allocator.getAllocationPoint(this);
        AllocationPoint dstPoint = allocator.getAllocationPoint(ret);

        int route = 0;
        MemcpyDirection direction = MemcpyDirection.HOST_TO_HOST;
        val prof = PerformanceTracker.getInstance().helperStartTransaction();

        if (srcPoint.isActualOnDeviceSide()) {
            route = 1;
            NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(dstPoint.getDevicePointer(), srcPoint.getDevicePointer(), this.data.length() * this.data.getElementSize(), CudaConstants.cudaMemcpyDeviceToDevice, blocking ? context.getOldStream() : context.getSpecialStream());
            dstPoint.tickDeviceWrite();
            direction = MemcpyDirection.DEVICE_TO_DEVICE;
        } else {
            route = 3;
            NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(dstPoint.getDevicePointer(), srcPoint.getHostPointer(), this.data.length() * this.data.getElementSize(), CudaConstants.cudaMemcpyHostToDevice, blocking ? context.getOldStream() : context.getSpecialStream());
            dstPoint.tickDeviceWrite();
            direction = MemcpyDirection.HOST_TO_DEVICE;
        }



        if (blocking)
            context.syncOldStream();
        else
            context.syncSpecialStream();

        PerformanceTracker.getInstance().helperRegisterTransaction(dstPoint.getDeviceId(), prof, dstPoint.getNumberOfBytes(), direction);


        return ret;
    }

    @Override
    public INDArray leverageTo(String id) {
        if (!isAttached()) {
            return this;
        }

        if (!Nd4j.getWorkspaceManager().checkIfWorkspaceExists(id)) {
            return this;
        }

        WorkspaceUtils.assertValidArray(this, "Cannot leverage INDArray to new workspace");

        MemoryWorkspace current = Nd4j.getMemoryManager().getCurrentWorkspace();

        MemoryWorkspace target = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(id);

        if (current == target) {
            return this;
        }

        if (this.data.getParentWorkspace() == target) {
            return this;
        }

        Nd4j.getMemoryManager().setCurrentWorkspace(target);

        INDArray copy = null;
        if (!this.isView()) {
            Nd4j.getExecutioner().commit();

            val buffer = Nd4j.createBuffer(this.length(), false);

            val pointDst = AtomicAllocator.getInstance().getAllocationPoint(buffer);
            val pointSrc = AtomicAllocator.getInstance().getAllocationPoint(this.data);

            val context = AtomicAllocator.getInstance().getFlowController().prepareAction(pointDst, pointSrc);

            MemcpyDirection direction = MemcpyDirection.DEVICE_TO_DEVICE;
            val perfD = PerformanceTracker.getInstance().helperStartTransaction();

            if (pointSrc.isActualOnDeviceSide()) {
                if (NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getDevicePointer(), pointSrc.getDevicePointer(), this.length() * Nd4j.sizeOfDataType(buffer.dataType()), CudaConstants.cudaMemcpyDeviceToDevice, context.getOldStream()) == 0)
                    throw new ND4JIllegalStateException("memcpyAsync failed");
            } else {
                if (NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getDevicePointer(), pointSrc.getHostPointer(), this.length() * Nd4j.sizeOfDataType(buffer.dataType()), CudaConstants.cudaMemcpyHostToDevice, context.getOldStream()) == 0)
                    throw new ND4JIllegalStateException("memcpyAsync failed");

                direction = MemcpyDirection.HOST_TO_DEVICE;
            }

            context.syncOldStream();

            PerformanceTracker.getInstance().helperRegisterTransaction(pointDst.getDeviceId(), perfD, pointSrc.getNumberOfBytes(), direction);

            copy = Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInfoDataBuffer());

            // tag buffer as valid on device side
            pointDst.tickDeviceWrite();

            AtomicAllocator.getInstance().getFlowController().registerAction(context, pointDst, pointSrc);
        } else {
            copy = this.dup(this.ordering());

            Nd4j.getExecutioner().commit();
        }

        Nd4j.getMemoryManager().setCurrentWorkspace(current);

        return copy;
    }

    @Override
    public INDArray migrate() {
        WorkspaceUtils.assertValidArray(this, "Cannot leverage INDArray to new workspace");
        MemoryWorkspace current = Nd4j.getMemoryManager().getCurrentWorkspace();

        if (current == null)
            return this;

        INDArray copy = null;

        if (!this.isView()) {
            Nd4j.getExecutioner().commit();

            val buffer = Nd4j.createBuffer(this.dataType(), this.length(), false);

            val pointDst = AtomicAllocator.getInstance().getAllocationPoint(buffer);
            val pointSrc = AtomicAllocator.getInstance().getAllocationPoint(this.data);


            val context = AtomicAllocator.getInstance().getFlowController().prepareAction(pointDst, pointSrc);

            MemcpyDirection direction = MemcpyDirection.DEVICE_TO_DEVICE;
            val perfD = PerformanceTracker.getInstance().helperStartTransaction();

            if (pointSrc.isActualOnDeviceSide()) {
                if (NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getDevicePointer(), pointSrc.getDevicePointer(), this.length() * Nd4j.sizeOfDataType(buffer.dataType()), CudaConstants.cudaMemcpyDeviceToDevice, context.getOldStream()) == 0)
                    throw new ND4JIllegalStateException("memcpyAsync failed");
            } else {
                if (NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getDevicePointer(), pointSrc.getHostPointer(), this.length() * Nd4j.sizeOfDataType(buffer.dataType()), CudaConstants.cudaMemcpyHostToDevice, context.getOldStream()) == 0)
                    throw new ND4JIllegalStateException("memcpyAsync failed");

                direction = MemcpyDirection.HOST_TO_DEVICE;
            }

            context.syncOldStream();

            PerformanceTracker.getInstance().helperRegisterTransaction(pointDst.getDeviceId(), perfD, pointDst.getNumberOfBytes(), direction);

            if (pointDst.getDeviceId() != Nd4j.getMemoryManager().getCurrentWorkspace().getDeviceId()) {
                pointDst.setDeviceId(Nd4j.getMemoryManager().getCurrentWorkspace().getDeviceId());
            }

            copy = Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInfoDataBuffer());

            // tag buffer as valid on device side
            pointDst.tickDeviceWrite();

            AtomicAllocator.getInstance().getFlowController().registerAction(context, pointDst, pointSrc);
        } else {
            copy = this.dup(this.ordering());
        }

        return copy;
    }

    protected int stringBuffer(FlatBufferBuilder builder, DataBuffer buffer) {
        Preconditions.checkArgument(buffer.dataType() == DataType.UTF8, "This method can be called on UTF8 buffers only");
        try {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            DataOutputStream dos = new DataOutputStream(bos);

            val numWords = this.length();
            val ub = (CudaUtf8Buffer) buffer;
            // writing length first
            val t = length();
            val ptr = (BytePointer) ub.pointer();

            // now write all strings as bytes
            for (int i = 0; i < ub.length(); i++) {
                dos.writeByte(ptr.get(i));
            }

            val bytes = bos.toByteArray();
            return FlatArray.createBufferVector(builder, bytes);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String getString(long index) {
        if (!isS())
            throw new UnsupportedOperationException("This method is usable only on String dataType, but got [" + this.dataType() + "]");

        return ((CudaUtf8Buffer) data).getString(index);
    }


}
