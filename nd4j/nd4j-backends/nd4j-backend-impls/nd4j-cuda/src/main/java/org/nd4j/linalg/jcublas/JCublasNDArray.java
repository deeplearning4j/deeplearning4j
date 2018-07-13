/*-
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

package org.nd4j.linalg.jcublas;


import lombok.val;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.BaseNDArrayProxy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.JvmShapeInfo;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.CudaLongDataBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.memory.MemcpyDirection;
import org.nd4j.linalg.workspace.WorkspaceUtils;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 *
 * Created by mjk on 8/23/14.
 *
 * @author mjk
 * @author Adam Gibson
 * @author raver119@gmail.com
 */

public class JCublasNDArray extends BaseNDArray {


    public JCublasNDArray(DataBuffer buffer, CudaLongDataBuffer shapeInfo, long[] javaShapeInfo) {
        this.jvmShapeInfo = new JvmShapeInfo(javaShapeInfo);
        this.shapeInformation = shapeInfo;
        this.data = buffer;
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

    public JCublasNDArray(int[] shape, int[] stride, long offset, char ordering, boolean initialize, DataBuffer.Type dType) {
        super(shape, stride, offset, ordering, initialize, dType);
    }

    public JCublasNDArray(long[] shape, long[] stride, long offset, char ordering, boolean initialize, DataBuffer.Type dType) {
        super(shape, stride, offset, ordering, initialize, dType);
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

    public JCublasNDArray(DataBuffer data, int[] shape, int[] stride, long offset) {
        super(data, shape, stride, offset);
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

    public JCublasNDArray(DataBuffer buffer, long[] shape, long[] stride, long offset, char ordering) {
        super(buffer, shape, stride, offset, ordering);
    }

    public JCublasNDArray(float[] data, char order) {
        super(data, order);
    }

    public JCublasNDArray(FloatBuffer floatBuffer, char order) {
        super(floatBuffer, order);
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
        /*
        if (!isView() && ordering() == Nd4j.order() && Shape.strideDescendingCAscendingF(this)) {
            AtomicAllocator allocator = AtomicAllocator.getInstance();
            INDArray array = Nd4j.createUninitialized(shape(), ordering());
        
            CudaContext context = allocator.getFlowController().prepareAction(array, this);
        
            Configuration configuration = CudaEnvironment.getInstance().getConfiguration();
        
            if (configuration.getMemoryModel() == Configuration.MemoryModel.IMMEDIATE && configuration.getFirstMemory() == AllocationStatus.DEVICE) {
        //                log.info("Path 0");
                allocator.memcpyDevice(array.data(), allocator.getPointer(this.data, context), this.data.length() * this.data().getElementSize(), 0, context);
            } else if (configuration.getMemoryModel() == Configuration.MemoryModel.DELAYED || configuration.getFirstMemory() == AllocationStatus.HOST) {
                AllocationPoint pointSrc = allocator.getAllocationPoint(this);
                AllocationPoint pointDst = allocator.getAllocationPoint(array);
        
                if (pointSrc.getAllocationStatus() == AllocationStatus.HOST) {
        //                    log.info("Path A");
                    NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getPointers().getHostPointer(), pointSrc.getPointers().getHostPointer(), length * data.getElementSize(), CudaConstants.cudaMemcpyHostToHost, context.getOldStream());
                } else {
        //                    log.info("Path B. SRC dId: [{}], DST dId: [{}], cId: [{}]", pointSrc.getDeviceId(), pointDst.getDeviceId(), allocator.getDeviceId());
                    // this code branch is possible only with DELAYED memoryModel and src point being allocated on device
                    if (pointDst.getAllocationStatus() != AllocationStatus.DEVICE) {
                        allocator.getMemoryHandler().alloc(AllocationStatus.DEVICE, pointDst, pointDst.getShape(), false);
                    }
        
                    NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getPointers().getDevicePointer(), pointSrc.getPointers().getHostPointer(), length * data.getElementSize(), CudaConstants.cudaMemcpyHostToDevice, context.getOldStream());
                }
            }
        
            allocator.getFlowController().registerAction(context, array, this);
            return array;
        } else */return super.dup();
    }

    @Override
    public INDArray dup(char order) {
        if (this.isCompressed() && this.ordering() == order) {
            INDArray ret = Nd4j.createArrayFromShapeBuffer(data().dup(), this.shapeInfoDataBuffer());
            ret.markAsCompressed(true);
            return ret;
        }
        /*
        if (!isView() && ordering() == order && Shape.strideDescendingCAscendingF(this)) {
            AtomicAllocator allocator = AtomicAllocator.getInstance();
            INDArray array = Nd4j.createUninitialized(shape(), order);
        
            CudaContext context = allocator.getFlowController().prepareAction(array, this);
        
            Configuration configuration = CudaEnvironment.getInstance().getConfiguration();
        
            if (configuration.getMemoryModel() == Configuration.MemoryModel.IMMEDIATE && configuration.getFirstMemory() == AllocationStatus.DEVICE) {
                allocator.memcpyDevice(array.data(), allocator.getPointer(this.data, context), this.data.length() * this.data().getElementSize(), 0, context);
            } else if (configuration.getMemoryModel() == Configuration.MemoryModel.DELAYED || configuration.getFirstMemory() == AllocationStatus.HOST) {
                AllocationPoint pointSrc = allocator.getAllocationPoint(this);
                AllocationPoint pointDst = allocator.getAllocationPoint(array);
        
                if (pointSrc.getAllocationStatus() == AllocationStatus.HOST) {
                    NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getPointers().getHostPointer(), pointSrc.getPointers().getHostPointer(), length * data.getElementSize(), CudaConstants.cudaMemcpyHostToHost, context.getOldStream());
                } else {
                    // this code branch is possible only with DELAYED memoryModel and src point being allocated on device
                    if (pointDst.getAllocationStatus() != AllocationStatus.DEVICE) {
                        allocator.getMemoryHandler().alloc(AllocationStatus.DEVICE, pointDst, pointDst.getShape(), false);
                    }
        
                    NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getPointers().getDevicePointer(), pointSrc.getPointers().getDevicePointer(), length * data.getElementSize(), CudaConstants.cudaMemcpyHostToDevice, context.getOldStream());
                }
            }
        
            allocator.getFlowController().registerAction(context, array, this);
        
            return array;
        } else */return super.dup(order);
    }

    @Override
    public boolean equals(Object o) {
        //if (o != null) AtomicAllocator.getInstance().synchronizeHostData((INDArray) o);
        //AtomicAllocator.getInstance().synchronizeHostData(this);
        return super.equals(o);
    }

    /**
     * Generate string representation of the matrix.
     */
    @Override
    public String toString() {
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
        this.shapeInformation = buffer;
        this.jvmShapeInfo = new JvmShapeInfo(shapeInformation.asLong());
    }

    private Object writeReplace() throws java.io.ObjectStreamException {
        return new BaseNDArrayProxy(this);
    }

    @Override
    public INDArray permutei(int... rearrange) {
        Nd4j.getExecutioner().push();

        return super.permutei(rearrange);
    }

    /**
     * This method does direct array copy. Impossible to use on views or mixed orders.
     *
     * PLEASE NOTE: YOU SHOULD NEVER USE THIS METHOD, UNLESS YOU 100% CLEAR ABOUT IT
     *
     * @return
     */
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


        //Nd4j.getExecutioner().commit();

        AtomicAllocator allocator = AtomicAllocator.getInstance();
        CudaContext context = (CudaContext) allocator.getDeviceContext().getContext();

        AllocationPoint srcPoint = allocator.getAllocationPoint(this);
        AllocationPoint dstPoint = allocator.getAllocationPoint(ret);

        int route = 0;
//        long time1 = System.currentTimeMillis();
        MemcpyDirection direction = MemcpyDirection.HOST_TO_HOST;
        val prof = PerformanceTracker.getInstance().helperStartTransaction();

        if (dstPoint.getAllocationStatus() == AllocationStatus.DEVICE && srcPoint.getAllocationStatus() == AllocationStatus.DEVICE) {
            // d2d copy
            route = 1;
            NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(dstPoint.getDevicePointer(), srcPoint.getDevicePointer(), this.data.length() * this.data.getElementSize(), CudaConstants.cudaMemcpyDeviceToDevice, blocking ? context.getOldStream() : context.getSpecialStream());
            dstPoint.tickDeviceWrite();
            direction = MemcpyDirection.DEVICE_TO_DEVICE;
        } else if (dstPoint.getAllocationStatus() == AllocationStatus.HOST && srcPoint.getAllocationStatus() == AllocationStatus.DEVICE) {
            route = 2;
            NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(dstPoint.getHostPointer(), srcPoint.getDevicePointer(), this.data.length() * this.data.getElementSize(), CudaConstants.cudaMemcpyDeviceToHost, blocking ? context.getOldStream() : context.getSpecialStream());
            dstPoint.tickHostWrite();
            direction = MemcpyDirection.DEVICE_TO_HOST;
        } else if (dstPoint.getAllocationStatus() == AllocationStatus.DEVICE && srcPoint.getAllocationStatus() == AllocationStatus.HOST) {
            route = 3;
            NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(dstPoint.getDevicePointer(), srcPoint.getHostPointer(), this.data.length() * this.data.getElementSize(), CudaConstants.cudaMemcpyHostToDevice, blocking ? context.getOldStream() : context.getSpecialStream());
            dstPoint.tickDeviceWrite();
            direction = MemcpyDirection.HOST_TO_DEVICE;
        } else {
            route = 4;
            NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(dstPoint.getHostPointer(), srcPoint.getHostPointer(), this.data.length() * this.data.getElementSize(), CudaConstants.cudaMemcpyHostToHost, blocking ? context.getOldStream() : context.getSpecialStream());
            dstPoint.tickHostWrite();
        }


        //allocator.memcpyDevice(ret.data(), allocator.getAllocationPoint(this.data).getDevicePointer(), this.data.length() * this.data().getElementSize(), 0, context);

        if (blocking)
            context.syncOldStream();
        else
            context.syncSpecialStream();

        PerformanceTracker.getInstance().helperRegisterTransaction(dstPoint.getDeviceId(), prof, dstPoint.getNumberOfBytes(), direction);

//        AtomicAllocator.getInstance().synchronizeHostData(ret);
/*
        long time2 = System.currentTimeMillis();

        long bytes = this.data.length() * this.data.getElementSize();
        long spent = time2 - time1;

        float bw = (1000 * bytes / spent) / 1024 / 1024.0f / 1024; //1000 / spent * bytes / 1024 / 1024 / 1024;

        log.info("Route: [{}]; Blocking: {}; {} bytes; {} ms; Bandwidth: {} GB/s", route, blocking, bytes, spent, String.format("%.2f", bw));
*/
        return ret;
    }

    @Override
    public INDArray leverageTo(String id) {
        if (!isAttached()) {
//            log.info("Skipping detached");
            return this;
        }

        if (!Nd4j.getWorkspaceManager().checkIfWorkspaceExists(id)) {
//            log.info("Skipping non-existent");
            return this;
        }

        WorkspaceUtils.assertValidArray(this, "Cannot leverage INDArray to new workspace");

        MemoryWorkspace current = Nd4j.getMemoryManager().getCurrentWorkspace();

        MemoryWorkspace target = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(id);

        if (current == target) {
//            log.info("Skipping equals A");
            return this;
        }

        if (this.data.getParentWorkspace() == target) {
//            log.info("Skipping equals B");
            return this;
        }

        Nd4j.getMemoryManager().setCurrentWorkspace(target);

//        log.info("Leveraging...");

        INDArray copy = null;
        if (!this.isView()) {
        //if (1 < 0) {
            Nd4j.getExecutioner().commit();

            DataBuffer buffer = Nd4j.createBuffer(this.lengthLong(), false);

            AllocationPoint pointDst = AtomicAllocator.getInstance().getAllocationPoint(buffer);
            AllocationPoint pointSrc = AtomicAllocator.getInstance().getAllocationPoint(this.data);

            CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(pointDst, pointSrc);
/*
            if (NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(pointDst.getDevicePointer(), 0, 1, 0, context.getOldStream()) == 0)
                throw new ND4JIllegalStateException("memsetAsync 1 failed");

            context.syncOldStream();

            if (NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(pointSrc.getDevicePointer(), 0, 1, 0, context.getOldStream()) == 0)
                throw new ND4JIllegalStateException("memsetAsync 2 failed");

            context.syncOldStream();
*/

            MemcpyDirection direction = MemcpyDirection.DEVICE_TO_DEVICE;
            val perfD = PerformanceTracker.getInstance().helperStartTransaction();

            if (pointSrc.isActualOnDeviceSide()) {
                if (NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getDevicePointer(), pointSrc.getDevicePointer(), this.lengthLong() * Nd4j.sizeOfDataType(buffer.dataType()), CudaConstants.cudaMemcpyDeviceToDevice, context.getOldStream()) == 0)
                    throw new ND4JIllegalStateException("memcpyAsync failed");
            } else {
                if (NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getDevicePointer(), pointSrc.getHostPointer(), this.lengthLong() * Nd4j.sizeOfDataType(buffer.dataType()), CudaConstants.cudaMemcpyHostToDevice, context.getOldStream()) == 0)
                    throw new ND4JIllegalStateException("memcpyAsync failed");

                direction = MemcpyDirection.HOST_TO_DEVICE;
            }

            context.syncOldStream();

            PerformanceTracker.getInstance().helperRegisterTransaction(pointDst.getDeviceId(), perfD, pointSrc.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);

            copy = Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInfoDataBuffer());

            // tag buffer as valid on device side
            pointDst.tickHostRead();
            pointDst.tickDeviceWrite();

            AtomicAllocator.getInstance().getFlowController().registerAction(context, pointDst, pointSrc);
        } else {
            copy = this.dup(this.ordering());

            Nd4j.getExecutioner().commit();
        }

        Nd4j.getMemoryManager().setCurrentWorkspace(current);

        return copy;
    }


    /**
     * This method pulls this INDArray into current Workspace.
     *
     * PLEASE NOTE: If there's no current Workspace - INDArray returned as is
     *
     * @return
     */
    @Override
    public INDArray migrate() {
        WorkspaceUtils.assertValidArray(this, "Cannot leverage INDArray to new workspace");
        MemoryWorkspace current = Nd4j.getMemoryManager().getCurrentWorkspace();

        if (current == null)
            return this;

        INDArray copy = null;

        if (!this.isView()) {
            Nd4j.getExecutioner().commit();

            DataBuffer buffer = Nd4j.createBuffer(this.lengthLong(), false);

            AllocationPoint pointDst = AtomicAllocator.getInstance().getAllocationPoint(buffer);
            AllocationPoint pointSrc = AtomicAllocator.getInstance().getAllocationPoint(this.data);

//            CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

            CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(pointDst, pointSrc);

            MemcpyDirection direction = MemcpyDirection.DEVICE_TO_DEVICE;
            val perfD = PerformanceTracker.getInstance().helperStartTransaction();

            if (pointSrc.isActualOnDeviceSide()) {
                if (NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getDevicePointer(), pointSrc.getDevicePointer(), this.lengthLong() * Nd4j.sizeOfDataType(buffer.dataType()), CudaConstants.cudaMemcpyDeviceToDevice, context.getOldStream()) == 0)
                    throw new ND4JIllegalStateException("memcpyAsync failed");
            } else {
                if (NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(pointDst.getDevicePointer(), pointSrc.getHostPointer(), this.lengthLong() * Nd4j.sizeOfDataType(buffer.dataType()), CudaConstants.cudaMemcpyHostToDevice, context.getOldStream()) == 0)
                    throw new ND4JIllegalStateException("memcpyAsync failed");

                direction = MemcpyDirection.HOST_TO_DEVICE;
            }

            context.syncOldStream();

            PerformanceTracker.getInstance().helperRegisterTransaction(pointDst.getDeviceId(), perfD, pointDst.getNumberOfBytes(), direction);

            if (pointDst.getDeviceId() != Nd4j.getMemoryManager().getCurrentWorkspace().getDeviceId()) {
                //log.info("Swapping [{}] -> [{}]", pointDst.getDeviceId(), Nd4j.getMemoryManager().getCurrentWorkspace().getDeviceId());
                pointDst.setDeviceId(Nd4j.getMemoryManager().getCurrentWorkspace().getDeviceId());
            }

            copy = Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInfoDataBuffer());

            // tag buffer as valid on device side
            pointDst.tickHostRead();
            pointDst.tickDeviceWrite();

            AtomicAllocator.getInstance().getFlowController().registerAction(context, pointDst, pointSrc);
        } else {
            copy = this.dup(this.ordering());
        }

        return copy;
    }

    @Override
    public INDArray convertToHalfs() {
        if (data.dataType() == DataBuffer.Type.HALF)
            return this;

        val factory = Nd4j.getNDArrayFactory();
        val buffer = Nd4j.createBuffer(new long[]{this.length()}, DataBuffer.Type.HALF);

        factory.convertDataEx(convertType(data.dataType()), AtomicAllocator.getInstance().getPointer(this.data()),
                DataBuffer.TypeEx.FLOAT16, AtomicAllocator.getInstance().getPointer(buffer), buffer.length());

        AtomicAllocator.getInstance().getAllocationPoint(buffer).tickDeviceWrite();

        return Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInformation);
    }


    @Override
    public INDArray convertToFloats() {
        if (data.dataType() == DataBuffer.Type.FLOAT)
            return this;

        val factory = Nd4j.getNDArrayFactory();
        val buffer = Nd4j.createBuffer(new long[]{this.length()}, DataBuffer.Type.FLOAT);

        factory.convertDataEx(convertType(data.dataType()), AtomicAllocator.getInstance().getPointer(this.data()), DataBuffer.TypeEx.FLOAT, AtomicAllocator.getInstance().getPointer(buffer), buffer.length());

        AtomicAllocator.getInstance().getAllocationPoint(buffer).tickHostWrite();

        return Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInformation);
    }

    @Override
    public INDArray convertToDoubles() {
        if (data.dataType() == DataBuffer.Type.DOUBLE)
            return this;

        val factory = Nd4j.getNDArrayFactory();
        val buffer = Nd4j.createBuffer(new long[]{this.length()}, DataBuffer.Type.DOUBLE);

        factory.convertDataEx(convertType(data.dataType()), AtomicAllocator.getInstance().getPointer(this.data()), DataBuffer.TypeEx.DOUBLE, AtomicAllocator.getInstance().getPointer(buffer), buffer.length());

        AtomicAllocator.getInstance().getAllocationPoint(buffer).tickHostWrite();

        return Nd4j.createArrayFromShapeBuffer(buffer, this.shapeInformation);
    }


}
