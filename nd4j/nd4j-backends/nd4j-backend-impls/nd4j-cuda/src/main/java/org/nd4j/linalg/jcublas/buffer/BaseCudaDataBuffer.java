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

import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.impl.CudaDeallocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.pointers.PointersPair;
import org.nd4j.jita.workspace.CudaWorkspaceDeallocator;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.memory.MemcpyDirection;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LongUtils;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;

import java.io.*;
import java.nio.*;
import java.util.Collection;

/**
 * Base class for a data buffer
 *
 * CUDA implementation for DataBuffer always uses JavaCPP
 * as allocationMode, and device access is masked by
 * appropriate allocator mover implementation.
 *
 * Memory allocation/deallocation is strictly handled by allocator,
 * since JavaCPP alloc/dealloc has nothing to do with CUDA.
 * But besides that, host pointers obtained from CUDA are 100%
 * compatible with CPU
 *
 * @author Adam Gibson
 * @author raver119@gmail.com
 */
public abstract class BaseCudaDataBuffer extends BaseDataBuffer implements JCudaBuffer, Deallocatable {

    @Getter
    protected transient volatile AllocationPoint allocationPoint;

    private static AtomicAllocator allocator = AtomicAllocator.getInstance();

    private static Logger log = LoggerFactory.getLogger(BaseCudaDataBuffer.class);

    protected DataType globalType = DataTypeUtil.getDtypeFromContext();

    public BaseCudaDataBuffer() {

    }

    public BaseCudaDataBuffer(@NonNull Pointer pointer, @NonNull Pointer specialPointer, @NonNull Indexer indexer, long length) {
        this.allocationPoint = AtomicAllocator.getInstance().pickExternalBuffer(this);
        this.allocationPoint.setPointers(new PointersPair(specialPointer, pointer));
        this.trackingPoint = allocationPoint.getObjectId();
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;

        this.indexer = indexer;

        this.offset = 0;
        this.originalOffset = 0;
        this.underlyingLength = length;
        this.length = length;

        initTypeAndSize();
    }

    /**
     * Meant for creating another view of a buffer
     *
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length  the length of the view
     */
    public BaseCudaDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);

        //cuda specific bits
        this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this, new AllocationShape(length, elementSize, dataType()), false);

        Nd4j.getDeallocatorService().pickObject(this);

        // now we're
        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        val perfD = PerformanceTracker.getInstance().helperStartTransaction();

        NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(allocationPoint.getHostPointer(), pointer, length * getElementSize(), CudaConstants.cudaMemcpyHostToHost, context.getSpecialStream());
        NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(allocationPoint.getDevicePointer(), allocationPoint.getHostPointer(), length * getElementSize(), CudaConstants.cudaMemcpyHostToHost, context.getSpecialStream());

        context.getSpecialStream().synchronize();

        PerformanceTracker.getInstance().helperRegisterTransaction(allocationPoint.getDeviceId(), perfD / 2, allocationPoint.getNumberOfBytes(), MemcpyDirection.HOST_TO_HOST);
        PerformanceTracker.getInstance().helperRegisterTransaction(allocationPoint.getDeviceId(), perfD / 2, allocationPoint.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);

        this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length * getElementSize(), 0);

        switch (dataType()) {
            case INT: {
                setIndexer(IntIndexer.create(((CudaPointer) this.pointer).asIntPointer()));
            }
            break;
            case FLOAT: {
                setIndexer(FloatIndexer.create(((CudaPointer) this.pointer).asFloatPointer()));
            }
            break;
            case DOUBLE: {
                setIndexer(DoubleIndexer.create(((CudaPointer) this.pointer).asDoublePointer()));
            }
            break;
            case HALF: {
                setIndexer(ShortIndexer.create(((CudaPointer) this.pointer).asShortPointer()));
            }
            break;
            case LONG: {
                setIndexer(LongIndexer.create(((CudaPointer) this.pointer).asLongPointer()));
            }
            break;
        }

        this.trackingPoint = allocationPoint.getObjectId();

    }

    public BaseCudaDataBuffer(float[] data, boolean copy) {
        //super(data, copy);
        this(data, copy, 0);
    }

    public BaseCudaDataBuffer(float[] data, boolean copy, MemoryWorkspace workspace) {
        //super(data, copy);
        this(data, copy, 0, workspace);
    }

    public BaseCudaDataBuffer(float[] data, boolean copy, long offset) {
        this(data.length, 4, false);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
        set(data, this.length, offset, offset);
    }

    public BaseCudaDataBuffer(double[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        this(data.length, 8, false, workspace);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
        set(data, this.length, offset, offset);
    }

    public BaseCudaDataBuffer(float[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        this(data.length, 4,false, workspace);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
        set(data, this.length, offset, offset);
    }

    public BaseCudaDataBuffer(double[] data, boolean copy) {
        this(data, copy, 0);
    }

    public BaseCudaDataBuffer(double[] data, boolean copy, long offset) {
        this(data.length, 8, false);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
        set(data, this.length, offset, offset);
    }

    public BaseCudaDataBuffer(int[] data, boolean copy) {
        this(data, copy, 0);
    }

    public BaseCudaDataBuffer(int[] data, boolean copy, MemoryWorkspace workspace) {
        this(data, copy, 0, workspace);
    }

    public BaseCudaDataBuffer(int[] data, boolean copy, long offset) {
        this(data.length, 4, false);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
        set(data, this.length, offset, offset);
    }

    public BaseCudaDataBuffer(int[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        this(data.length, 4, false, workspace);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
        set(data, this.length, offset, offset);
    }

    protected void initPointers(long length, DataType dtype, boolean initialize) {
        initPointers(length, Nd4j.sizeOfDataType(dtype), initialize);
    }

    protected void lazyAllocateHostPointer() {
        if (allocationPoint.getPointers().getHostPointer() == null)
            initHostPointerAndIndexer();
    }

    protected void initHostPointerAndIndexer() {
        if (allocationPoint.getPointers().getHostPointer() == null) {
            val location = allocationPoint.getAllocationStatus();
            val ptr = AtomicAllocator.getInstance().getMemoryHandler().alloc(AllocationStatus.HOST, this.allocationPoint, this.allocationPoint.getShape(), false);
            this.allocationPoint.getPointers().setHostPointer(ptr.getHostPointer());
            this.allocationPoint.setAllocationStatus(location);
            this.allocationPoint.tickDeviceWrite();
        }

        switch (dataType()) {
            case DOUBLE:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asDoublePointer();
                indexer = DoubleIndexer.create((DoublePointer) pointer);
                break;
            case FLOAT:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asFloatPointer();
                indexer = FloatIndexer.create((FloatPointer) pointer);
                break;
            case UINT32:
            case INT:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asIntPointer();
                indexer = IntIndexer.create((IntPointer) pointer);
                break;
            case BFLOAT16:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                break;
            case HALF:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                indexer = HalfIndexer.create((ShortPointer) pointer);
                break;
            case UINT64:
            case LONG:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asLongPointer();
                indexer = LongIndexer.create((LongPointer) pointer);
                break;
            case UINT16:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                indexer = UShortIndexer.create((ShortPointer) pointer);
                break;
            case SHORT:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                indexer = ShortIndexer.create((ShortPointer) pointer);
                break;
            case UBYTE:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asBytePointer();
                indexer = UByteIndexer.create((BytePointer) pointer);
                break;
            case BYTE:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asBytePointer();
                indexer = ByteIndexer.create((BytePointer) pointer);
                break;
            case BOOL:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asBooleanPointer();
                indexer = BooleanIndexer.create((BooleanPointer) pointer);
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    protected void initPointers(long length, int elementSize, boolean initialize) {
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
        this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this, new AllocationShape(length, elementSize, dataType()), initialize);
        this.length = length;
        //allocationPoint.attachBuffer(this);
        this.elementSize =  (byte) elementSize;
        this.trackingPoint = allocationPoint.getObjectId();
        this.offset = 0;
        this.originalOffset = 0;

        Nd4j.getDeallocatorService().pickObject(this);

        // if only host
        if (allocationPoint.getPointers().getHostPointer() == null)
            return;

        initHostPointerAndIndexer();
    }

    public BaseCudaDataBuffer(long length, int elementSize, boolean initialize) {
        initTypeAndSize();
        initPointers(length, elementSize, initialize);
    }

    public BaseCudaDataBuffer(long length, int elementSize, boolean initialize, @NonNull MemoryWorkspace workspace) {
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
        initTypeAndSize();

        this.attached = true;
        this.parentWorkspace = workspace;

        this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this, new AllocationShape(length, this.elementSize, dataType()), initialize);
        this.length = length;

        this.trackingPoint = allocationPoint.getObjectId();
        this.offset = 0;
        this.originalOffset = 0;

        Nd4j.getDeallocatorService().pickObject(this);

        switch (dataType()) {
            case DOUBLE:
                this.attached = true;
                this.parentWorkspace = workspace;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asDoublePointer();
                indexer = DoubleIndexer.create((DoublePointer) pointer);
                break;
            case FLOAT:
                this.attached = true;
                this.parentWorkspace = workspace;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asFloatPointer();
                indexer = FloatIndexer.create((FloatPointer) pointer);
                break;
            case UINT32:
            case INT:
                this.attached = true;
                this.parentWorkspace = workspace;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asIntPointer();
                indexer = IntIndexer.create((IntPointer) pointer);
                break;
            case BFLOAT16:
                this.attached = true;
                this.parentWorkspace = workspace;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                break;
            case HALF:
                this.attached = true;
                this.parentWorkspace = workspace;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                indexer = HalfIndexer.create((ShortPointer) pointer);
                break;
            case UINT64:
            case LONG:
                this.attached = true;
                this.parentWorkspace = workspace;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asLongPointer();
                indexer = LongIndexer.create((LongPointer) pointer);
                break;
            case BOOL:
                this.attached = true;
                this.parentWorkspace = workspace;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asBooleanPointer();
                indexer = BooleanIndexer.create((BooleanPointer) pointer);
                break;
            case UINT16:
                this.attached = true;
                this.parentWorkspace = workspace;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                indexer = UShortIndexer.create((ShortPointer) pointer);
                break;
            case SHORT:
                this.attached = true;
                this.parentWorkspace = workspace;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                indexer = ShortIndexer.create((ShortPointer) pointer);
                break;
            case BYTE:
                this.attached = true;
                this.parentWorkspace = workspace;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asBytePointer();
                indexer = ByteIndexer.create((BytePointer) pointer);
                break;
            case UBYTE:
                this.attached = true;
                this.parentWorkspace = workspace;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asBytePointer();
                indexer = UByteIndexer.create((BytePointer) pointer);
                break;
            default:
                throw new UnsupportedOperationException("Unknown data type: " + dataType());
        }

        workspaceGenerationId = workspace.getGenerationId();
    }

    @Override
    protected void setIndexer(Indexer indexer) {
        //TODO: to be abstracted
        this.indexer = indexer;
    }

    /**
     * Base constructor. It's used within all constructors internally
     *
     * @param length      the length of the buffer
     * @param elementSize the size of each element
     */
    public BaseCudaDataBuffer(long length, int elementSize) {
        this(length, elementSize, true);
    }

    public BaseCudaDataBuffer(long length, int elementSize, MemoryWorkspace workspace) {
        this(length, elementSize, true, workspace);
    }

    public BaseCudaDataBuffer(long length, int elementSize, long offset) {
        this(length, elementSize);
        this.offset = offset;
        this.originalOffset = offset;
    }

    public BaseCudaDataBuffer(@NonNull DataBuffer underlyingBuffer, long length, long offset) {
        //this(length, underlyingBuffer.getElementSize(), offset);
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
        initTypeAndSize();
        this.wrappedDataBuffer = underlyingBuffer;
        this.originalBuffer = underlyingBuffer.originalDataBuffer() == null ? underlyingBuffer
                        : underlyingBuffer.originalDataBuffer();
        this.length = length;
        this.offset = offset;
        this.originalOffset = offset;
        this.trackingPoint = underlyingBuffer.getTrackingPoint();
        this.elementSize = (byte) underlyingBuffer.getElementSize();
        this.allocationPoint = ((BaseCudaDataBuffer) underlyingBuffer).allocationPoint;

        // in case of view creation, we initialize underlying buffer regardless of anything
        ((BaseCudaDataBuffer) underlyingBuffer).lazyAllocateHostPointer();;

        switch (underlyingBuffer.dataType()) {
            case DOUBLE:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asDoublePointer();
                indexer = DoubleIndexer.create((DoublePointer) pointer);
                break;
            case FLOAT:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asFloatPointer();
                indexer = FloatIndexer.create((FloatPointer) pointer);
                break;
            case UINT32:
            case INT:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asIntPointer();
                indexer = IntIndexer.create((IntPointer) pointer);
                break;
            case BFLOAT16:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asShortPointer();
                indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                break;
            case HALF:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asShortPointer();
                indexer = HalfIndexer.create((ShortPointer) pointer);
                break;
            case UINT64:
            case LONG:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asLongPointer();
                indexer = LongIndexer.create((LongPointer) pointer);
                break;
            case UINT16:
            case SHORT:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asShortPointer();
                indexer = ShortIndexer.create((ShortPointer) pointer);
                break;
            case BOOL:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asBooleanPointer();
                indexer = BooleanIndexer.create((BooleanPointer) pointer);
                break;
            case BYTE:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asBytePointer();
                indexer = ByteIndexer.create((BytePointer) pointer);
                break;
            case UBYTE:
                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asBytePointer();
                indexer = UByteIndexer.create((BytePointer) pointer);
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    public BaseCudaDataBuffer(long length) {
        this(length, Nd4j.sizeOfDataType(Nd4j.dataType()));
    }

    public BaseCudaDataBuffer(float[] data) {
        //super(data);
        this(data.length, Nd4j.sizeOfDataType(DataType.FLOAT), false);
        set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(int[] data) {
        //super(data);
        this(data.length, Nd4j.sizeOfDataType(DataType.INT), false);
        set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(long[] data) {
        //super(data);
        this(data.length, Nd4j.sizeOfDataType(DataType.LONG), false);
        set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(long[] data, boolean copy) {
        //super(data);
        this(data.length, Nd4j.sizeOfDataType(DataType.LONG), false);

        if (copy)
            set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(double[] data) {
        // super(data);
        this(data.length, Nd4j.sizeOfDataType(DataType.DOUBLE), false);
        set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(byte[] data, long length, DataType type) {
        this(ByteBuffer.wrap(data), length, type);
    }

    public BaseCudaDataBuffer(ByteBuffer buffer, long length, DataType type) {
        //super(buffer,length);
        this(buffer, length, 0, type);
    }

    public BaseCudaDataBuffer(ByteBuffer buffer, long length, long offset, DataType type) {
        //super(buffer, length, offset);
        this(length, Nd4j.sizeOfDataType(type), offset);

        Pointer srcPtr = new CudaPointer(new Pointer(buffer.order(ByteOrder.nativeOrder())));

        allocator.memcpyAsync(this, srcPtr, length * elementSize, offset * elementSize);
    }

    /**
     * This method always returns host pointer
     *
     * @return
     */
    @Override
    public long address() {
        return allocationPoint.getPointers().getHostPointer().address();
    }


    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(int[] data, long length, long srcOffset, long dstOffset) {
        // TODO: make sure getPointer returns proper pointer

        switch (dataType()) {
            case BOOL: {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case BYTE: {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case UBYTE: {
                    for (int e = 0; e < data.length; e++) {
                        put(e, data[e]);
                    }
                }
                break;
            case SHORT: {
                    val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case INT: {
                    val pointer = new IntPointer(data);
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case LONG: {
                    val pointer = new LongPointer(LongUtils.toLongs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case HALF: {
                    val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case FLOAT: {
                    val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case DOUBLE: {
                    val pointer = new DoublePointer(ArrayUtil.toDouble(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }


    public void set(long[] data, long length, long srcOffset, long dstOffset) {
        // TODO: make sure getPointer returns proper pointer

        switch (dataType()) {
            case BOOL: {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case BYTE: {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case UBYTE: {
                    for (int e = 0; e < data.length; e++) {
                        put(e, data[e]);
                    }
                }
                break;
            case SHORT: {
                    val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case UINT32:
            case INT: {
                    val pointer = new IntPointer(ArrayUtil.toInts(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case UINT64:
            case LONG: {
                    val pointer = new LongPointer(data);
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case BFLOAT16: {
                val pointer = new ShortPointer(ArrayUtil.toBfloats(data));
                val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                // we're keeping pointer reference for JVM
                pointer.address();
            }
            break;
            case HALF: {
                    val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case FLOAT: {
                    val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case DOUBLE: {
                    val pointer = new DoublePointer(ArrayUtil.toDouble(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);
                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }

    }

    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(float[] data, long length, long srcOffset, long dstOffset) {
        switch (dataType()) {
            case BOOL: {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case BYTE: {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case UBYTE: {
                    for (int e = 0; e < data.length; e++) {
                        put(e, data[e]);
                    }
                }
                break;
            case SHORT: {
                    val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case INT: {
                    val pointer = new IntPointer(ArrayUtil.toInts(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case LONG: {
                    val pointer = new LongPointer(ArrayUtil.toLongArray(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case HALF: {
                    val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case FLOAT: {
                    val pointer = new FloatPointer(data);
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case DOUBLE: {
                    DoublePointer pointer = new DoublePointer(ArrayUtil.toDoubles(data));
                    Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }

    /**
     *
     * PLEASE NOTE: length, srcOffset, dstOffset are considered numbers of elements, not byte offsets
     *
     * @param data
     * @param length
     * @param srcOffset
     * @param dstOffset
     */
    public void set(double[] data, long length, long srcOffset, long dstOffset) {
        switch (dataType()) {
            case BOOL:  {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case BYTE: {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case UBYTE: {
                    for (int e = 0; e < data.length; e++) {
                        put(e, data[e]);
                    }
                }
                break;
            case SHORT: {
                    val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case INT: {
                    val pointer = new IntPointer(ArrayUtil.toInts(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case LONG: {
                    val pointer = new LongPointer(ArrayUtil.toLongs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case HALF: {
                    val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case FLOAT: {
                    val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case DOUBLE: {
                    val pointer = new DoublePointer(data);
                    val srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }

    @Override
    public void setData(int[] data) {
        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(long[] data) {
        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(float[] data) {
        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(double[] data) {
        set(data, data.length, 0, 0);
    }

    @Override
    protected void setNioBuffer() {
        throw new UnsupportedOperationException("setNioBuffer() is not supported for CUDA backend");
    }

    @Override
    public void copyAtStride(DataBuffer buf, long n, long stride, long yStride, long offset, long yOffset) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        allocator.synchronizeHostData(buf);
        super.copyAtStride(buf, n, stride, yStride, offset, yOffset);
    }

    @Override
    public AllocationMode allocationMode() {
        return allocationMode;
    }

    @Override
    public ByteBuffer getHostBuffer() {
        return pointer.asByteBuffer();
    }

    @Override
    public Pointer getHostPointer() {
        return AtomicAllocator.getInstance().getHostPointer(this);
    }

    @Override
    public Pointer getHostPointer(long offset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void removeReferencing(String id) {
        //referencing.remove(id);
    }

    @Override
    public Collection<String> references() {
        //return referencing;
        return null;
    }

    @Override
    public int getElementSize() {
        return elementSize;
    }


    @Override
    public void addReferencing(String id) {
        //referencing.add(id);
    }


    @Deprecated
    public Pointer getHostPointer(INDArray arr, int stride, long offset, int length) {
        throw new UnsupportedOperationException("This method is deprecated");
    }

    @Deprecated
    public void set(Pointer pointer) {
        throw new UnsupportedOperationException("set(Pointer) is not supported");
    }

    @Override
    public void put(long i, float element) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        allocator.tickHostWrite(this);
        super.put(i, element);
    }

    @Override
    public void put(long i, boolean element) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        allocator.tickHostWrite(this);
        super.put(i, element);
    }

    @Override
    public void put(long i, double element) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        allocator.tickHostWrite(this);
        super.put(i, element);
    }

    @Override
    public void put(long i, int element) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        allocator.tickHostWrite(this);
        super.put(i, element);
    }

    @Override
    public void put(long i, long element) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        allocator.tickHostWrite(this);
        super.put(i, element);
    }

    @Override
    public Pointer addressPointer() {
        return AtomicAllocator.getInstance().getHostPointer(this);
    }

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    @Deprecated
    protected void set(long index, long length, Pointer from, long inc) {


        long offset = getElementSize() * index;
        if (offset >= length() * getElementSize())
            throw new IllegalArgumentException(
                            "Illegal offset " + offset + " with index of " + index + " and length " + length());

        // TODO: fix this
        throw new UnsupportedOperationException("Deprecated set() call");
    }

    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    @Deprecated
    protected void set(long index, long length, Pointer from) {
        set(index, length, from, 1);
    }

    @Override
    public void assign(DataBuffer data) {
        /*JCudaBuffer buf = (JCudaBuffer) data;
        set(0, buf.getHostPointer());
        */
        /*
        memcpyAsync(
                new Pointer(allocator.getPointer(this).address()),
                new Pointer(allocator.getPointer(data).address()),
                data.length()
        );*/
        allocator.memcpy(this, data);
    }

    @Override
    public void assign(long[] indices, float[] data, boolean contiguous, long inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length "
                    + length() + " where the indices are of length " + data.length);

        // TODO: eventually consider memcpy here
        for (int i = 0; i < indices.length; i++) {
            put(indices[i], data[i]);
        }
    }

    @Override
    public void assign(long[] indices, double[] data, boolean contiguous, long inc) {

        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length "
                    + length() + " where the indices are of length " + data.length);

        // TODO: eventually consider memcpy here
        for (int i = 0; i < indices.length; i++) {
            put(indices[i], data[i]);
        }
    }


    /**
     * Set an individual element
     *
     * @param index the index of the element
     * @param from  the element to get data from
     */
    @Deprecated
    protected void set(long index, Pointer from) {
        set(index, 1, from);
    }

    @Override
    public void flush() {
        //
    }


    @Override
    public void destroy() {}

    @Override
    public void write(DataOutputStream out) throws IOException {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        super.write(out);
    }

    @Override
    public void write(OutputStream dos) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        super.write(dos);
    }

    private void writeObject(java.io.ObjectOutputStream stream) throws IOException {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        stream.defaultWriteObject();
        write(stream);
    }

    private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException {
        doReadObject(stream);
    }

    @Override
    public String toString() {
        //lazyAllocateHostPointer();
        //AtomicAllocator.getInstance().synchronizeHostData(this);
        //return super.toString();
        return "-119";
    }

    @Override
    public boolean sameUnderlyingData(DataBuffer buffer) {
        return buffer.getTrackingPoint() == getTrackingPoint();
    }

    /**
     * PLEASE NOTE: this method implies STRICT equality only.
     * I.e: this == object
     *
     * @param o
     * @return
     */
    @Override
    public boolean equals(Object o) {
        if (o == null)
            return false;
        if (this == o)
            return true;

        return false;
    }

    @Override
    public void read(InputStream is, AllocationMode allocationMode, long length, DataType dataType) {
        if (allocationPoint == null) {
            initPointers(length, dataType, false);
        }
        super.read(is, allocationMode, length, dataType);
        this.allocationPoint.tickHostWrite();
    }

    @Override
    public void pointerIndexerByCurrentType(DataType currentType) {
        //
        /*
        switch (currentType) {
            case LONG:
                pointer = new LongPointer(length());
                setIndexer(LongIndexer.create((LongPointer) pointer));
                type = DataType.LONG;
                break;
            case INT:
                pointer = new IntPointer(length());
                setIndexer(IntIndexer.create((IntPointer) pointer));
                type = DataType.INT;
                break;
            case DOUBLE:
                pointer = new DoublePointer(length());
                indexer = DoubleIndexer.create((DoublePointer) pointer);
                break;
            case FLOAT:
                pointer = new FloatPointer(length());
                setIndexer(FloatIndexer.create((FloatPointer) pointer));
                break;
            case HALF:
                pointer = new ShortPointer(length());
                setIndexer(HalfIndexer.create((ShortPointer) pointer));
                break;
            case COMPRESSED:
                break;
            default:
                throw new UnsupportedOperationException();
        }
        */
    }

    //@Override
    public void read(DataInputStream s) {
        try {
            val savedMode = AllocationMode.valueOf(s.readUTF());
            allocationMode = AllocationMode.MIXED_DATA_TYPES;

            long locLength = 0;

            if (savedMode.ordinal() < 3)
                locLength = s.readInt();
            else
                locLength = s.readLong();

            boolean reallocate = locLength != length || indexer == null;
            length = locLength;

            val t = DataType.valueOf(s.readUTF());
            //                  log.info("Restoring buffer ["+t+"] of length ["+ length+"]");
            if (globalType == null && Nd4j.dataType() != null) {
                globalType = Nd4j.dataType();
            }

            if (t == DataType.COMPRESSED) {
                type = t;
                return;
            }

            this.elementSize = (byte) Nd4j.sizeOfDataType(t);
            this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this, new AllocationShape(length, elementSize, t), false);
            this.trackingPoint = allocationPoint.getObjectId();
            this.type = t;

            Nd4j.getDeallocatorService().pickObject(this);

            switch (type) {
                case DOUBLE: {
                        this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asDoublePointer();
                        indexer = DoubleIndexer.create((DoublePointer) pointer);
                    }
                    break;
                case FLOAT: {
                        this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asFloatPointer();
                        indexer = FloatIndexer.create((FloatPointer) pointer);
                    }
                    break;
                case HALF: {
                        this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asShortPointer();
                        indexer = HalfIndexer.create((ShortPointer) pointer);
                    }
                    break;
                case LONG: {
                        this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asLongPointer();
                        indexer = LongIndexer.create((LongPointer) pointer);
                    }
                    break;
                case INT: {
                        this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asIntPointer();
                        indexer = IntIndexer.create((IntPointer) pointer);
                    }
                    break;
                case SHORT: {
                        this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asShortPointer();
                        indexer = ShortIndexer.create((ShortPointer) pointer);
                    }
                    break;
                case UBYTE: {
                        this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asBytePointer();
                        indexer = UByteIndexer.create((BytePointer) pointer);
                    }
                    break;
                case BYTE: {
                        this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asBytePointer();
                        indexer = ByteIndexer.create((BytePointer) pointer);
                    }
                    break;
                case BOOL: {
                        this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asBooleanPointer();
                        indexer = BooleanIndexer.create((BooleanPointer) pointer);
                    }
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported data type: " + type);
            }

            readContent(s, t, t);
            allocationPoint.tickHostWrite();

        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        // we call sync to copyback data to host
        AtomicAllocator.getInstance().getFlowController().synchronizeToDevice(allocationPoint);
        //allocator.synchronizeHostData(this);
    }

    @Override
    public byte[] asBytes() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asBytes();
    }

    @Override
    public double[] asDouble() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asDouble();
    }

    @Override
    public float[] asFloat() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asFloat();
    }

    @Override
    public int[] asInt() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asInt();
    }

    @Override
    public ByteBuffer asNio() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asNio();
    }

    @Override
    public DoubleBuffer asNioDouble() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asNioDouble();
    }

    @Override
    public FloatBuffer asNioFloat() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asNioFloat();
    }

    @Override
    public IntBuffer asNioInt() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asNioInt();
    }

    @Override
    public DataBuffer dup() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        DataBuffer buffer = create(this.length);
        allocator.memcpyBlocking(buffer, new CudaPointer(allocator.getHostPointer(this).address()), this.length * elementSize, 0);
        return buffer;
    }

    @Override
    public Number getNumber(long i) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.getNumber(i);
    }

    @Override
    public double getDouble(long i) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.getDouble(i);
    }

    @Override
    public long getLong(long i) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.getLong(i);
    }


    @Override
    public float getFloat(long i) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.getFloat(i);
    }

    @Override
    public int getInt(long ix) {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.getInt(ix);
    }

    @Override
    public DataBuffer reallocate(long length) {

        // we want to be sure this array isn't used anywhere RIGHT AT THIS MOMENT
        Nd4j.getExecutioner().commit();


            AllocationPoint old = allocationPoint;
            allocationPoint = AtomicAllocator.getInstance().allocateMemory(this, new AllocationShape(length, elementSize, dataType()), false);

            Nd4j.getDeallocatorService().pickObject(this);
            trackingPoint = allocationPoint.getObjectId();

            switch(dataType()){
                case DOUBLE:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asDoublePointer();
                    indexer = DoubleIndexer.create((DoublePointer) pointer);
                    break;
                case FLOAT:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asFloatPointer();
                    indexer = FloatIndexer.create((FloatPointer) pointer);
                    break;
                case BFLOAT16:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                    indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                    break;
                case HALF:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                    indexer = ShortIndexer.create((ShortPointer) pointer);
                    break;
                case LONG:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asLongPointer();
                    indexer = LongIndexer.create((LongPointer) pointer);
                    break;
                case UINT64:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asLongPointer();
                    indexer = LongIndexer.create((LongPointer) pointer);
                    break;
                case INT:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asIntPointer();
                    indexer = IntIndexer.create((IntPointer) pointer);
                    break;
                case UINT32:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asIntPointer();
                    indexer = IntIndexer.create((IntPointer) pointer);
                    break;
                case SHORT:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                    indexer = ShortIndexer.create((ShortPointer) pointer);
                    break;
                case UINT16:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                    indexer = UShortIndexer.create((ShortPointer) pointer);
                    break;
                case BYTE:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asBytePointer();
                    indexer = ByteIndexer.create((BytePointer) pointer);
                    break;
                default:
                    throw new UnsupportedOperationException();
            }

            CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();
            NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(allocationPoint.getDevicePointer(), 0, length * elementSize, 0, context.getSpecialStream());

            MemcpyDirection direction = MemcpyDirection.DEVICE_TO_DEVICE;
            val perfD = PerformanceTracker.getInstance().helperStartTransaction();

            if (old.isActualOnDeviceSide()) {
                NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(allocationPoint.getDevicePointer(), old.getDevicePointer(), this.length * elementSize, CudaConstants.cudaMemcpyDeviceToDevice, context.getSpecialStream());
            } else if (old.isActualOnHostSide()) {
                NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(allocationPoint.getDevicePointer(), old.getHostPointer(), this.length * elementSize, CudaConstants.cudaMemcpyHostToDevice, context.getSpecialStream());
                direction = MemcpyDirection.HOST_TO_DEVICE;
            }

            context.getSpecialStream().synchronize();

            PerformanceTracker.getInstance().helperRegisterTransaction(allocationPoint.getDeviceId(), perfD, allocationPoint.getNumberOfBytes(), direction);

            allocationPoint.tickDeviceWrite();
            // we're keeping pointer reference for JVM
            pointer.address();


            // we need to update length with new value now
            //this.length = length;
        if(isAttached()){
            // do nothing here, that's workspaces
        } else{
            AtomicAllocator.getInstance().freeMemory(old);
        }

        return this;
    }

    @Override
    public long capacity() {
        return pointer.capacity();
    }

    @Override
    protected void release() {
        AtomicAllocator.getInstance().freeMemory(allocationPoint);
        released = true;
    }

    /*
    protected short fromFloat( float fval ) {
        int fbits = Float.floatToIntBits( fval );
        int sign = fbits >>> 16 & 0x8000;          // sign only
        int val = ( fbits & 0x7fffffff ) + 0x1000; // rounded value
    
        if( val >= 0x47800000 )               // might be or become NaN/Inf
        {                                     // avoid Inf due to rounding
            if( ( fbits & 0x7fffffff ) >= 0x47800000 )
            {                                 // is or must become NaN/Inf
                if( val < 0x7f800000 )        // was value but too large
                    return (short) (sign | 0x7c00);     // make it +/-Inf
                return (short) (sign | 0x7c00 |        // remains +/-Inf or NaN
                        ( fbits & 0x007fffff ) >>> 13); // keep NaN (and Inf) bits
            }
            return (short) (sign | 0x7bff);             // unrounded not quite Inf
        }
        if( val >= 0x38800000 )               // remains normalized value
            return (short) (sign | val - 0x38000000 >>> 13); // exp - 127 + 15
        if( val < 0x33000000 )                // too small for subnormal
            return (short) sign;                      // becomes +/-0
        val = ( fbits & 0x7fffffff ) >>> 23;  // tmp exp for subnormal calc
        return (short) (sign | ( ( fbits & 0x7fffff | 0x800000 ) // add subnormal bit
                + ( 0x800000 >>> val - 102 )     // round depending on cut off
                >>> 126 - val ));   // div by 2^(1-(exp-127+15)) and >> 13 | exp=0
    }
    */

    @Override
    public String getUniqueId() {
        return "BCDB_" + allocationPoint.getObjectId();
    }

    /**
     * This method returns deallocator associated with this instance
     * @return
     */
    @Override
    public Deallocator deallocator() {
        return new CudaDeallocator(this);
    }

    @Override
    public int targetDevice() {
        return AtomicAllocator.getInstance().getAllocationPoint(this).getDeviceId();
    }
}
