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
import org.nd4j.base.Preconditions;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.impl.CudaDeallocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.MemcpyDirection;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LongUtils;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    protected OpaqueDataBuffer ptrDataBuffer;

    @Getter
    protected transient volatile AllocationPoint allocationPoint;

    private static AtomicAllocator allocator = AtomicAllocator.getInstance();

    private static Logger log = LoggerFactory.getLogger(BaseCudaDataBuffer.class);

    protected DataType globalType = DataTypeUtil.getDtypeFromContext();

    public BaseCudaDataBuffer() {

    }

    public OpaqueDataBuffer getOpaqueDataBuffer() {
        return ptrDataBuffer;
    }


    public BaseCudaDataBuffer(@NonNull Pointer pointer, @NonNull Pointer specialPointer, @NonNull Indexer indexer, long length) {
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;

        this.indexer = indexer;

        this.offset = 0;
        this.originalOffset = 0;
        this.underlyingLength = length;
        this.length = length;

        initTypeAndSize();

        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(0, this.type, false);
        this.allocationPoint = new AllocationPoint(ptrDataBuffer, this.type.width() * length);
        this.allocationPoint.setPointers(pointer, specialPointer, length);

        Nd4j.getDeallocatorService().pickObject(this);
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

        // allocating interop buffer
        this.ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length, type, false);

        // passing existing pointer to native holder
        this.ptrDataBuffer.setPrimaryBuffer(pointer, length);

        //cuda specific bits
        this.allocationPoint = new AllocationPoint(ptrDataBuffer, length * elementSize);
        Nd4j.getDeallocatorService().pickObject(this);

        // now we're getting context and copying our stuff to device
        val context = AtomicAllocator.getInstance().getDeviceContext();

        val perfD = PerformanceTracker.getInstance().helperStartTransaction();

        NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(allocationPoint.getDevicePointer(), pointer, length * getElementSize(), CudaConstants.cudaMemcpyHostToDevice, context.getSpecialStream());

        PerformanceTracker.getInstance().helperRegisterTransaction(allocationPoint.getDeviceId(), perfD / 2, allocationPoint.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);
        context.getSpecialStream().synchronize();
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

    public void lazyAllocateHostPointer() {
        if (length() == 0)
            return;

        // java side might be unaware of native-side buffer allocation
        if (this.indexer == null || this.pointer == null || this.pointer.address() == 0) {
            initHostPointerAndIndexer();
        } else if (allocationPoint.getHostPointer() != null && allocationPoint.getHostPointer().address() != this.pointer.address()) {
            initHostPointerAndIndexer();
        }
    }

    protected BaseCudaDataBuffer(ByteBuffer buffer, DataType dtype, long length, long offset) {
        this(length, Nd4j.sizeOfDataType(dtype));

        Pointer temp = null;

        switch (dataType()){
            case DOUBLE:
                temp = new DoublePointer(buffer.asDoubleBuffer());
                break;
            case FLOAT:
                temp = new FloatPointer(buffer.asFloatBuffer());
                break;
            case HALF:
                temp = new ShortPointer(buffer.asShortBuffer());
                break;
            case LONG:
                temp = new LongPointer(buffer.asLongBuffer());
                break;
            case INT:
                temp = new IntPointer(buffer.asIntBuffer());
                break;
            case SHORT:
                temp = new ShortPointer(buffer.asShortBuffer());
                break;
            case UBYTE: //Fall through
            case BYTE:
                temp = new BytePointer(buffer);
                break;
            case BOOL:
                temp = new BooleanPointer(length());
                break;
            case UTF8:
                temp = new BytePointer(length());
                break;
            case BFLOAT16:
                temp = new ShortPointer(length());
                break;
            case UINT16:
                temp = new ShortPointer(length());
                break;
            case UINT32:
                temp = new IntPointer(length());
                break;
            case UINT64:
                temp = new LongPointer(length());
                break;
        }

        // copy data to device
        val stream = AtomicAllocator.getInstance().getDeviceContext().getSpecialStream();
        val ptr = ptrDataBuffer.specialBuffer();

        if (offset > 0)
            temp = new PagedPointer(temp.address() + offset * getElementSize());

        NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(ptr, temp, length * Nd4j.sizeOfDataType(dtype), CudaConstants.cudaMemcpyHostToDevice, stream);
        stream.synchronize();

        // mark device buffer as updated
        allocationPoint.tickDeviceWrite();
    }

    protected void initHostPointerAndIndexer() {
        if (length() == 0)
            return;

        if (allocationPoint.getHostPointer() == null) {
            val location = allocationPoint.getAllocationStatus();
            if (parentWorkspace == null) {
                // let cpp allocate primary buffer
                NativeOpsHolder.getInstance().getDeviceNativeOps().dbAllocatePrimaryBuffer(ptrDataBuffer);
            } else {
                //log.info("ws alloc step");
                val ptr = parentWorkspace.alloc(this.length * this.elementSize, MemoryKind.HOST, this.dataType(), false);
                ptrDataBuffer.setPrimaryBuffer(ptr, this.length);
            }
            this.allocationPoint.setAllocationStatus(location);
            this.allocationPoint.tickDeviceWrite();
        }

        val hostPointer = allocationPoint.getHostPointer();

        assert hostPointer != null;

        switch (dataType()) {
            case DOUBLE:
                this.pointer = new CudaPointer(hostPointer, length, 0).asDoublePointer();
                indexer = DoubleIndexer.create((DoublePointer) pointer);
                break;
            case FLOAT:
                this.pointer = new CudaPointer(hostPointer, length, 0).asFloatPointer();
                indexer = FloatIndexer.create((FloatPointer) pointer);
                break;
            case UINT32:
            case INT:
                this.pointer = new CudaPointer(hostPointer, length, 0).asIntPointer();
                indexer = IntIndexer.create((IntPointer) pointer);
                break;
            case BFLOAT16:
                this.pointer = new CudaPointer(hostPointer, length, 0).asShortPointer();
                indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                break;
            case HALF:
                this.pointer = new CudaPointer(hostPointer, length, 0).asShortPointer();
                indexer = HalfIndexer.create((ShortPointer) pointer);
                break;
            case UINT64:
            case LONG:
                this.pointer = new CudaPointer(hostPointer, length, 0).asLongPointer();
                indexer = LongIndexer.create((LongPointer) pointer);
                break;
            case UINT16:
                this.pointer = new CudaPointer(hostPointer, length, 0).asShortPointer();
                indexer = UShortIndexer.create((ShortPointer) pointer);
                break;
            case SHORT:
                this.pointer = new CudaPointer(hostPointer, length, 0).asShortPointer();
                indexer = ShortIndexer.create((ShortPointer) pointer);
                break;
            case UBYTE:
                this.pointer = new CudaPointer(hostPointer, length, 0).asBytePointer();
                indexer = UByteIndexer.create((BytePointer) pointer);
                break;
            case BYTE:
                this.pointer = new CudaPointer(hostPointer, length, 0).asBytePointer();
                indexer = ByteIndexer.create((BytePointer) pointer);
                break;
            case BOOL:
                this.pointer = new CudaPointer(hostPointer, length, 0).asBooleanPointer();
                indexer = BooleanIndexer.create((BooleanPointer) pointer);
                break;
            case UTF8:
                this.pointer = new CudaPointer(hostPointer, length, 0).asBytePointer();
                indexer = ByteIndexer.create((BytePointer) pointer);
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    protected void initPointers(long length, int elementSize, boolean initialize) {
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
        this.length = length;
        this.elementSize =  (byte) elementSize;

        this.offset = 0;
        this.originalOffset = 0;

        // we allocate native DataBuffer AND it will contain our device pointer
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length, type, false);
        this.allocationPoint = new AllocationPoint(ptrDataBuffer, length * type.width());

        if (initialize) {
            val ctx = AtomicAllocator.getInstance().getDeviceContext();
            val devicePtr = allocationPoint.getDevicePointer();
            NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(devicePtr, 0, length * elementSize, 0, ctx.getSpecialStream());
            ctx.getSpecialStream().synchronize();
        }

        // let deallocator pick up this object
        Nd4j.getDeallocatorService().pickObject(this);
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

        this.length = length;

        this.offset = 0;
        this.originalOffset = 0;

        // allocating empty databuffer
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(0, type, false);

        if (workspace.getWorkspaceConfiguration().getPolicyMirroring() == MirroringPolicy.FULL) {
            val devicePtr = workspace.alloc(length * elementSize, MemoryKind.DEVICE, type, initialize);

            // allocate from workspace, and pass it  to native DataBuffer
            ptrDataBuffer.setSpecialBuffer(devicePtr, this.length);

            if (initialize) {
                val ctx = AtomicAllocator.getInstance().getDeviceContext();
                NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(devicePtr, 0, length * elementSize, 0, ctx.getSpecialStream());
                ctx.getSpecialStream().synchronize();
            }
        }  else {
            // we can register this pointer as device, because it's pinned memory
            val devicePtr = workspace.alloc(length * elementSize, MemoryKind.HOST, type, initialize);
            ptrDataBuffer.setSpecialBuffer(devicePtr, this.length);

            if (initialize) {
                val ctx = AtomicAllocator.getInstance().getDeviceContext();
                NativeOpsHolder.getInstance().getDeviceNativeOps().memsetAsync(devicePtr, 0, length * elementSize, 0, ctx.getSpecialStream());
                ctx.getSpecialStream().synchronize();
            }
        }

        this.allocationPoint = new AllocationPoint(ptrDataBuffer, elementSize * length);

        // registering for deallocation
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
        this.attached = true;
        this.parentWorkspace = workspace;
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
        this.elementSize = (byte) underlyingBuffer.getElementSize();

        // in case of view creation, we initialize underlying buffer regardless of anything
        ((BaseCudaDataBuffer) underlyingBuffer).lazyAllocateHostPointer();

        // we're creating view of the native DataBuffer
        ptrDataBuffer = ((BaseCudaDataBuffer) underlyingBuffer).ptrDataBuffer.createView(length * underlyingBuffer.getElementSize(), offset * underlyingBuffer.getElementSize());
        this.allocationPoint = new AllocationPoint(ptrDataBuffer, length);
        val hostPointer = allocationPoint.getHostPointer();

        Nd4j.getDeallocatorService().pickObject(this);

        switch (underlyingBuffer.dataType()) {
            case DOUBLE:
                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asDoublePointer();
                indexer = DoubleIndexer.create((DoublePointer) pointer);
                break;
            case FLOAT:
                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asFloatPointer();
                indexer = FloatIndexer.create((FloatPointer) pointer);
                break;
            case UINT32:
            case INT:
                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asIntPointer();
                indexer = IntIndexer.create((IntPointer) pointer);
                break;
            case BFLOAT16:
                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asShortPointer();
                indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                break;
            case HALF:
                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asShortPointer();
                indexer = HalfIndexer.create((ShortPointer) pointer);
                break;
            case UINT64:
            case LONG:
                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asLongPointer();
                indexer = LongIndexer.create((LongPointer) pointer);
                break;
            case UINT16:
                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asShortPointer();
                indexer = UShortIndexer.create((ShortPointer) pointer);
                break;
            case SHORT:
                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asShortPointer();
                indexer = ShortIndexer.create((ShortPointer) pointer);
                break;
            case BOOL:
                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asBooleanPointer();
                indexer = BooleanIndexer.create((BooleanPointer) pointer);
                break;
            case BYTE:
                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asBytePointer();
                indexer = ByteIndexer.create((BytePointer) pointer);
                break;
            case UBYTE:
                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asBytePointer();
                indexer = UByteIndexer.create((BytePointer) pointer);
                break;
            case UTF8:
                Preconditions.checkArgument(offset == 0, "String array can't be a view");

                this.pointer = new CudaPointer(hostPointer, originalBuffer.length()).asBytePointer();
                indexer = ByteIndexer.create((BytePointer) pointer);
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


    /**
     * This method always returns host pointer
     *
     * @return
     */
    @Override
    public long address() {
        if (released)
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        return allocationPoint.getHostPointer().address();
    }

    @Override
    public long platformAddress() {
        return allocationPoint.getDevicePointer().address();
    }

    @Override
    public Pointer pointer() {
        if (released)
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        // FIXME: very bad thing,
        lazyAllocateHostPointer();

        return super.pointer();
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
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case BYTE: {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

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
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case INT: {
                    val pointer = new IntPointer(data);
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case LONG: {
                    val pointer = new LongPointer(LongUtils.toLongs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case HALF: {
                    val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case FLOAT: {
                    val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case DOUBLE: {
                    val pointer = new DoublePointer(ArrayUtil.toDouble(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

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
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case BYTE: {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case UBYTE: {
                data = ArrayUtil.cutBelowZero(data);
                    for (int e = 0; e < data.length; e++) {
                        put(e, data[e]);
                    }
                }
                break;
            case UINT16:
                data = ArrayUtil.cutBelowZero(data);
            case SHORT: {
                    val pointer = new ShortPointer(ArrayUtil.toShorts(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case UINT32:
                data = ArrayUtil.cutBelowZero(data);
            case INT: {
                    val pointer = new IntPointer(ArrayUtil.toInts(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case UINT64:
                data = ArrayUtil.cutBelowZero(data);
            case LONG: {
                    val pointer = new LongPointer(data);
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case BFLOAT16: {
                val pointer = new ShortPointer(ArrayUtil.toBfloats(data));
                val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                // we're keeping pointer reference for JVM
                pointer.address();
            }
            break;
            case HALF: {
                    val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case FLOAT: {
                    val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case DOUBLE: {
                    val pointer = new DoublePointer(ArrayUtil.toDouble(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

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
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case BYTE: {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

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
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case INT: {
                    val pointer = new IntPointer(ArrayUtil.toInts(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case LONG: {
                    val pointer = new LongPointer(ArrayUtil.toLongArray(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case HALF: {
                    val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case FLOAT: {
                    val pointer = new FloatPointer(data);
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case DOUBLE: {
                    DoublePointer pointer = new DoublePointer(ArrayUtil.toDoubles(data));
                    Pointer srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

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
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case BYTE: {
                    val pointer = new BytePointer(ArrayUtil.toBytes(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

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
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case INT: {
                    val pointer = new IntPointer(ArrayUtil.toInts(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case LONG: {
                    val pointer = new LongPointer(ArrayUtil.toLongs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case HALF: {
                    val pointer = new ShortPointer(ArrayUtil.toHalfs(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case FLOAT: {
                    val pointer = new FloatPointer(ArrayUtil.toFloats(data));
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

                    allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

                    // we're keeping pointer reference for JVM
                    pointer.address();
                }
                break;
            case DOUBLE: {
                    val pointer = new DoublePointer(data);
                    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));

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
        if (data.length == 0)
            return;

        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(long[] data) {
        if (data.length == 0)
            return;

        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(float[] data) {
        if (data.length == 0)
            return;

        set(data, data.length, 0, 0);
    }

    @Override
    public void setData(double[] data) {
        if (data.length == 0)
            return;

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
        if (released)
            throw new IllegalStateException("You can't use DataBuffer once it was released");

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
    protected double getDoubleUnsynced(long index) {
        return super.getDouble(index);
    }

    @Override
    protected float getFloatUnsynced(long index) {
        return super.getFloat(index);
    }

    @Override
    protected long getLongUnsynced(long index) {
        return super.getLong(index);
    }

    @Override
    protected int getIntUnsynced(long index) {
        return super.getInt(index);
    }

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
        lazyAllocateHostPointer();
        AtomicAllocator.getInstance().synchronizeHostData(this);
        return super.toString();
    }

    @Override
    public boolean sameUnderlyingData(DataBuffer buffer) {
        return ptrDataBuffer.address() == ((BaseCudaDataBuffer) buffer).ptrDataBuffer.address();
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

            this.type = t;

            Nd4j.getDeallocatorService().pickObject(this);

            switch (type) {
                case DOUBLE: {
                        this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asDoublePointer();
                        indexer = DoubleIndexer.create((DoublePointer) pointer);
                    }
                    break;
                case FLOAT: {
                        this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asFloatPointer();
                        indexer = FloatIndexer.create((FloatPointer) pointer);
                    }
                    break;
                case HALF: {
                        this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asShortPointer();
                        indexer = HalfIndexer.create((ShortPointer) pointer);
                    }
                    break;
                case LONG: {
                        this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asLongPointer();
                        indexer = LongIndexer.create((LongPointer) pointer);
                    }
                    break;
                case INT: {
                        this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asIntPointer();
                        indexer = IntIndexer.create((IntPointer) pointer);
                    }
                    break;
                case SHORT: {
                        this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asShortPointer();
                        indexer = ShortIndexer.create((ShortPointer) pointer);
                    }
                    break;
                case UBYTE: {
                        this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asBytePointer();
                        indexer = UByteIndexer.create((BytePointer) pointer);
                    }
                    break;
                case BYTE: {
                        this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asBytePointer();
                        indexer = ByteIndexer.create((BytePointer) pointer);
                    }
                    break;
                case BOOL: {
                        this.pointer = new CudaPointer(allocationPoint.getHostPointer(), length).asBooleanPointer();
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
    public long[] asLong() {
        lazyAllocateHostPointer();
        allocator.synchronizeHostData(this);
        return super.asLong();
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

    public void actualizePointerAndIndexer() {
        val cptr = ptrDataBuffer.primaryBuffer();

        // skip update if pointers are equal
        if (cptr != null && pointer != null && cptr.address() == pointer.address())
            return;

        val t = dataType();
        if (t == DataType.BOOL) {
            pointer = new PagedPointer(cptr, length).asBoolPointer();
            setIndexer(BooleanIndexer.create((BooleanPointer) pointer));
        } else if (t == DataType.UBYTE) {
            pointer = new PagedPointer(cptr, length).asBytePointer();
            setIndexer(UByteIndexer.create((BytePointer) pointer));
        } else if (t == DataType.BYTE) {
            pointer = new PagedPointer(cptr, length).asBytePointer();
            setIndexer(ByteIndexer.create((BytePointer) pointer));
        } else if (t == DataType.UINT16) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(UShortIndexer.create((ShortPointer) pointer));
        } else if (t == DataType.SHORT) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(ShortIndexer.create((ShortPointer) pointer));
        } else if (t == DataType.UINT32) {
            pointer = new PagedPointer(cptr, length).asIntPointer();
            setIndexer(IntIndexer.create((IntPointer) pointer));
        } else if (t == DataType.INT) {
            pointer = new PagedPointer(cptr, length).asIntPointer();
            setIndexer(IntIndexer.create((IntPointer) pointer));
        } else if (t == DataType.UINT64) {
            pointer = new PagedPointer(cptr, length).asLongPointer();
            setIndexer(LongIndexer.create((LongPointer) pointer));
        } else if (t == DataType.LONG) {
            pointer = new PagedPointer(cptr, length).asLongPointer();
            setIndexer(LongIndexer.create((LongPointer) pointer));
        } else if (t == DataType.BFLOAT16) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(Bfloat16Indexer.create((ShortPointer) pointer));
        } else if (t == DataType.HALF) {
            pointer = new PagedPointer(cptr, length).asShortPointer();
            setIndexer(HalfIndexer.create((ShortPointer) pointer));
        } else if (t == DataType.FLOAT) {
            pointer = new PagedPointer(cptr, length).asFloatPointer();
            setIndexer(FloatIndexer.create((FloatPointer) pointer));
        } else if (t == DataType.DOUBLE) {
            pointer = new PagedPointer(cptr, length).asDoublePointer();
            setIndexer(DoubleIndexer.create((DoublePointer) pointer));
        } else if (t == DataType.UTF8) {
            pointer = new PagedPointer(cptr, length()).asBytePointer();
            setIndexer(ByteIndexer.create((BytePointer) pointer));
        } else
            throw new IllegalArgumentException("Unknown datatype: " + dataType());
    }

    @Override
    public DataBuffer reallocate(long length) {
        val oldHostPointer = this.ptrDataBuffer.primaryBuffer();
        val oldDevicePointer = this.ptrDataBuffer.specialBuffer();

        if (isAttached()) {
            val capacity = length * getElementSize();

            if (oldDevicePointer != null && oldDevicePointer.address() != 0) {
                val nPtr = getParentWorkspace().alloc(capacity, MemoryKind.DEVICE, dataType(), false);
                NativeOpsHolder.getInstance().getDeviceNativeOps().memcpySync(nPtr, oldDevicePointer, length * getElementSize(), 3, null);
                this.ptrDataBuffer.setPrimaryBuffer(nPtr, length);

                allocationPoint.tickDeviceRead();
            }

            if (oldHostPointer != null && oldHostPointer.address() != 0) {
                val nPtr = getParentWorkspace().alloc(capacity, MemoryKind.HOST, dataType(), false);
                Pointer.memcpy(nPtr, oldHostPointer, this.length() * getElementSize());
                this.ptrDataBuffer.setPrimaryBuffer(nPtr, length);

                allocationPoint.tickHostRead();

                switch (dataType()) {
                    case BOOL:
                        pointer = nPtr.asBoolPointer();
                        indexer = BooleanIndexer.create((BooleanPointer) pointer);
                        break;
                    case UTF8:
                    case BYTE:
                    case UBYTE:
                        pointer = nPtr.asBytePointer();
                        indexer = ByteIndexer.create((BytePointer) pointer);
                        break;
                    case UINT16:
                    case SHORT:
                        pointer = nPtr.asShortPointer();
                        indexer = ShortIndexer.create((ShortPointer) pointer);
                        break;
                    case UINT32:
                    case INT:
                        pointer = nPtr.asIntPointer();
                        indexer = IntIndexer.create((IntPointer) pointer);
                        break;
                    case DOUBLE:
                        pointer = nPtr.asDoublePointer();
                        indexer = DoubleIndexer.create((DoublePointer) pointer);
                        break;
                    case FLOAT:
                        pointer = nPtr.asFloatPointer();
                        indexer = FloatIndexer.create((FloatPointer) pointer);
                        break;
                    case HALF:
                        pointer = nPtr.asShortPointer();
                        indexer = HalfIndexer.create((ShortPointer) pointer);
                        break;
                    case BFLOAT16:
                        pointer = nPtr.asShortPointer();
                        indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                        break;
                    case UINT64:
                    case LONG:
                        pointer = nPtr.asLongPointer();
                        indexer = LongIndexer.create((LongPointer) pointer);
                        break;
                }
            }


            workspaceGenerationId = getParentWorkspace().getGenerationId();
        } else {
            this.ptrDataBuffer.expand(length);
            val nPtr = new PagedPointer(this.ptrDataBuffer.primaryBuffer(), length);

            switch (dataType()) {
                case BOOL:
                    pointer = nPtr.asBoolPointer();
                    indexer = BooleanIndexer.create((BooleanPointer) pointer);
                    break;
                case UTF8:
                case BYTE:
                case UBYTE:
                    pointer = nPtr.asBytePointer();
                    indexer = ByteIndexer.create((BytePointer) pointer);
                    break;
                case UINT16:
                case SHORT:
                    pointer = nPtr.asShortPointer();
                    indexer = ShortIndexer.create((ShortPointer) pointer);
                    break;
                case UINT32:
                case INT:
                    pointer = nPtr.asIntPointer();
                    indexer = IntIndexer.create((IntPointer) pointer);
                    break;
                case DOUBLE:
                    pointer = nPtr.asDoublePointer();
                    indexer = DoubleIndexer.create((DoublePointer) pointer);
                    break;
                case FLOAT:
                    pointer = nPtr.asFloatPointer();
                    indexer = FloatIndexer.create((FloatPointer) pointer);
                    break;
                case HALF:
                    pointer = nPtr.asShortPointer();
                    indexer = HalfIndexer.create((ShortPointer) pointer);
                    break;
                case BFLOAT16:
                    pointer = nPtr.asShortPointer();
                    indexer = Bfloat16Indexer.create((ShortPointer) pointer);
                    break;
                case UINT64:
                case LONG:
                    pointer = nPtr.asLongPointer();
                    indexer = LongIndexer.create((LongPointer) pointer);
                    break;
            }
        }

        this.underlyingLength = length;
        this.length = length;
        return this;
    }

    @Override
    public long capacity() {
        if (allocationPoint.getHostPointer() != null)
            return pointer.capacity();
        else
            return length;
    }

    @Override
    protected void release() {
        if (!released) {
            //AtomicAllocator.getInstance().freeMemory(allocationPoint);n
            NativeOpsHolder.getInstance().getDeviceNativeOps().dbClose(allocationPoint.getPtrDataBuffer());
            allocationPoint.setReleased(true);
        }
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

    @Override
    public void syncToPrimary(){
        ptrDataBuffer.syncToPrimary();
    }

    @Override
    public void syncToSpecial(){
        ptrDataBuffer.syncToSpecial();
    }
}
