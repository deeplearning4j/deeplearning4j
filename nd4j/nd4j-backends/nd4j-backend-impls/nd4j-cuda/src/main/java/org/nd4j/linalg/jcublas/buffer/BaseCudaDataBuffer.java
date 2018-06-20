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

package org.nd4j.linalg.jcublas.buffer;

import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AllocationShape;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.memory.MemcpyDirection;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LongUtils;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
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
public abstract class BaseCudaDataBuffer extends BaseDataBuffer implements JCudaBuffer {

    @Getter
    protected transient AllocationPoint allocationPoint;

    private static AtomicAllocator allocator = AtomicAllocator.getInstance();

    private static Logger log = LoggerFactory.getLogger(BaseCudaDataBuffer.class);

    protected Type globalType = DataTypeUtil.getDtypeFromContext();

    public BaseCudaDataBuffer() {

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
        this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this,
                        new AllocationShape(length, elementSize, dataType()), false);

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


    public BaseCudaDataBuffer(long length, int elementSize, boolean initialize) {
        this.allocationMode = AllocationMode.LONG_SHAPE;
        initTypeAndSize();
        this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this,
                        new AllocationShape(length, elementSize, dataType()), initialize);
        this.length = length;
        //allocationPoint.attachBuffer(this);
        this.elementSize =  (byte) elementSize;
        this.trackingPoint = allocationPoint.getObjectId();
        this.offset = 0;
        this.originalOffset = 0;

        //  if (Nd4j.getAffinityManager().getDeviceForCurrentThread() == 0)
        //log.info("Allocating {} bytes on device_{}", length, Nd4j.getAffinityManager().getDeviceForCurrentThread());

        if (dataType() == Type.DOUBLE) {
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asDoublePointer();
            indexer = DoubleIndexer.create((DoublePointer) pointer);
        } else if (dataType() == Type.FLOAT) {
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asFloatPointer();
            indexer = FloatIndexer.create((FloatPointer) pointer);
        } else if (dataType() == Type.INT) {
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asIntPointer();
            indexer = IntIndexer.create((IntPointer) pointer);
        } else if (dataType() == Type.HALF) {
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
            indexer = HalfIndexer.create((ShortPointer) pointer);
        } else if (dataType() == Type.LONG) {
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asLongPointer();
            indexer = LongIndexer.create((LongPointer) pointer);
        }
    }

    public BaseCudaDataBuffer(long length, int elementSize, boolean initialize, @NonNull MemoryWorkspace workspace) {
        this.allocationMode = AllocationMode.LONG_SHAPE;
        initTypeAndSize();

        this.attached = true;
        this.parentWorkspace = workspace;

        this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this, new AllocationShape(length, this.elementSize, dataType()), initialize);
        this.length = length;
        //allocationPoint.attachBuffer(this);
        //this.elementSize = elementSize;
        this.trackingPoint = allocationPoint.getObjectId();
        this.offset = 0;
        this.originalOffset = 0;


        if (dataType() == Type.DOUBLE) {
            this.attached = true;
            this.parentWorkspace = workspace;

            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asDoublePointer();
            indexer = DoubleIndexer.create((DoublePointer) pointer);
        } else if (dataType() == Type.FLOAT) {
            this.attached = true;
            this.parentWorkspace = workspace;

            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asFloatPointer();
            indexer = FloatIndexer.create((FloatPointer) pointer);
        } else if (dataType() == Type.INT) {
            this.attached = true;
            this.parentWorkspace = workspace;

            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asIntPointer();
            indexer = IntIndexer.create((IntPointer) pointer);
        } else if (dataType() == Type.HALF) {
            this.attached = true;
            this.parentWorkspace = workspace;

            // FIXME: proper pointer and proper indexer should be used here
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
            indexer = HalfIndexer.create((ShortPointer) pointer);
        } else if (dataType() == Type.LONG) {
            this.attached = true;
            this.parentWorkspace = workspace;

            // FIXME: proper pointer and proper indexer should be used here
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asLongPointer();
            indexer = LongIndexer.create((LongPointer) pointer);
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
        this.allocationMode = AllocationMode.LONG_SHAPE;
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

        if (underlyingBuffer.dataType() == Type.DOUBLE) {
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asDoublePointer();
            indexer = DoubleIndexer.create((DoublePointer) pointer);
        } else if (underlyingBuffer.dataType() == Type.FLOAT) {
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asFloatPointer();
            indexer = FloatIndexer.create((FloatPointer) pointer);
        } else if (underlyingBuffer.dataType() == Type.INT) {
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asIntPointer();
            indexer = IntIndexer.create((IntPointer) pointer);
        } else if (underlyingBuffer.dataType() == Type.HALF) {
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asShortPointer();
            indexer = HalfIndexer.create((ShortPointer) pointer);
        } else if (underlyingBuffer.dataType() == Type.LONG) {
            this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), originalBuffer.length()).asLongPointer();
            indexer = LongIndexer.create((LongPointer) pointer);
        }
    }

    public BaseCudaDataBuffer(long length) {
        this(length, Nd4j.dataType() == Type.DOUBLE ? 8 : Nd4j.dataType() == Type.FLOAT ? 4 : 2);
    }

    public BaseCudaDataBuffer(float[] data) {
        //super(data);
        this(data.length, Nd4j.dataType() == Type.DOUBLE ? 8 : Nd4j.dataType() == Type.FLOAT ? 4 : 2, false);
        set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(int[] data) {
        //super(data);
        this(data.length, Nd4j.dataType() == Type.DOUBLE ? 8 : Nd4j.dataType() == Type.FLOAT ? 4 : 2, false);
        set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(long[] data) {
        //super(data);
        this(data.length, Nd4j.dataType() == Type.DOUBLE ? 8 : Nd4j.dataType() == Type.FLOAT ? 4 : 2, false);
        set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(long[] data, boolean copy) {
        //super(data);
        this(data.length, Nd4j.dataType() == Type.DOUBLE ? 8 : Nd4j.dataType() == Type.FLOAT ? 4 : 2, false);

        if (copy)
            set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(double[] data) {
        // super(data);
        this(data.length, Nd4j.dataType() == Type.DOUBLE ? 8 : Nd4j.dataType() == Type.FLOAT ? 4 : 2, false);
        set(data, data.length, 0, 0);
    }

    public BaseCudaDataBuffer(byte[] data, long length) {
        this(ByteBuffer.wrap(data), length);
    }

    public BaseCudaDataBuffer(ByteBuffer buffer, long length) {
        //super(buffer,length);
        this(buffer, length, 0);
    }

    public BaseCudaDataBuffer(ByteBuffer buffer, long length, long offset) {
        //super(buffer, length, offset);
        this(length, Nd4j.dataType() == Type.DOUBLE ? 8 : Nd4j.dataType() == Type.FLOAT ? 4 : 2, offset);

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
        if (dataType() == Type.DOUBLE) {
            DoublePointer pointer = new DoublePointer(ArrayUtil.toDouble(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.FLOAT) {
            FloatPointer pointer = new FloatPointer(ArrayUtil.toFloats(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.INT) {
            IntPointer pointer = new IntPointer(data);
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.HALF) {
            ShortPointer pointer = new ShortPointer(ArrayUtil.toHalfs(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.LONG) {
            LongPointer pointer = new LongPointer(LongUtils.toLongs(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        }
    }


    public void set(long[] data, long length, long srcOffset, long dstOffset) {
        // TODO: make sure getPointer returns proper pointer
        if (dataType() == Type.DOUBLE) {
            DoublePointer pointer = new DoublePointer(ArrayUtil.toDouble(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.FLOAT) {
            FloatPointer pointer = new FloatPointer(ArrayUtil.toFloats(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.INT) {
            IntPointer pointer = new IntPointer(ArrayUtil.toInts(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.HALF) {
            ShortPointer pointer = new ShortPointer(ArrayUtil.toHalfs(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.LONG) {
            LongPointer pointer = new LongPointer(data);
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
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
        // TODO: make sure getPointer returns proper pointer
        //        log.info("Set called");
        if (dataType() == Type.DOUBLE) {
            //Pointer dstPtr = dstOffset > 0 ? new Pointer(allocator.getPointer(this).address()).withByteOffset(dstOffset * 4) : new Pointer(allocator.getPointer(this).address());
            //Pointer srcPtr = srcOffset > 0 ? Pointer.to(ArrayUtil.toDoubles(data)).withByteOffset(srcOffset * elementSize) : Pointer.to(ArrayUtil.toDoubles(data));
            DoublePointer pointer = new DoublePointer(ArrayUtil.toDoubles(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.FLOAT) {
            //Pointer srcPtr = srcOffset > 0 ? Pointer.to(data).withByteOffset(srcOffset * elementSize) : Pointer.to(data);
            FloatPointer pointer = new FloatPointer(data);
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            //log.info("Memcpy params: byteLength: ["+(length * elementSize)+"], srcOffset: ["+(srcOffset * elementSize)+"], dstOffset: [" +(dstOffset* elementSize) + "]" );

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.INT) {
            //Pointer srcPtr = srcOffset > 0 ? Pointer.to(ArrayUtil.toInts(data)).withByteOffset(srcOffset * elementSize) : Pointer.to(ArrayUtil.toInts(data));
            IntPointer pointer = new IntPointer(ArrayUtil.toInts(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.HALF) {
            ShortPointer pointer = new ShortPointer(ArrayUtil.toHalfs(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
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
        // TODO: make sure getPointer returns proper pointer
        if (dataType() == Type.DOUBLE) {
            DoublePointer pointer = new DoublePointer(data);
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.FLOAT) {
            FloatPointer pointer = new FloatPointer(ArrayUtil.toFloats(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.INT) {
            IntPointer pointer = new IntPointer(ArrayUtil.toInts(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
        } else if (dataType() == Type.HALF) {
            ShortPointer pointer = new ShortPointer(ArrayUtil.toHalfs(data));
            Pointer srcPtr = new CudaPointer(pointer.address() + (dstOffset * elementSize));

            allocator.memcpyAsync(this, srcPtr, length * elementSize, dstOffset * elementSize);

            // we're keeping pointer reference for JVM
            pointer.address();
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

    @Override
    public void put(long i, IComplexNumber result) {
        throw new UnsupportedOperationException("ComplexNumbers are not supported yet");
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
        allocator.synchronizeHostData(this);
        allocator.tickHostWrite(this);
        super.put(i, element);
    }

    @Override
    public void put(long i, double element) {
        allocator.synchronizeHostData(this);
        allocator.tickHostWrite(this);
        super.put(i, element);
    }

    @Override
    public void put(long i, int element) {
        allocator.synchronizeHostData(this);
        allocator.tickHostWrite(this);
        super.put(i, element);
    }

    @Override
    public Pointer addressPointer() {
        return AtomicAllocator.getInstance().getHostPointer(this);
    }

    @Override
    public IComplexFloat getComplexFloat(long i) {
        return Nd4j.createFloat(getFloat(i), getFloat(i + 1));
    }

    @Override
    public IComplexDouble getComplexDouble(long i) {
        return Nd4j.createDouble(getDouble(i), getDouble(i + 1));
    }

    @Override
    public IComplexNumber getComplex(long i) {
        return dataType() == Type.FLOAT ? getComplexFloat(i) : getComplexDouble(i);
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
        allocator.synchronizeHostData(this);
        super.write(out);
    }

    @Override
    public void write(OutputStream dos) {
        allocator.synchronizeHostData(this);
        super.write(dos);
    }

    private void writeObject(java.io.ObjectOutputStream stream) throws IOException {
        allocator.synchronizeHostData(this);
        stream.defaultWriteObject();
        write(stream);
    }

    private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException {
        doReadObject(stream);
        // TODO: to be implemented
        /*
        copied = new HashMap<>();
        pointersToContexts = HashBasedTable.create();
        ref = new WeakReference<DataBuffer>(this,Nd4j.bufferRefQueue());
        freed = new AtomicBoolean(false);
        */
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < length(); i++) {
            sb.append(getDouble(i));
            if (i < length() - 1)
                sb.append(",");
        }
        sb.append("]");
        return sb.toString();

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
    public void read(DataInputStream s) {
        try {
            allocationMode = AllocationMode.valueOf(s.readUTF());

            long locLength = 0;

            if (allocationMode.ordinal() < 3)
                locLength = s.readInt();
            else
                locLength = s.readLong();

            boolean reallocate = locLength != length || indexer == null;
            length = locLength;

            Type t = Type.valueOf(s.readUTF());
            //                  log.info("Restoring buffer ["+t+"] of length ["+ length+"]");
            if (globalType == null && Nd4j.dataType() != null) {
                globalType = Nd4j.dataType();
            }

            if (t != globalType && (!t.equals(Type.INT) && !t.equals(Type.LONG)) && Nd4j.sizeOfDataType(globalType) < Nd4j.sizeOfDataType(t)) {
                log.warn("Loading a data stream with opType different from what is set globally. Expect precision loss");
                if (globalType == Type.INT)
                    log.warn("Int to float/double widening UNSUPPORTED!!!");
            }
            if (t == Type.COMPRESSED) {
                type = t;
                return;
            } else if (t == Type.LONG || globalType == Type.LONG) {
                this.elementSize = 8;
                this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this,
                        new AllocationShape(length, elementSize, t), false);
                this.trackingPoint = allocationPoint.getObjectId();

                // we keep int buffer's dtype after ser/de
                this.type = t;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asLongPointer();
                indexer = LongRawIndexer.create((LongPointer) pointer);

                LongRawIndexer Lindexer = (LongRawIndexer) indexer;

                for (int i = 0; i < length(); i++) {
                    if (t == Type.LONG)
                        Lindexer.put(i, s.readLong());
                    else if (t == Type.INT)
                        Lindexer.put(i, s.readInt());
                    else if (t == Type.DOUBLE)
                        Lindexer.put(i, (int) s.readDouble());
                    else if (t == Type.FLOAT)
                        Lindexer.put(i, (int) s.readFloat());
                    else if (t == Type.HALF)
                        Lindexer.put(i, (int) toFloat((int) s.readShort()));
                }

                allocationPoint.tickHostWrite();

            } else if (t == Type.INT || globalType == Type.INT) {
                this.elementSize = 4;
                this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this,
                                new AllocationShape(length, elementSize, t), false);
                this.trackingPoint = allocationPoint.getObjectId();

                // we keep int buffer's dtype after ser/de
                this.type = t;

                this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asIntPointer();
                indexer = IntIndexer.create((IntPointer) pointer);

                IntIndexer Iindexer = (IntIndexer) indexer;

                for (int i = 0; i < length(); i++) {
                    if (t == Type.INT)
                        Iindexer.put(i, s.readInt());
                    else if (t == Type.LONG)
                        Iindexer.put(i, (int) s.readLong());
                    else if (t == Type.DOUBLE)
                        Iindexer.put(i, (int) s.readDouble());
                    else if (t == Type.FLOAT)
                        Iindexer.put(i, (int) s.readFloat());
                    else if (t == Type.HALF)
                        Iindexer.put(i, (int) toFloat((int) s.readShort()));
                }

                allocationPoint.tickHostWrite();

            } else if (globalType == Type.DOUBLE) {
                this.elementSize = 8;

                if (reallocate) {
                    MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();
                    if (workspace != null && (workspace instanceof DummyWorkspace)) {
                        this.attached = true;
                        this.parentWorkspace = workspace;
                        workspaceGenerationId = workspace.getGenerationId();
                    }

                    this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this,
                            new AllocationShape(length, elementSize, globalType), false);
                    //allocationPoint.attachBuffer(this);
                    this.trackingPoint = allocationPoint.getObjectId();

                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length)
                            .asDoublePointer();
                    indexer = DoubleIndexer.create((DoublePointer) pointer);
                }

                DoubleIndexer Dindexer = (DoubleIndexer) indexer;

                for (int i = 0; i < length(); i++) {
                    if (t == Type.DOUBLE)
                        Dindexer.put(i, s.readDouble());
                    else if (t == Type.LONG)
                        Dindexer.put(i, (double) s.readLong());
                    else if (t == Type.FLOAT)
                        Dindexer.put(i, (double) s.readFloat());
                    else if (t == Type.HALF)
                        Dindexer.put(i, (double) toFloat((int) s.readShort()));
                }

                allocationPoint.tickHostWrite();

            } else if (globalType == Type.FLOAT) {
                this.elementSize = 4;
                if (reallocate) {
                    this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this,
                            new AllocationShape(length, elementSize, dataType()), false);
                    this.trackingPoint = allocationPoint.getObjectId();

                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asFloatPointer();
                    indexer = FloatIndexer.create((FloatPointer) pointer);
                }

                FloatIndexer Findexer = (FloatIndexer) indexer;

                for (int i = 0; i < length; i++) {


                    if (t == Type.DOUBLE)
                        Findexer.put(i, (float) s.readDouble());
                    else if (t == Type.LONG)
                        Findexer.put(i, (float) s.readLong());
                    else if (t == Type.FLOAT)
                        Findexer.put(i, s.readFloat());
                    else if (t == Type.HALF) {
                        Findexer.put(i, toFloat((int) s.readShort()));
                    }
                }

                allocationPoint.tickHostWrite();
            } else if (globalType == Type.HALF) {
                this.elementSize = 2;
                if (reallocate) {
                    this.allocationPoint = AtomicAllocator.getInstance().allocateMemory(this,
                            new AllocationShape(length, elementSize, dataType()), false);
                    this.trackingPoint = allocationPoint.getObjectId();

                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length).asShortPointer();
                    indexer = HalfIndexer.create((ShortPointer) this.pointer);

                }

                HalfIndexer Hindexer = (HalfIndexer) indexer;

                for (int i = 0; i < length; i++) {

                    if (t == Type.DOUBLE)
                        Hindexer.put(i, (float) s.readDouble());
                    else if (t == Type.LONG)
                        Hindexer.put(i, (float) s.readLong());
                    else if (t == Type.FLOAT)
                        Hindexer.put(i, s.readFloat());
                    else if (t == Type.HALF) {
                        Hindexer.put(i, toFloat((int) s.readShort()));
                    }
                }

                // for HALF & HALF2 datatype we just tag data as fresh on host
                allocationPoint.tickHostWrite();
            } else
                throw new IllegalStateException("Unknown dataType: [" + t.toString() + "]");

            /*
            this.wrappedBuffer = this.pointer.asByteBuffer();
            this.wrappedBuffer.order(ByteOrder.nativeOrder());
            */

        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // we call sync to copyback data to host
        AtomicAllocator.getInstance().getFlowController().synchronizeToDevice(allocationPoint);
        //allocator.synchronizeHostData(this);
    }

    @Override
    public byte[] asBytes() {
        allocator.synchronizeHostData(this);
        return super.asBytes();
    }

    @Override
    public double[] asDouble() {
        allocator.synchronizeHostData(this);
        return super.asDouble();
    }

    @Override
    public float[] asFloat() {
        allocator.synchronizeHostData(this);
        return super.asFloat();
    }

    @Override
    public int[] asInt() {
        allocator.synchronizeHostData(this);
        return super.asInt();
    }

    @Override
    public ByteBuffer asNio() {
        allocator.synchronizeHostData(this);
        return super.asNio();
    }

    @Override
    public DoubleBuffer asNioDouble() {
        allocator.synchronizeHostData(this);
        return super.asNioDouble();
    }

    @Override
    public FloatBuffer asNioFloat() {
        allocator.synchronizeHostData(this);
        return super.asNioFloat();
    }

    @Override
    public IntBuffer asNioInt() {
        allocator.synchronizeHostData(this);
        return super.asNioInt();
    }

    @Override
    public DataBuffer dup() {
        allocator.synchronizeHostData(this);
        DataBuffer buffer = create(this.length);
        allocator.memcpyBlocking(buffer, new CudaPointer(allocator.getHostPointer(this).address()),
                        this.length * elementSize, 0);
        return buffer;
    }

    @Override
    public Number getNumber(long i) {
        allocator.synchronizeHostData(this);
        return super.getNumber(i);
    }

    @Override
    public double getDouble(long i) {
        allocator.synchronizeHostData(this);
        return super.getDouble(i);
    }

    @Override
    public float getFloat(long i) {
        allocator.synchronizeHostData(this);

        //log.info("Requesting data:  trackingPoint: ["+ trackingPoint+"], length: ["+length+"], offset: ["+ offset+ "], position: ["+ i  +"], elementSize: [" +getElementSize() + "], byteoffset: ["+ (offset + i) * getElementSize() + "], bufferCapacity: ["+this.wrappedBuffer.capacity()+"], dtype: ["+dataType()+"]");

        return super.getFloat(i);
        //return wrappedBuffer.getFloat((int)(offset + i) * getElementSize());
    }

    @Override
    public int getInt(long ix) {
        allocator.synchronizeHostData(this);
        return super.getInt(ix);
    }

    @Override
    public DataBuffer reallocate(long length) {

        // we want to be sure this array isn't used anywhere RIGHT AT THIS MOMENT
        Nd4j.getExecutioner().commit();


            AllocationPoint old = allocationPoint;
            allocationPoint = AtomicAllocator.getInstance().allocateMemory(this, new AllocationShape(length, elementSize, dataType()), false);

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
                case HALF:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asShortPointer();
                    indexer = ShortIndexer.create((ShortPointer) pointer);
                    break;
                case INT:
                    this.pointer = new CudaPointer(allocationPoint.getPointers().getHostPointer(), length, 0).asIntPointer();
                    indexer = IntIndexer.create((IntPointer) pointer);
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
}
