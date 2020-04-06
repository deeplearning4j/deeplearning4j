/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.linalg.cpu.nativecpu.buffer;

import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.linalg.api.buffer.BaseDataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.AllocUtil;
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.nio.ByteBuffer;

import static org.nd4j.linalg.api.buffer.DataType.INT8;

/**
 * Base implementation for DataBuffer for CPU-like backend
 *
 * @author raver119@gmail.com
 */
public abstract class BaseCpuDataBuffer extends BaseDataBuffer implements Deallocatable {

    protected transient OpaqueDataBuffer ptrDataBuffer;

    private transient final long instanceId = Nd4j.getDeallocatorService().nextValue();

    protected BaseCpuDataBuffer() {

    }


    @Override
    public String getUniqueId() {
        return new String("BCDB_" + instanceId);
    }

    @Override
    public Deallocator deallocator() {
        return new CpuDeallocator(this);
    }

    public OpaqueDataBuffer getOpaqueDataBuffer() {
        return ptrDataBuffer;
    }

    @Override
    public int targetDevice() {
        // TODO: once we add NUMA support this might change. Or might not.
        return 0;
    }


    /**
     *
     * @param length
     * @param elementSize
     */
    public BaseCpuDataBuffer(long length, int elementSize) {
        if (length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        allocationMode = AllocUtil.getAllocationModeFromContext();
        this.length = length;
        this.underlyingLength = length;
        this.elementSize = (byte) elementSize;

        if (dataType() != DataType.UTF8)
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length, dataType(), false);

        if (dataType() == DataType.DOUBLE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asDoublePointer();

            indexer = DoubleIndexer.create((DoublePointer) pointer);
        } else if (dataType() == DataType.FLOAT) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asFloatPointer();

            setIndexer(FloatIndexer.create((FloatPointer) pointer));
        } else if (dataType() == DataType.INT32) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asIntPointer();

            setIndexer(IntIndexer.create((IntPointer) pointer));
        } else if (dataType() == DataType.LONG) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asLongPointer();

            setIndexer(LongIndexer.create((LongPointer) pointer));
        } else if (dataType() == DataType.SHORT) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();

            setIndexer(ShortIndexer.create((ShortPointer) pointer));
        } else if (dataType() == DataType.BYTE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBytePointer();

            setIndexer(ByteIndexer.create((BytePointer) pointer));
        } else if (dataType() == DataType.UBYTE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBytePointer();

            setIndexer(UByteIndexer.create((BytePointer) pointer));
        } else if (dataType() == DataType.UTF8) {
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length, INT8, false);
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBytePointer();

            setIndexer(ByteIndexer.create((BytePointer) pointer));
        }

        Nd4j.getDeallocatorService().pickObject(this);
    }

    /**
     *
     * @param length
     * @param elementSize
     */
    public BaseCpuDataBuffer(int length, int elementSize, long offset) {
        this(length, elementSize);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = length - offset;
        this.underlyingLength = length;
    }


    protected BaseCpuDataBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);

        // for vew we need "externally managed" pointer and deallocator registration
        ptrDataBuffer = ((BaseCpuDataBuffer) underlyingBuffer).ptrDataBuffer.createView(length * underlyingBuffer.getElementSize(), offset * underlyingBuffer.getElementSize());
        Nd4j.getDeallocatorService().pickObject(this);


        // update pointer now
        actualizePointerAndIndexer();
    }

    protected BaseCpuDataBuffer(ByteBuffer buffer, DataType dtype, long length, long offset) {
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

        val ptr = ptrDataBuffer.primaryBuffer();

        if (offset > 0)
            temp = new PagedPointer(temp.address() + offset * getElementSize());

        Pointer.memcpy(ptr, temp, length * Nd4j.sizeOfDataType(dtype));
    }

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
    public void pointerIndexerByCurrentType(DataType currentType) {

        type = currentType;

        if (ptrDataBuffer == null) {
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length(), type, false);
            Nd4j.getDeallocatorService().pickObject(this);
        }

        actualizePointerAndIndexer();
    }

    /**
     * Instantiate a buffer with the given length
     *
     * @param length the length of the buffer
     */
    protected BaseCpuDataBuffer(long length) {
        this(length, true);
    }

    protected BaseCpuDataBuffer(long length, boolean initialize) {
        if (length < 0)
            throw new IllegalArgumentException("Length must be >= 0");
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();
        if (length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

        if (dataType() != DataType.UTF8)
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length, dataType(), false);

        if (dataType() == DataType.DOUBLE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asDoublePointer();

            indexer = DoubleIndexer.create((DoublePointer) pointer);

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.FLOAT) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asFloatPointer();

            setIndexer(FloatIndexer.create((FloatPointer) pointer));

            if (initialize)
                fillPointerWithZero();

        } else if (dataType() == DataType.HALF) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();

            setIndexer(HalfIndexer.create((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.BFLOAT16) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();

            setIndexer(Bfloat16Indexer.create((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.INT) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asIntPointer();

            setIndexer(IntIndexer.create((IntPointer) pointer));
            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.LONG) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asLongPointer();

            setIndexer(LongIndexer.create((LongPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.BYTE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBytePointer();

            setIndexer(ByteIndexer.create((BytePointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.SHORT) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();

            setIndexer(ShortIndexer.create((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UBYTE) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBytePointer();

            setIndexer(UByteIndexer.create((BytePointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UINT16) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asShortPointer();

            setIndexer(UShortIndexer.create((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UINT32) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asIntPointer();

            // FIXME: we need unsigned indexer here
            setIndexer(IntIndexer.create((IntPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UINT64) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asLongPointer();

            // FIXME: we need unsigned indexer here
            setIndexer(LongIndexer.create((LongPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.BOOL) {
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length).asBoolPointer();

            setIndexer(BooleanIndexer.create((BooleanPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UTF8) {
            // we are allocating buffer as INT8 intentionally
            ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(length(), INT8, false);
            pointer = new PagedPointer(ptrDataBuffer.primaryBuffer(), length()).asBytePointer();

            setIndexer(ByteIndexer.create((BytePointer) pointer));

            if (initialize)
                fillPointerWithZero();
        }

        Nd4j.getDeallocatorService().pickObject(this);
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
    public Pointer addressPointer() {
        // we're fetching actual pointer right from C++
        val tempPtr = new PagedPointer(ptrDataBuffer.primaryBuffer());

        switch (this.type) {
            case DOUBLE: return tempPtr.asDoublePointer();
            case FLOAT: return tempPtr.asFloatPointer();
            case UINT16:
            case SHORT:
            case BFLOAT16:
            case HALF: return tempPtr.asShortPointer();
            case UINT32:
            case INT: return tempPtr.asIntPointer();
            case UBYTE:
            case BYTE: return tempPtr.asBytePointer();
            case UINT64:
            case LONG: return tempPtr.asLongPointer();
            case BOOL: return tempPtr.asBoolPointer();
            default: return tempPtr.asBytePointer();
        }
    }

    protected BaseCpuDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        if (length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();



        if (length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

        // creating empty native DataBuffer
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(0, dataType(), false);

        if (dataType() == DataType.DOUBLE) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asDoublePointer(); //new DoublePointer(length());
            indexer = DoubleIndexer.create((DoublePointer) pointer);

        } else if (dataType() == DataType.FLOAT) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asFloatPointer(); //new FloatPointer(length());
            setIndexer(FloatIndexer.create((FloatPointer) pointer));

        } else if (dataType() == DataType.HALF) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new FloatPointer(length());
            setIndexer(HalfIndexer.create((ShortPointer) pointer));

        } else if (dataType() == DataType.BFLOAT16) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new FloatPointer(length());
            setIndexer(Bfloat16Indexer.create((ShortPointer) pointer));
        } else if (dataType() == DataType.INT) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asIntPointer(); //new IntPointer(length());
            setIndexer(IntIndexer.create((IntPointer) pointer));

        } else if (dataType() == DataType.UINT32) {
            attached = true;
            parentWorkspace = workspace;

            // FIXME: need unsigned indexer here
            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asIntPointer(); //new IntPointer(length());
            setIndexer(IntIndexer.create((IntPointer) pointer));

        } else if (dataType() == DataType.UINT64) {
            attached = true;
            parentWorkspace = workspace;

            // FIXME: need unsigned indexer here
            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asLongPointer(); //new IntPointer(length());
            setIndexer(LongIndexer.create((LongPointer) pointer));

        } else if (dataType() == DataType.LONG) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asLongPointer(); //new LongPointer(length());
            setIndexer(LongIndexer.create((LongPointer) pointer));
        } else if (dataType() == DataType.BYTE) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asBytePointer(); //new LongPointer(length());
            setIndexer(ByteIndexer.create((BytePointer) pointer));
        } else if (dataType() == DataType.UBYTE) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asBytePointer(); //new LongPointer(length());
            setIndexer(UByteIndexer.create((BytePointer) pointer));
        } else if (dataType() == DataType.UINT16) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new IntPointer(length());
            setIndexer(UShortIndexer.create((ShortPointer) pointer));

        } else if (dataType() == DataType.SHORT) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asShortPointer(); //new LongPointer(length());
            setIndexer(ShortIndexer.create((ShortPointer) pointer));
        } else if (dataType() == DataType.BOOL) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asBoolPointer(); //new LongPointer(length());
            setIndexer(BooleanIndexer.create((BooleanPointer) pointer));
        } else if (dataType() == DataType.UTF8) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asLongPointer(); //new LongPointer(length());
            setIndexer(LongIndexer.create((LongPointer) pointer));
        }

        // storing pointer into native DataBuffer
        ptrDataBuffer.setPrimaryBuffer(pointer, length);

        // adding deallocator reference
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
    }

    public BaseCpuDataBuffer(Pointer pointer, Indexer indexer, long length) {
        super(pointer, indexer, length);

        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(0, type, false);
        ptrDataBuffer.setPrimaryBuffer(this.pointer, length);
        Nd4j.getDeallocatorService().pickObject(this);;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(float[] data, boolean copy, long offset) {
        this(data, copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;

    }

    public BaseCpuDataBuffer(float[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        this(data, copy, workspace);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(float[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new FloatPointer(data);

        // creating & registering native DataBuffer
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(data.length, DataType.FLOAT, false);
        ptrDataBuffer.setPrimaryBuffer(pointer, data.length);
        Nd4j.getDeallocatorService().pickObject(this);

        setIndexer(FloatIndexer.create((FloatPointer) pointer));
        //wrappedBuffer = pointer.asByteBuffer();

        length = data.length;
        underlyingLength = data.length;
    }

    public BaseCpuDataBuffer(float[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asFloatPointer().put(data);

        this.ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(0, dataType(), false);
        this.ptrDataBuffer.setPrimaryBuffer(pointer, this.length);
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
        setIndexer(FloatIndexer.create((FloatPointer) pointer));
        //wrappedBuffer = pointer.asByteBuffer();
    }

    public BaseCpuDataBuffer(double[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asDoublePointer().put(data);

        this.ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(0, dataType(), false);
        this.ptrDataBuffer.setPrimaryBuffer(pointer, this.length);
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
        indexer = DoubleIndexer.create((DoublePointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }


    public BaseCpuDataBuffer(int[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asIntPointer().put(data);

        this.ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(0, dataType(), false);
        this.ptrDataBuffer.setPrimaryBuffer(pointer, this.length);
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
        indexer = IntIndexer.create((IntPointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }

    public BaseCpuDataBuffer(long[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asLongPointer().put(data);

        this.ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(0, dataType(), false);
        this.ptrDataBuffer.setPrimaryBuffer(pointer, this.length);
        Nd4j.getDeallocatorService().pickObject(this);

        workspaceGenerationId = workspace.getGenerationId();
        indexer = LongIndexer.create((LongPointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }


    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(double[] data, boolean copy, long offset) {
        this(data, copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.underlyingLength = data.length;
        this.length = underlyingLength - offset;
    }

    public BaseCpuDataBuffer(double[] data, boolean copy, long offset, MemoryWorkspace workspace) {
        this(data, copy, workspace);
        this.offset = offset;
        this.originalOffset = offset;
        this.underlyingLength = data.length;
        this.length = underlyingLength - offset;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(double[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new DoublePointer(data);
        indexer = DoubleIndexer.create((DoublePointer) pointer);

        // creating & registering native DataBuffer
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(data.length, DataType.DOUBLE, false);
        ptrDataBuffer.setPrimaryBuffer(pointer, data.length);
        Nd4j.getDeallocatorService().pickObject(this);

        length = data.length;
        underlyingLength = data.length;
    }


    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(int[] data, boolean copy, long offset) {
        this(data, copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(int[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new IntPointer(data);
        setIndexer(IntIndexer.create((IntPointer) pointer));

        // creating & registering native DataBuffer
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(data.length, DataType.INT32, false);
        ptrDataBuffer.setPrimaryBuffer(pointer, data.length);
        Nd4j.getDeallocatorService().pickObject(this);

        length = data.length;
        underlyingLength = data.length;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseCpuDataBuffer(long[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new LongPointer(data);
        setIndexer(LongIndexer.create((LongPointer) pointer));

        // creating & registering native DataBuffer
        ptrDataBuffer = OpaqueDataBuffer.allocateDataBuffer(data.length, DataType.INT64, false);
        ptrDataBuffer.setPrimaryBuffer(pointer, data.length);
        Nd4j.getDeallocatorService().pickObject(this);

        length = data.length;
        underlyingLength = data.length;
    }


    /**
     *
     * @param data
     */
    public BaseCpuDataBuffer(double[] data) {
        this(data, true);
    }

    /**
     *
     * @param data
     */
    public BaseCpuDataBuffer(int[] data) {
        this(data, true);
    }

    /**
     *
     * @param data
     */
    public BaseCpuDataBuffer(float[] data) {
        this(data, true);
    }

    public BaseCpuDataBuffer(float[] data, MemoryWorkspace workspace) {
        this(data, true, workspace);
    }

    @Override
    protected void release() {
        ptrDataBuffer.closeBuffer();
        super.release();
    }

    /**
     * Reallocate the native memory of the buffer
     * @param length the new length of the buffer
     * @return this databuffer
     * */
    @Override
    public DataBuffer reallocate(long length) {
        val oldPointer = ptrDataBuffer.primaryBuffer();

        if (isAttached()) {
            val capacity = length * getElementSize();
            val nPtr = getParentWorkspace().alloc(capacity, dataType(), false);
            this.ptrDataBuffer.setPrimaryBuffer(nPtr, length);

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

            Pointer.memcpy(pointer, oldPointer, this.length() * getElementSize());
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
    public void syncToPrimary(){
        ptrDataBuffer.syncToPrimary();
    }

    @Override
    public void syncToSpecial(){
        ptrDataBuffer.syncToSpecial();
    }
}
