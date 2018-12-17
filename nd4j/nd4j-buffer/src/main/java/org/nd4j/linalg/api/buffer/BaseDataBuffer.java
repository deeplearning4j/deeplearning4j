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

package org.nd4j.linalg.api.buffer;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.util.AllocUtil;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.primitives.AtomicDouble;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.*;
import java.nio.*;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;


/**
 * Base class for a data buffer
 * handling basic byte operations
 * among other things.
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseDataBuffer implements DataBuffer {

    /**
     * @deprecated Use {@link ND4JSystemProperties#DATABUFFER_TO_STRING_MAX_ELEMENTS}
     */
    public static String TO_STRING_MAX_ELEMENTS = ND4JSystemProperties.DATABUFFER_TO_STRING_MAX_ELEMENTS;
    private static int TO_STRING_MAX;
    static {
        String s = System.getProperty(ND4JSystemProperties.DATABUFFER_TO_STRING_MAX_ELEMENTS);
        if(s != null ){
            try {
                TO_STRING_MAX = Integer.parseInt(s);
            } catch (NumberFormatException e){
                log.warn("Invalid value for key {}: \"{}\"", ND4JSystemProperties.DATABUFFER_TO_STRING_MAX_ELEMENTS, s);
                TO_STRING_MAX = 1000;
            }
        } else {
            TO_STRING_MAX = 1000;
        }
    }

    protected DataType type;
    protected long length;
    protected long underlyingLength;
    protected long offset;
    protected byte elementSize;
    //protected transient ByteBuffer wrappedBuffer;
    protected transient DataBuffer wrappedDataBuffer;
    protected transient long workspaceGenerationId = 0L;

    //protected Collection<String> referencing = Collections.synchronizedSet(new HashSet<String>());
    //protected boolean isPersist = false;
    protected AllocationMode allocationMode;
    protected transient Pointer pointer;
    protected transient Indexer indexer;
    //protected AtomicBoolean dirty = new AtomicBoolean(false);

    protected transient boolean attached = false;
    protected transient MemoryWorkspace parentWorkspace;

    // Allocator-related stuff. Moved down here to avoid opType casting.
    protected transient DataBuffer originalBuffer;
    protected transient long originalOffset = 0;
    protected transient Long trackingPoint;

    protected transient boolean constant = false;
    protected transient boolean released = false;

    protected transient AtomicBoolean referenced = new AtomicBoolean(false);
    protected transient Collection<BaseDataBuffer> references = new ArrayList<>();

    public BaseDataBuffer() {}

    /**
     * Initialize the opType of this buffer
     */
    protected abstract void initTypeAndSize();

    @Override
    public int getElementSize() {
        return elementSize;
    }


    @Override
    public long getGenerationId() {
        if(parentWorkspace != null){
            return workspaceGenerationId;
        } else if(wrappedDataBuffer != null && wrappedDataBuffer.isAttached()){
            return wrappedDataBuffer.getGenerationId();
        } else if(originalBuffer != null && originalBuffer.isAttached()){
            return originalBuffer.getGenerationId();
        }
        return workspaceGenerationId;
    }

    /**
     *
     * Meant for creating another view of a buffer
     * @param pointer the underlying buffer to create a view from
     * @param indexer the indexer for the pointer
     * @param length the length of the view
     */
    public BaseDataBuffer(Pointer pointer, Indexer indexer, long length) {
        if (length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        this.length = length;
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
        this.underlyingLength = length;
        this.wrappedDataBuffer = this;

        this.pointer = pointer;
        setIndexer(indexer);
    }


    protected void setIndexer(Indexer indexer) {
        this.indexer = indexer;
    }

    /**
     *
     * Meant for creating another view of a buffer
     * @param underlyingBuffer the underlying buffer to create a view from
     * @param length the length of the view
     * @param offset the offset for the view
     */
    protected BaseDataBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        if (length < 0)
            throw new IllegalArgumentException("Length must be >= 1");

        if (length == 0)
            length = 1;



        initTypeAndSize();
        this.length = length;
        this.offset = offset;
        this.allocationMode = underlyingBuffer.allocationMode();
        this.elementSize = (byte) underlyingBuffer.getElementSize();
        this.underlyingLength = underlyingBuffer.underlyingLength();
        this.wrappedDataBuffer = underlyingBuffer;
        ((BaseDataBuffer) underlyingBuffer).referenced.compareAndSet(false, true);
        ((BaseDataBuffer) underlyingBuffer).references.add(this);

        // Adding link to original databuffer
        if (underlyingBuffer.originalDataBuffer() == null) {
            this.originalBuffer = underlyingBuffer;
            this.originalOffset = offset;
        } else {

            this.originalBuffer = underlyingBuffer.originalDataBuffer();

            // FIXME: please don't remove this comment, since there's probably a bug in current offset() impl,
            // and this line will change originalOffset accroding to proper offset() impl
            // FIXME: raver119@gmail.com
            this.originalOffset = offset; // + underlyingBuffer.originalOffset();
        }


        pointer = underlyingBuffer.pointer();
        setIndexer(underlyingBuffer.indexer());
    }

    /**
     * Original DataBuffer.
     * In case if we have a view derived from another view, derived from some other view, original DataBuffer will point to the originating DataBuffer, where all views come from.
     */
    @Override
    public DataBuffer originalDataBuffer() {
        return originalBuffer;
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(float[] data, boolean copy, long offset) {
        this(data, copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = data.length - offset;
        this.underlyingLength = data.length;

    }

    public BaseDataBuffer(float[] data, boolean copy, long offset, MemoryWorkspace workspace) {
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
    public BaseDataBuffer(float[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new FloatPointer(data);

        setIndexer(FloatIndexer.create((FloatPointer) pointer));
        //wrappedBuffer = pointer.asByteBuffer();

        length = data.length;
        underlyingLength = data.length;
    }

    public BaseDataBuffer(float[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asFloatPointer().put(data);
        workspaceGenerationId = workspace.getGenerationId();
        setIndexer(FloatIndexer.create((FloatPointer) pointer));
        //wrappedBuffer = pointer.asByteBuffer();
    }

    public BaseDataBuffer(double[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asDoublePointer().put(data);
        workspaceGenerationId = workspace.getGenerationId();
        indexer = DoubleIndexer.create((DoublePointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }


    public BaseDataBuffer(int[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asIntPointer().put(data);
        workspaceGenerationId = workspace.getGenerationId();
        indexer = IntIndexer.create((IntPointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }

    public BaseDataBuffer(long[] data, boolean copy, MemoryWorkspace workspace) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        length = data.length;
        underlyingLength = data.length;
        attached = true;
        parentWorkspace = workspace;

        initTypeAndSize();

        //log.info("Allocating FloatPointer from array of {} elements", data.length);

        pointer = workspace.alloc(data.length * getElementSize(), dataType(), false).asLongPointer().put(data);
        workspaceGenerationId = workspace.getGenerationId();
        indexer = LongIndexer.create((LongPointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();
    }


    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(double[] data, boolean copy, long offset) {
        this(data, copy);
        this.offset = offset;
        this.originalOffset = offset;
        this.underlyingLength = data.length;
        this.length = underlyingLength - offset;
    }

    public BaseDataBuffer(double[] data, boolean copy, long offset, MemoryWorkspace workspace) {
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
    public BaseDataBuffer(double[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new DoublePointer(data);
        indexer = DoubleIndexer.create((DoublePointer) pointer);
        //wrappedBuffer = pointer.asByteBuffer();

        length = data.length;
        underlyingLength = data.length;
    }


    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(int[] data, boolean copy, long offset) {
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
    public BaseDataBuffer(int[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new IntPointer(data);
        setIndexer(IntIndexer.create((IntPointer) pointer));

        length = data.length;
        underlyingLength = data.length;

        // // log.info("Creating new buffer of size: {}; dtype: {}; B", data.length, dataType());
    }

    /**
     *
     * @param data
     * @param copy
     */
    public BaseDataBuffer(long[] data, boolean copy) {
        allocationMode = AllocUtil.getAllocationModeFromContext();
        initTypeAndSize();

        pointer = new LongPointer(data);
        setIndexer(LongIndexer.create((LongPointer) pointer));

        length = data.length;
        underlyingLength = data.length;
    }

    /**
     *
     * @param data
     */
    public BaseDataBuffer(double[] data) {
        this(data, true);
    }

    /**
     *
     * @param data
     */
    public BaseDataBuffer(int[] data) {
        this(data, true);
    }

    /**
     *
     * @param data
     */
    public BaseDataBuffer(float[] data) {
        this(data, true);
    }

    public BaseDataBuffer(float[] data, MemoryWorkspace workspace) {
        this(data, true, workspace);
    }

    /**
     *
     * @param length
     * @param elementSize
     */
    public BaseDataBuffer(int length, int elementSize, long offset) {
        this(length, elementSize);
        this.offset = offset;
        this.originalOffset = offset;
        this.length = length - offset;
        this.underlyingLength = length;
    }

    /**
     *
     * @param length
     * @param elementSize
     */
    public BaseDataBuffer(long length, int elementSize) {
        if (length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        allocationMode = AllocUtil.getAllocationModeFromContext();
        this.length = length;
        this.underlyingLength = length;
        this.elementSize = (byte) elementSize;

        if (dataType() == DataType.DOUBLE) {
            pointer = new DoublePointer(length);
            indexer = DoubleIndexer.create((DoublePointer) pointer);
        } else if (dataType() == DataType.FLOAT) {
            pointer = new FloatPointer(length);
            setIndexer(FloatIndexer.create((FloatPointer) pointer));
        } else if (dataType() == DataType.INT) {
            pointer = new IntPointer(length);
            setIndexer(IntIndexer.create((IntPointer) pointer));
        } else if (dataType() == DataType.LONG) {
            pointer = new LongPointer(length);
            setIndexer(LongIndexer.create((LongPointer) pointer));
        } else if (dataType() == DataType.SHORT) {
            pointer = new ShortPointer(length);
            setIndexer(ShortIndexer.create((ShortPointer) pointer));
        } else if (dataType() == DataType.BYTE) {
            pointer = new BytePointer(length);
            setIndexer(ByteIndexer.create((BytePointer) pointer));
        } else if (dataType() == DataType.UBYTE) {
            pointer = new BytePointer(length);
            setIndexer(UByteIndexer.create((BytePointer) pointer));
        } else if (dataType() == DataType.UTF8) {
            pointer = new LongPointer(length);
            setIndexer(LongIndexer.create((LongPointer) pointer));
        }

        // log.info("Creating new buffer of size: {}; dtype: {}; C", length, dataType());
    }

    /**
     * Create a data buffer from
     * the given length
     *
     * @param buffer
     * @param length
     */
    public BaseDataBuffer(ByteBuffer buffer, long length, long offset) {
        this(buffer, length);
        this.offset = offset;
        this.originalOffset = offset;
        this.underlyingLength = length;
        this.length = length - offset;

    }

    /**
     * Create a data buffer from
     * the given length
     *
     * @param buffer
     * @param length
     */
    public BaseDataBuffer(ByteBuffer buffer, long length) {
        if (length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();

        this.length = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();

        if (dataType() == DataType.DOUBLE) {
            pointer = new DoublePointer(buffer.asDoubleBuffer());
            setIndexer(DoubleIndexer.create((DoublePointer) pointer));
        } else if (dataType() == DataType.FLOAT) {
            pointer = new FloatPointer(buffer.asFloatBuffer());
            setIndexer(FloatIndexer.create((FloatPointer) pointer));
        } else if (dataType() == DataType.INT) {
            pointer = new IntPointer(buffer.asIntBuffer());
            setIndexer(IntIndexer.create((IntPointer) pointer));
        } else if (dataType() == DataType.LONG) {
            pointer = new LongPointer(buffer.asLongBuffer());
            setIndexer(LongIndexer.create((LongPointer) pointer));
        }

        // log.info("Creating new buffer of size: {}; dtype: {}; D", length, dataType());
    }

    //sets the nio wrapped buffer (allows to be overridden for other use cases like cuda)
    protected void setNioBuffer() {
        if (elementSize * length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create buffer of length " + length);
        //wrappedBuffer = pointer().asByteBuffer();

    }


    /**
     *
     * @param data
     * @param length
     */
    public BaseDataBuffer(byte[] data, long length) {
        this(ByteBuffer.wrap(data), length);
    }


    /**
     * Returns the indexer for the buffer
     *
     * @return
     */
    @Override
    public Indexer indexer() {
        return indexer;
    }

    @Override
    public Pointer pointer() {
        return underlyingDataBuffer() != null && underlyingDataBuffer() != this ? underlyingDataBuffer().pointer()
                        : pointer;
    }

    @Override
    public DataBuffer underlyingDataBuffer() {
        return wrappedDataBuffer;
    }

    @Override
    public long offset() {
        return offset;
    }

    @Override
    public AllocationMode allocationMode() {
        return allocationMode;
    }

    @Override
    @Deprecated
    public void persist() {
        //isPersist = true;
        throw new UnsupportedOperationException();
    }

    @Override
    @Deprecated
    public boolean isPersist() {
        throw new UnsupportedOperationException();
    }

    @Override
    @Deprecated
    public void unPersist() {
        throw new UnsupportedOperationException();
    }

    private void fillPointerWithZero() {
        Pointer.memset(this.pointer(), 0, getElementSize() * length());
    }

    /**
     * Instantiate a buffer with the given length
     *
     * @param length the length of the buffer
     */
    protected BaseDataBuffer(long length) {
        this(length, true);
    }

    protected BaseDataBuffer(long length, boolean initialize) {
        if (length < 0)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();
        if (length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

        if (dataType() == DataType.DOUBLE) {
            pointer = new DoublePointer(length());
            indexer = DoubleIndexer.create((DoublePointer) pointer);
            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.FLOAT) {
            pointer = new FloatPointer(length());
            setIndexer(FloatIndexer.create((FloatPointer) pointer));

            if (initialize)
                fillPointerWithZero();

        } else if (dataType() == DataType.HALF) {
            pointer = new ShortPointer(length());
            setIndexer(HalfIndexer.create((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();

        } else if (dataType() == DataType.INT) {
            pointer = new IntPointer(length());
            setIndexer(IntIndexer.create((IntPointer) pointer));
            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.LONG) {
            pointer = new LongPointer(length());
            setIndexer(LongIndexer.create((LongPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.BYTE) {
            pointer = new BytePointer(length());
            setIndexer(ByteIndexer.create((BytePointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.SHORT) {
            pointer = new ShortPointer(length());
            setIndexer(ShortIndexer.create((ShortPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UBYTE) {
            pointer = new BytePointer(length());
            setIndexer(UByteIndexer.create((BytePointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.BOOL) {
            pointer = new BooleanPointer(length());
            setIndexer(BooleanIndexer.create((BooleanPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        } else if (dataType() == DataType.UTF8) {
            pointer = new LongPointer(length());
            setIndexer(LongIndexer.create((LongPointer) pointer));

            if (initialize)
                fillPointerWithZero();
        }

        //// log.info("Creating new buffer of size: {}; dtype: {}; A", length, dataType());
    }

    protected BaseDataBuffer(long length, boolean initialize, MemoryWorkspace workspace) {
        if (length < 1)
            throw new IllegalArgumentException("Length must be >= 1");
        initTypeAndSize();
        this.length = length;
        this.underlyingLength = length;
        allocationMode = AllocUtil.getAllocationModeFromContext();



        if (length < 0)
            throw new IllegalArgumentException("Unable to create a buffer of length <= 0");

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

        } else if (dataType() == DataType.INT) {
            attached = true;
            parentWorkspace = workspace;

            pointer = workspace.alloc(length * getElementSize(), dataType(), initialize).asIntPointer(); //new IntPointer(length());
            setIndexer(IntIndexer.create((IntPointer) pointer));

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

        workspaceGenerationId = workspace.getGenerationId();

    }

    @Override
    public void copyAtStride(DataBuffer buf, long n, long stride, long yStride, long offset, long yOffset) {
        if (dataType() == DataType.FLOAT) {
            for (int i = 0; i < n; i++) {
                put(offset + i * stride, buf.getFloat(yOffset + i * yStride));
            }
        } else {
            for (int i = 0; i < n; i++) {
                put(offset + i * stride, buf.getDouble(yOffset + i * yStride));
            }
        }

    }

    @Override
    @Deprecated
    public void removeReferencing(String id) {
        //referencing.remove(id);
    }

    @Override
    @Deprecated
    public Collection<String> references() {
        throw new UnsupportedOperationException();
        //return referencing;
    }

    @Override
    public Pointer addressPointer() {

        if (offset() > 0) {
            Pointer ret;
            final long retAddress = pointer().address() + getElementSize() * offset();
            // directly set address at construction since Pointer.address has not setter.
            if (dataType() == DataType.DOUBLE) {
                ret = new DoublePointer(pointer()) {
                    {
                        address = retAddress;
                    }
                };
            } else if (dataType() == DataType.FLOAT) {
                ret = new FloatPointer(pointer()) {
                    {
                        address = retAddress;
                    }
                };
            } else if (dataType() == DataType.INT) {
                ret = new IntPointer(pointer()) {
                    {
                        address = retAddress;
                    }
                };
            } else if (dataType() == DataType.LONG) {
                ret = new LongPointer(pointer()) {
                    {
                        address = retAddress;
                    }
                };
            } else {
                ret = new Pointer(pointer()) {
                    {
                        address = retAddress;
                    }
                };
            }
            ret.limit(ret.limit() - offset());
            ret.capacity(ret.capacity() - offset());
            return ret;
        }
        return pointer();
    }

    @Override
    public long address() {
        return pointer().address() + getElementSize() * offset();
    }

    @Override
    @Deprecated
    public void addReferencing(String id) {
        //referencing.add(id);
    }

    @Override
    public void assign(long[] indices, float[] data, boolean contiguous, long inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length "
                            + length() + " where the indices are of length " + data.length);
        for (int i = 0; i < indices.length; i++) {
            put(indices[i], data[i]);
        }
    }



    @Override
    public void setData(int[] data) {
        for (int i = 0; i < data.length; i++) {
            put(i, data[i]);
        }
    }

    @Override
    public void setData(float[] data) {
        for (int i = 0; i < data.length; i++) {
            put(i, data[i]);
        }
    }

    @Override
    public void setData(double[] data) {
        for (int i = 0; i < data.length; i++) {
            put(i, data[i]);
        }
    }

    @Override
    public void setData(long[] data) {
        for (int i = 0; i < data.length; i++) {
            put(i, data[i]);
        }
    }

    @Override
    public void setData(byte[] data) {
        for (int i = 0; i < data.length; i++) {
            put(i, data[i]);
        }
    }

    @Override
    public void setData(short[] data) {
        for (int i = 0; i < data.length; i++) {
            put(i, data[i]);
        }
    }

    @Override
    public void setData(boolean[] data) {
        for (int i = 0; i < data.length; i++) {
            put(i, data[i]);
        }
    }

    @Override
    public void assign(long[] indices, double[] data, boolean contiguous, long inc) {
        if (indices.length != data.length)
            throw new IllegalArgumentException("Indices and data length must be the same");
        if (indices.length > length())
            throw new IllegalArgumentException("More elements than space to assign. This buffer is of length "
                            + length() + " where the indices are of length " + data.length);
        for (int i = 0; i < indices.length; i += inc) {
            put(indices[i], data[i]);
        }
    }

    @Override
    public void assign(DataBuffer data) {
        if (data.length() != length())
            throw new IllegalArgumentException("Unable to assign buffer of length " + data.length()
                            + " to this buffer of length " + length());

        for (int i = 0; i < data.length(); i++) {
            put(i, data.getDouble(i));
        }
    }

    @Override
    public void assign(long[] indices, float[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }

    @Override
    public void assign(long[] indices, double[] data, boolean contiguous) {
        assign(indices, data, contiguous, 1);
    }

    @Override
    public long underlyingLength() {
        return underlyingLength;
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public void assign(Number value) {
        assign(value, 0);
    }

    @Override
    public double[] getDoublesAt(long offset, long inc, int length) {
        if (offset + length > length())
            length -= offset;

        double[] ret = new double[length];
        for (int i = 0; i < length; i++) {
            ret[i] = getDouble(i * inc + offset);
        }
        return ret;
    }

    @Override
    public double[] getDoublesAt(long offset, int length) {
        return getDoublesAt(offset, 1, length);
    }

    @Override
    public float[] getFloatsAt(long offset, int length) {
        return getFloatsAt(offset, 1, length);
    }

    @Override
    public float[] getFloatsAt(long offset, long inc, int length) {
        if (offset + length > length())
            length -= offset;
        float[] ret = new float[length];
        for (int i = 0; i < length; i ++) {
            ret[i] = getFloat(i * inc + offset);
        }
        return ret;
    }

    @Override
    public long[] getLongsAt(long offset, int length) {
        return getLongsAt(offset, 1, length);
    }

    @Override
    public long[] getLongsAt(long offset, long inc, int length) {
        if (offset + length > length())
            length -= offset;
        long[] ret = new long[length];
        for (int i = 0; i < length; i++) {
            ret[i] = getLong(i * inc + offset);
        }
        return ret;
    }

    @Override
    public int[] getIntsAt(long offset, int length) {
        return getIntsAt(offset, 1, length);
    }

    @Override
    public int[] getIntsAt(long offset, long inc, int length) {
        if (offset + length > length())
            length -= offset;
        int[] ret = new int[length];
        for (int i = 0; i < length; i++) {
            ret[i] = getInt(i * inc + offset);
        }
        return ret;
    }

    @Override
    public DataBuffer dup() {
        DataBuffer ret = create(length);
        for (int i = 0; i < ret.length(); i++)
            ret.put(i, getDouble(i));

        return ret;
    }



    /**
     * Create with length
     * @param length a databuffer of the same opType as
     *               this with the given length
     * @return a data buffer with the same length and datatype as this one
     */
    protected abstract DataBuffer create(long length);


    /**
     * Create the data buffer
     * with respect to the given byte buffer
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    public abstract DataBuffer create(double[] data);

    /**
     * Create the data buffer
     * with respect to the given byte buffer
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    public abstract DataBuffer create(float[] data);

    /**
     * Create the data buffer
     * with respect to the given byte buffer
     * @param data the buffer to create
     * @return the data buffer based on the given buffer
     */
    public abstract DataBuffer create(int[] data);


    @Override
    public void assign(long[] offsets, long[] strides, DataBuffer... buffers) {
        assign(offsets, strides, length(), buffers);
    }

    @Override
    public byte[] asBytes() {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        val dataType = dataType();
        switch (dataType) {
            case DOUBLE:
                    for (int i = 0; i < length(); i++) {
                        try {
                            dos.writeDouble(getDouble(i));
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                    break;
            case FLOAT:
                    for (int i = 0; i < length(); i++) {
                        try {
                            dos.writeFloat(getFloat(i));
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                    break;
            case HALF:
                for (int i = 0; i < length(); i++) {
                    try {
                        dos.writeShort(HalfIndexer.fromFloat(getFloat(i)));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
            case BOOL:
                for (int i = 0; i < length(); i++) {
                    try {
                        dos.writeByte(getInt(i) == 0 ? (byte) 0 : (byte) 1);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
            case BYTE:
                for (int i = 0; i < length(); i++) {
                    try {
                        dos.writeByte((byte) getShort(i));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
            case UBYTE:
                for (int i = 0; i < length(); i++) {
                    //try {
                        throw new UnsupportedOperationException();
                        //dos.writeByte(getShort(i));
                    //} catch (IOException e) {
                    //    e.printStackTrace();
                    //}
                }
                break;
            case SHORT:
                    for (int i = 0; i < length(); i++) {
                        try {
                            dos.writeShort(getShort(i));
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                    break;
            case INT:
                for (int i = 0; i < length(); i++) {
                    try {
                        dos.writeInt(getInt(i));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
            case LONG:
                for (int i = 0; i < length(); i++) {
                    try {
                        dos.writeLong(getLong(i));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
                default:
                    throw new UnsupportedOperationException("Unknown data type: [" + dataType + "]");
        }
        return bos.toByteArray();
    }

    @Override
    public float[] asFloat() {
        if (length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        float[] ret = new float[(int) length];
        for (int i = 0; i < length; i++)
            ret[i] = getFloat(i);
        return ret;
    }

    @Override
    public double[] asDouble() {
        if (length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        double[] ret = new double[(int) length];
        for (int i = 0; i < length; i++)
            ret[i] = getDouble(i);
        return ret;
    }

    @Override
    public int[] asInt() {
        if (length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        int[] ret = new int[(int) length];
        for (int i = 0; i < length; i++)
            ret[i] = getInt(i);
        return ret;
    }

    @Override
    public long[] asLong() {
        if (length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        long[] ret = new long[(int) length];
        for (int i = 0; i < length; i++)
            ret[i] = getLong(i);
        return ret;
    }

    @Override
    public double getDouble(long i) {
        if (indexer == null) {
            throw new IllegalStateException("Indexer must never be null");
        }
        switch (dataType()) {
            case FLOAT:
                return ((FloatIndexer) indexer).get(offset() + i);
            case INT:
                return ((IntIndexer) indexer).get(offset() + i);
            case HALF:
                return ((HalfIndexer) indexer).get(offset() + i);
            case SHORT:
                return ((ShortIndexer) indexer).get(offset() + i);
            case LONG:
                return ((LongIndexer) indexer).get(offset() + i);
            case BOOL:
                return ((BooleanIndexer) indexer).get(offset() + i) ? 1.0 : 0.0;
            case DOUBLE:
                return ((DoubleIndexer) indexer).get(offset() + i);
            case BYTE:
                return ((ByteIndexer) indexer).get(offset() + i);
            case UBYTE:
                return ((UByteIndexer) indexer).get(offset() + i);
            default:
                throw new UnsupportedOperationException();
        }
    }

    @Override
    public long getLong(long i) {
        switch (dataType()) {
            case FLOAT:
                return (long) ((FloatIndexer) indexer).get(offset() + i);
            case DOUBLE:
                return (long) ((DoubleIndexer) indexer).get(offset() + i);
            case HALF:
                return (long) ((HalfIndexer) indexer).get(offset() + i);
            case LONG:
                return ((LongIndexer) indexer).get(offset() + i);
            case INT:
                return (long) ((IntIndexer) indexer).get(offset() + i);
            case SHORT:
                return (long) ((ShortIndexer) indexer).get(offset() + i);
            case BYTE:
                return (long) ((ByteIndexer) indexer).get(offset() + i);
            case UBYTE:
                return (long) ((UByteIndexer) indexer).get(offset() + i);
            case BOOL:
                return  ((BooleanIndexer) indexer).get(offset() + i) ? 1L : 0L;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }

    /**
     * Special method for
     * @param i
     * @return
     */
    protected short getShort(long i) {
        switch (dataType()) {
            case DOUBLE:
                return (short) ((DoubleIndexer) indexer).get(offset() + i);
            case BOOL:
                return (short) (((BooleanIndexer) indexer).get(offset() + i) ? 1 : 0);
            case INT:
                return (short) ((IntIndexer) indexer).get(offset() + i);
            case SHORT:
                return ((ShortIndexer) indexer).get(offset() + i);
            case BYTE:
                return  (short) ((ByteIndexer) indexer).get(offset() + i);
            case LONG:
                return (short) ((LongIndexer) indexer).get(offset() + i);
            case FLOAT:
                return (short) ((FloatIndexer) indexer).get(offset() + i);
            default:
                throw new UnsupportedOperationException();
        }
    }

    /**
     *
     * @param v
     * @return
     */
    public static short fromFloat(float v) {
        return ArrayUtil.fromFloat(v);        
    }

    @Override
    public float getFloat(long i) {
        switch (dataType()) {
            case DOUBLE:
                return (float) ((DoubleIndexer) indexer).get(offset() + i);
            case BOOL:
                return ((BooleanIndexer) indexer).get(offset() + i) ? 1.f : 0.f;
            case INT:
                return (float) ((IntIndexer) indexer).get(offset() + i);
            case SHORT:
                return (float) ((ShortIndexer) indexer).get(offset() + i);
            case HALF:
                return (float) ((HalfIndexer) indexer).get(offset() + i);
            case UBYTE:
                return (float) ((UByteIndexer) indexer).get(offset() + i);
            case BYTE:
                return (float) ((ByteIndexer) indexer).get(offset() + i);
            case LONG:
                return (float)  ((LongIndexer) indexer).get(offset() + i);
            case FLOAT:
                return ((FloatIndexer) indexer).get(offset() + i);
            default:
                throw new UnsupportedOperationException();
        }
    }

    @Override
    public int getInt(long i) {
        switch (dataType()) {
            case DOUBLE:
                return (int) ((DoubleIndexer) indexer).get(offset() + i);
            case BOOL:
                return ((BooleanIndexer) indexer).get(offset() + i) ? 1 : 0;
            case INT:
                return ((IntIndexer) indexer).get(offset() + i);
            case HALF:
                return (int) ((HalfIndexer) indexer).get(offset() + i);
            case SHORT:
                return ((ShortIndexer) indexer).get(offset() + i);
            case UBYTE:
                return ((UByteIndexer) indexer).get(offset() + i);
            case BYTE:
                return ((ByteIndexer) indexer).get(offset() + i);
            case LONG:
                return (int) ((LongIndexer) indexer).get(offset() + i);
            case FLOAT:
                return (int) ((FloatIndexer) indexer).get(offset() + i);
            default:
                throw new UnsupportedOperationException();
        }
    }

    @Override
    public Number getNumber(long i) {
        if (dataType() == DataType.DOUBLE)
            return getDouble(i);
        else if (dataType() == DataType.INT)
            return getInt(i);
        else if (dataType() == DataType.LONG)
            return getLong(i);
        return getFloat(i);
    }

    public void pointerIndexerByCurrentType(DataType currentType) {
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
    }

    public void putByDestinationType(long i, Number element, DataType globalType) {
        if (globalType == DataType.INT || type == DataType.INT) {
            int anElement = element.intValue();
            put(i, anElement);
        } else if (globalType == DataType.LONG || type == DataType.LONG) {
            long anElement = element.longValue();
            put(i, anElement);
        } else if (globalType == DataType.FLOAT || globalType == DataType.HALF) {
            float anElement = element.floatValue();
            put(i, anElement);
        } else if (globalType == DataType.DOUBLE) {
            double anElement = element.doubleValue();
            put(i, anElement);
        } else {
            throw new IllegalStateException("Unknown type: " + globalType);
        }
    }

    @Override
    public void put(long i, float element) {
        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(offset() + i, element == 0.0 ? false : true);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(offset() + i, (byte) element);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(offset() + i,  (int) element);
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(offset() + i,  (short) element);
                break;
            case INT:
                ((IntIndexer) indexer).put(offset() + i, (int) element);
                break;
            case LONG:
                ((LongIndexer) indexer).put(offset() + i, (long) element);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(offset() + i,  element);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(offset() + i, element);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(offset() + i, element);
                break;
            default:
                throw new IllegalStateException("Unsupported type: " + dataType());
        }

        if (i == length) {
            length++;
        }
    }

    @Override
    public void put(long i, double element) {
        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(offset() + i,  element > 0.0);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(offset() + i, (byte) element);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(offset() + i, (short) element);
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(offset() + i,  (short) element);
                break;
            case INT:
                ((IntIndexer) indexer).put(offset() + i, (int) element);
                break;
            case LONG:
                ((LongIndexer) indexer).put(offset() + i, (long) element);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(offset() + i, (float) element);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(offset() + i, (float) element);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(offset() + i, element);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }

        if (i == length) {
            length++;
        }
    }

    @Override
    public void put(long i, int element) {
        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(offset() + i, element == 0 ? false : true);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(offset() + i,  (byte) element);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(offset() + i,  element);
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(offset() + i,  (short) element);
                break;
            case INT:
                ((IntIndexer) indexer).put(offset() + i, element);
                break;
            case LONG:
                ((LongIndexer) indexer).put(offset() + i, element);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(offset() + i, element);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(offset() + i, element);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(offset() + i, element);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }

        if (i == length) {
            length++;
        }
    }

    @Override
    public void put(long i, boolean element) {
        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(offset() + i, element);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(offset() + i, element ? (byte)1 : (byte) 0);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(offset() + i, element ? (byte)1 : (byte) 0);
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(offset() + i, element ? (short) 1 : (short) 0);
                break;
            case INT:
                ((IntIndexer) indexer).put(offset() + i, element ? 1 : 0);
                break;
            case LONG:
                ((LongIndexer) indexer).put(offset() + i, element ? 1 : 0);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(offset() + i, element ? 1.0f : 0.0f);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(offset() + i, element ? 1.0f : 0.0f);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(offset() + i,  element ? 1.0 : 0.0);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }

        if (i == length) {
            length++;
        }
    }

    @Override
    public void put(long i, long element) {
        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(offset() + i, element == 0 ? false : true);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(offset() + i, (byte) element);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(offset() + i, (short) element);
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(offset() + i, (short) element);
                break;
            case INT:
                ((IntIndexer) indexer).put(offset() + i, (int) element);
                break;
            case LONG:
                ((LongIndexer) indexer).put(offset() + i, element);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(offset() + i, (float) element);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(offset() + i, (float) element);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(offset() + i, (double) element);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }

        if (i == length) {
            length++;
        }
    }

    @Override
    @Deprecated
    public boolean dirty() {
        return false;
    }

    @Override
    public boolean sameUnderlyingData(DataBuffer buffer) {
        return pointer() == buffer.pointer();
    }

    protected ByteBuffer wrappedBuffer() {
        return pointer().asByteBuffer();
    }

    @Override
    public IntBuffer asNioInt() {
        if (offset() >= Integer.MAX_VALUE)
            throw new IllegalStateException("Index out of bounds " + offset());

        if (offset() == 0) {
            return wrappedBuffer().asIntBuffer();
        } else
            return (IntBuffer) wrappedBuffer().asIntBuffer().position((int) offset());
    }

    @Override
    public LongBuffer asNioLong() {
        if (offset() >= Integer.MAX_VALUE)
            throw new IllegalStateException("Index out of bounds " + offset());

        if (offset() == 0) {
            return wrappedBuffer().asLongBuffer();
        } else
            return (LongBuffer) wrappedBuffer().asLongBuffer().position((int) offset());
    }

    @Override
    public DoubleBuffer asNioDouble() {
        if (offset() >= Integer.MAX_VALUE)
            throw new IllegalStateException("Index out of bounds " + offset());

        if (offset() == 0) {
            return wrappedBuffer().asDoubleBuffer();
        } else {
            return (DoubleBuffer) wrappedBuffer().asDoubleBuffer().position((int) (offset()));
        }
    }

    @Override
    public FloatBuffer asNioFloat() {
        if (offset() >= Integer.MAX_VALUE)
            throw new IllegalStateException("Index out of bounds " + offset());

        if (offset() == 0) {
            return wrappedBuffer().asFloatBuffer();
        } else {
            return (FloatBuffer) wrappedBuffer().asFloatBuffer().position((int) (offset()));
        }

    }

    @Override
    public ByteBuffer asNio() {
        return wrappedBuffer();
    }

    @Override
    public void assign(Number value, long offset) {
        //note here that the final put will take care of the offset
        for (long i = offset; i < length(); i++)
            put(i, value.doubleValue());
    }

    @Override
    public void write(OutputStream dos) {
        if (dos instanceof DataOutputStream) {
            try {
                write((DataOutputStream) dos);
            } catch (IOException e) {
                throw new IllegalStateException("IO Exception writing buffer", e);
            }
        } else {
            DataOutputStream dos2 = new DataOutputStream(dos);
            try {

                write(dos2);
            } catch (IOException e) {
                throw new IllegalStateException("IO Exception writing buffer", e);
            }
        }

    }

    @Override
    public void read(InputStream is, AllocationMode allocationMode, long length, DataType dataType) {
        if (is instanceof DataInputStream) {
            read((DataInputStream) is, allocationMode, length, dataType);
        }

        else {
            DataInputStream dis2 = new DataInputStream(is);
            read(dis2, allocationMode, length, dataType);
        }
    }

    @Override
    public void flush() {

    }

    @Override
    public void assign(long[] offsets, long[] strides, long n, DataBuffer... buffers) {
        if (offsets.length != strides.length || strides.length != buffers.length)
            throw new IllegalArgumentException(
                            "Unable to assign buffers, please specify equal lengths strides, offsets, and buffers");
        int count = 0;
        for (int i = 0; i < buffers.length; i++) {
            //note here that the final put will take care of the offset
            for (long j = offsets[i]; j < buffers[i].length(); j += strides[i]) {
                put(count++, buffers[i].getDouble(j));
            }
        }

        if (count != n)
            throw new IllegalArgumentException("Strides and offsets didn't match up to length " + n);

    }

    @Override
    public void assign(DataBuffer... buffers) {
        long[] offsets = new long[buffers.length];
        long[] strides = new long[buffers.length];
        for (int i = 0; i < strides.length; i++)
            strides[i] = 1;
        assign(offsets, strides, buffers);
    }


    @Override
    public void destroy() {

    }

    /**
     * The data opType of the buffer
     *
     * @return the data opType of the buffer
     */
    @Override
    public DataType dataType() {
        return type;
    }

    @Override
    public boolean equals(Object o) {
        // FIXME: this is BAD. it takes too long to work, and it breaks general equals contract
        if (o instanceof DataBuffer) {
            DataBuffer d = (DataBuffer) o;
            if (d.length() != length())
                return false;
            for (int i = 0; i < length(); i++) {
                double eps = Math.abs(getDouble(i) - d.getDouble(i));
                if (eps > 1e-12)
                    return false;
            }
        }

        return true;
    }

    private void readObject(ObjectInputStream s) {
        doReadObject(s);
    }

    private void writeObject(ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        write(out);
    }


    protected void doReadObject(ObjectInputStream s) {
        try {
            s.defaultReadObject();
            val header = BaseDataBuffer.readHeader(s);
            read(s, header.getLeft(), header.getMiddle(), header.getRight());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }

    public static Triple<AllocationMode, Long, DataType> readHeader(@NonNull InputStream is)  {
        try {
            DataInputStream dis = is instanceof DataInputStream ? (DataInputStream) is : new DataInputStream(is);
            val alloc = AllocationMode.valueOf(dis.readUTF());
            long length = 0;
            if (alloc.ordinal() < 3) {
                length = dis.readInt();
            } else {
                length = dis.readLong();
            }
            val type = DataType.valueOf(dis.readUTF());

            return Triple.tripleOf(alloc, length, type);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void read(DataInputStream s, @NonNull AllocationMode allocMode, long len, @NonNull DataType dtype) {
        try {
            //referencing = Collections.synchronizedSet(new HashSet<String>());
            val savedMode = allocMode;
            this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
            type = dtype;
            length = len;

            // old AllocationMode values are: DIRECT, HEAP, JAVACPP. Just using legacy here
            if (savedMode.ordinal() < 3) {
                //Do an implicit conversion: keep current buffer data type unchanged, and convert values from source type
                length = len;
                DataType sourceType = dtype;
                pointerIndexerByCurrentType(type);      //also updates indexer based on newly set length

                if (sourceType != DataType.COMPRESSED) {
                    DataType thisType = dataType();
                    readContent(s, sourceType, thisType);
                }

                // we should switch types here

                //wrappedBuffer = pointer().asByteBuffer();

            } else if (savedMode.equals(AllocationMode.LONG_SHAPE)) {
                length = len;
                val currentType = dtype;
                type = currentType;

                if (currentType == DataType.LONG)
                    elementSize = 8;
                else if (DataTypeUtil.getDtypeFromContext() == DataType.DOUBLE && currentType != DataType.INT)
                    elementSize = 8;
                else if (DataTypeUtil.getDtypeFromContext() == DataType.FLOAT || currentType == DataType.INT)
                    elementSize = 4;
                else if (DataTypeUtil.getDtypeFromContext() == DataType.HALF && currentType != DataType.INT)
                    elementSize = 2;

                if (currentType != DataTypeUtil.getDtypeFromContext() && currentType != DataType.HALF && currentType != DataType.INT
                        && currentType != DataType.LONG && !(DataTypeUtil.getDtypeFromContext() == DataType.DOUBLE)) {
                    log.warn("Loading a data stream with opType different from what is set globally. Expect precision loss");
                    if (DataTypeUtil.getDtypeFromContext() == DataType.INT)
                        log.warn("Int to float/double widening UNSUPPORTED!!!");
                }
                pointerIndexerByCurrentType(currentType);

                if (currentType != DataType.COMPRESSED)
                    readContent(s, currentType, currentType);
            } else if (allocationMode.equals(AllocationMode.MIXED_DATA_TYPES)) {
                switch (type) {
                    case LONG:
                    case DOUBLE:
                        elementSize = 8;
                        break;
                    case FLOAT:
                    case INT:
                        elementSize = 4;
                        break;
                    case SHORT:
                    case HALF:
                        elementSize = 2;
                        break;
                    case BOOL:
                    case BYTE:
                    case UBYTE:
                    case UTF8:
                        elementSize = 1;
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }

                pointerIndexerByCurrentType(type);

                if (type != DataType.COMPRESSED)
                    readContent(s, type, type);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    protected void readContent(DataInputStream s, DataType sourceType, DataType thisType) {
        try {
            //Use AtomicX as a mutable Number class to reduce garbage vs. auto boxing to Double/Float etc classes
            if (sourceType == DataType.DOUBLE) {
                AtomicDouble aDbl = new AtomicDouble();
                for (long i = 0; i < length(); i++) {
                    aDbl.set(s.readDouble());
                    putByDestinationType(i, aDbl, thisType);
                }
            } else if (sourceType == DataType.FLOAT) {
                //TODO no AtomicFloat to use here?
                for (long i = 0; i < length(); i++) {
                    putByDestinationType(i, s.readFloat(), thisType);
                }
            } else if (sourceType == DataType.COMPRESSED) {
                String compressionAlgorithm = s.readUTF();
                long compressedLength = s.readLong();
                long originalLength = s.readLong();
                long numberOfElements = s.readLong();

                pointer = new BytePointer(compressedLength);
                type = DataType.COMPRESSED;
                val tp = (BytePointer) pointer;
                val ti = ByteIndexer.create(tp);

                for (long i = 0; i < compressedLength; i++) {
                    ti.put(i, s.readByte());
                }

            } else if (sourceType == DataType.HALF) {
                AtomicInteger aInt = new AtomicInteger();
                for (long i = 0; i < length(); i++) {
                    aInt.set(s.readShort());
                    putByDestinationType(i, aInt, thisType);
                }
            } else if (sourceType == DataType.LONG) {
                AtomicLong aLong = new AtomicLong();
                for (long i = 0; i < length(); i++) {
                    aLong.set(s.readLong());
                    putByDestinationType(i, aLong, thisType);
                }
            } else if (sourceType == DataType.INT ){
                AtomicInteger aInt = new AtomicInteger();
                for (long i = 0; i < length(); i++) {
                    aInt.set(s.readInt());
                    putByDestinationType(i, aInt, thisType);
                }
            } else {
                throw new UnsupportedOperationException("Cannot read type: " + sourceType + " to " + thisType);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void write(DataOutputStream out) throws IOException {
        out.writeUTF(allocationMode.name());
        out.writeLong(length());
        out.writeUTF(dataType().name());
        switch (dataType()) {
            case DOUBLE:
                for (long i = 0; i < length(); i++)
                    out.writeDouble(getDouble(i));
                break;
            case LONG:
                for (long i = 0; i < length(); i++)
                    out.writeLong(getLong(i));
                break;
            case INT:
                for (long i = 0; i < length(); i++)
                    out.writeInt(getInt(i));
                break;
            case SHORT:
                for (long i = 0; i < length(); i++)
                    out.writeShort((short) getInt(i));
                break;
            case UBYTE:
            case BYTE:
                for (long i = 0; i < length(); i++)
                    out.writeByte((byte) getInt(i));
                break;
            case BOOL:
                for (long i = 0; i < length(); i++)
                    out.writeByte(getInt(i) == 0 ? (byte) 0 : (byte) 1);
                break;
            case HALF:
                for (long i = 0; i < length(); i++)
                    out.writeShort(getShort(i));
                break;
            case FLOAT:
                for (long i = 0; i < length(); i++)
                    out.writeFloat(getFloat(i));
                break;
        }
    }

    public float toFloat(int hbits) {
        int mant = hbits & 0x03ff; // 10 bits mantissa
        int exp = hbits & 0x7c00; // 5 bits exponent
        if (exp == 0x7c00) // NaN/Inf
            exp = 0x3fc00; // -> NaN/Inf
        else if (exp != 0) // normalized value
        {
            exp += 0x1c000; // exp - 15 + 127
            // "smooth transition" is nonstandard behavior
            //            if( mant == 0 && exp > 0x1c400 )  // smooth transition
            //                return Float.intBitsToFloat( ( hbits & 0x8000 ) << 16
            //                                                | exp << 13 | 0x3ff );
        } else if (mant != 0) // && exp==0 -> subnormal
        {
            exp = 0x1c400; // make it normal
            do {
                mant <<= 1; // mantissa * 2
                exp -= 0x400; // decrease exp by 1
            } while ((mant & 0x400) == 0); // while not normal
            mant &= 0x3ff; // discard subnormal bit
        } // else +/-0 -> +/-0
        return Float.intBitsToFloat( // combine all parts
                        (hbits & 0x8000) << 16 // sign  << ( 31 - 15 )
                                        | (exp | mant) << 13); // value << ( 23 - 10 )
    }


    @Override
    public Object array() {
        return null;
    }

    @Override
    public String toString() {
        StringBuilder ret = new StringBuilder();
        ret.append("[");

        int max;
        if (TO_STRING_MAX >= 0) {
            max = (int)Math.min(length(), TO_STRING_MAX);
        } else {
            max = (int)Math.min(length(), Integer.MAX_VALUE);
        }

        for (int i = 0; i < max; i++) {
            switch (dataType()) {
                case UBYTE:
                case BYTE:
                case INT:
                case SHORT:
                case LONG:
                    ret.append(getNumber(i).intValue());
                    break;
                case BOOL:
                    ret.append(getNumber(i).intValue() == 0 ? " false" : " true");
                    break;
                case UTF8:
                    throw new UnsupportedOperationException();
                case HALF:
                case FLOAT:
                case DOUBLE:
                default:
                    ret.append(getNumber(i).floatValue());
                    break;
            }
            if (i < max - 1)
                ret.append(",");
        }
        if(max < length()){
            ret.append(",<")
                    .append(length()-max)
                    .append(" more elements>");
        }
        ret.append("]");

        return ret.toString();
    }

    @Override
    public int hashCode() {
        int result = (int) length;
        //result = 31 * result + (referencing != null ? referencing.hashCode() : 0);
        //result = 31 * result + (isPersist ? 1 : 0);
        result = 31 * result + (allocationMode != null ? allocationMode.hashCode() : 0);
        return result;
    }

    /**
     * Returns the offset of the buffer relative to originalDataBuffer
     *
     * @return
     */
    @Override
    public long originalOffset() {
        return originalOffset;
    }

    /**
     * Returns tracking point for Allocator
     *
     * PLEASE NOTE: Suitable & meaningful only for specific backends
     *
     * @return
     */
    @Override
    public Long getTrackingPoint() {
        if (underlyingDataBuffer() != this)
            return underlyingDataBuffer() == null ? trackingPoint : underlyingDataBuffer().getTrackingPoint();
        return trackingPoint;
    }

    /**
     * Sets tracking point used by Allocator
     *
     * PLEASE NOTE: Suitable & meaningful only for specific backends
     *
     * @param trackingPoint
     */
    public void setTrackingPoint(Long trackingPoint) {
        this.trackingPoint = trackingPoint;
    }

    /**
     * This method returns whether this DataBuffer is constant, or not.
     * Constant buffer means that it modified only during creation time, and then it stays the same for all lifecycle. I.e. used in shape info databuffers.
     *
     * @return
     */
    public boolean isConstant() {
        return constant;
    }

    /**
     *
     * This method allows you to mark databuffer as constant.
     *
     * PLEASE NOTE: DO NOT USE THIS METHOD, UNLESS YOU'RE 100% SURE WHAT YOU DO
     *
     * @param reallyConstant
     */
    public void setConstant(boolean reallyConstant) {
        this.constant = reallyConstant;
    }

    /**
     * This method returns True, if this DataBuffer is attached to some workspace. False otherwise
     *
     * @return
     */
    @Override
    public boolean isAttached() {
        return attached;
    }


    /**
     * This method checks, if given attached INDArray is still in scope of its parent Workspace
     * <p>
     * PLEASE NOTE: if this INDArray isn't attached to any Workspace, this method will return true
     *
     * @return
     */
    @Override
    public boolean isInScope() {
        if (!isAttached())
            return true;

        return parentWorkspace.isScopeActive();
    }


    @Override
    public MemoryWorkspace getParentWorkspace() {
        if(parentWorkspace != null){
            return parentWorkspace;
        }
        if(wrappedDataBuffer != null && wrappedDataBuffer.isAttached() && wrappedDataBuffer.getParentWorkspace() != null){
            return wrappedDataBuffer.getParentWorkspace();
        }
        if(originalBuffer != null && originalBuffer.isAttached() && originalBuffer.getParentWorkspace() != null){
            return originalBuffer.getParentWorkspace();
        }
        return null;
    }

    /**
     * Reallocate the native memory of the buffer
     * @param length the new length of the buffer
     * @return this databuffer
     * */
    @Override
    public DataBuffer reallocate(long length) {

        Pointer oldPointer = pointer;
        if (isAttached()) {
            long capacity = length * getElementSize();
            switch (dataType()) {
                case DOUBLE:
                    pointer = getParentWorkspace().alloc(capacity, DataType.DOUBLE, false).asDoublePointer();
                    indexer = DoubleIndexer.create((DoublePointer) pointer);
                    break;
                case FLOAT:
                    pointer = getParentWorkspace().alloc(capacity, DataType.FLOAT, false).asFloatPointer();
                    indexer = FloatIndexer.create((FloatPointer) pointer);
                    break;
                case INT:
                    pointer = getParentWorkspace().alloc(capacity, DataType.INT, false).asIntPointer();
                    indexer = IntIndexer.create((IntPointer) pointer);
                    break;
                case LONG:
                    pointer = getParentWorkspace().alloc(capacity, DataType.LONG, false).asLongPointer();
                    indexer = LongIndexer.create((LongPointer) pointer);
                    break;
            }

            workspaceGenerationId = getParentWorkspace().getGenerationId();
        } else {
            switch (dataType()) {
                case INT:
                    pointer = new IntPointer(length);
                    indexer = IntIndexer.create((IntPointer) pointer);
                    break;
                case DOUBLE:
                    pointer = new DoublePointer(length);
                    indexer = DoubleIndexer.create((DoublePointer) pointer);
                    break;
                case FLOAT:
                    pointer = new FloatPointer(length);
                    indexer = FloatIndexer.create((FloatPointer) pointer);
                    break;
                case LONG:
                    pointer = new LongPointer(length);
                    indexer = LongIndexer.create((LongPointer) pointer);
                    break;
            }
        }

        Pointer.memcpy(pointer, oldPointer, this.length() * getElementSize());
        //this.underlyingLength = length;
        return this;
    }

    /**
     * @return the capacity of the buffer
     * */
    @Override
    public long capacity() {
        return pointer().capacity();
    }

    @Override
    public boolean closeable() {
        if (released || isAttached() || isConstant())
            return false;

        if (wrappedDataBuffer != null && wrappedDataBuffer != this)
            return false;

        return true;
    }

    protected void markReleased() {
        this.released = true;

        for (val r:references)
            r.markReleased();
    }

    @Override
    public void close()  {
        if (!closeable())
            throw new IllegalStateException("Can't release this data buffer");

        // notifying other databuffers that their underlying
        for (val r:references)
            r.markReleased();

        release();
    }

    protected void release() {
        this.pointer.deallocate();
        this.indexer = null;
        this.pointer = null;
    }
}
