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

package org.nd4j.linalg.api.buffer;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.primitives.AtomicDouble;
import org.nd4j.common.primitives.Triple;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.io.*;
import java.math.BigInteger;
import java.nio.*;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;


@Slf4j
public abstract class BaseDataBuffer implements DataBuffer {

    /**
     * @deprecated Use {@link ND4JSystemProperties#DATABUFFER_TO_STRING_MAX_ELEMENTS}
     */
    public static String TO_STRING_MAX_ELEMENTS = ND4JSystemProperties.DATABUFFER_TO_STRING_MAX_ELEMENTS;
    private static int TO_STRING_MAX;
    static {
        String s = System.getProperty(ND4JSystemProperties.DATABUFFER_TO_STRING_MAX_ELEMENTS);
        if(s != null) {
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
    protected transient OpaqueDataBuffer ptrDataBuffer;
    protected transient Deallocator deallocator;
    protected StackTraceElement[] allocationTrace =  Nd4j.getEnvironment().isFuncTracePrintAllocate()
            || Nd4j.getEnvironment().isFuncTracePrintJavaOnly() ?
            Thread.currentThread().getStackTrace() : null;


    protected DataType type;
    protected long length;

    protected long deallocationId;
    protected long underlyingLength;
    protected byte elementSize;
    protected transient long workspaceGenerationId = 0L;

    protected AllocationMode allocationMode;

    protected transient Indexer indexer = null;
    protected transient Pointer pointer = null;

    protected transient boolean attached = false;
    protected transient MemoryWorkspace parentWorkspace;


    protected transient boolean constant = false;
    protected transient AtomicBoolean released = new AtomicBoolean(false);

    protected transient AtomicBoolean referenced = new AtomicBoolean(false);

    public BaseDataBuffer() {}


    @Override
    public StackTraceElement[] allocationTrace() {
        return allocationTrace;
    }

    @Override
    public Deallocator deallocator() {
        return deallocator;
    }

    /**
     * Initialize the opType of this buffer
     */
    protected abstract void initTypeAndSize();

    @Override
    public OpaqueDataBuffer opaqueBuffer() {
        return ptrDataBuffer;
    }

    @Override
    public int getElementSize() {
        return elementSize;
    }



    @Override
    public long getGenerationId() {
        if(parentWorkspace != null) {
            return workspaceGenerationId;
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
        if (length < 0)
            throw new IllegalArgumentException("Length must be >= 0");

        initTypeAndSize();
        this.length = length;
        this.allocationMode = AllocationMode.MIXED_DATA_TYPES;
        this.underlyingLength = length;

        if (length > 0 || indexer != null) {
            this.pointer = pointer;
            setIndexer(indexer);
        }
        if(!Nd4j.getDeallocatorService().getListeners().isEmpty()) {
            Nd4j.getDeallocatorService().registerDataBufferToListener(this);
        }
    }


    protected void setIndexer(Indexer indexer) {
        this.indexer = indexer;
    }

    protected void pickReferent(BaseDataBuffer referent) {
        referenced.compareAndSet(false, true);
    }


    //sets the nio wrapped buffer (allows to be overridden for other use cases like cuda)
    protected void setNioBuffer() {
        if (elementSize * length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create buffer of length " + length);

    }

    /**
     * Returns the indexer for the buffer
     *
     * @return
     */
    @Override
    public Indexer indexer() {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        return indexer;
    }

    @Override
    public Pointer pointer() {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");
        if (released.get())
            throw new IllegalStateException("This buffer was already released via close() call");

        return pointer;
    }


    @Override
    public AllocationMode allocationMode() {
        return allocationMode;
    }

    @Override
    @Deprecated
    public void persist() {
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

    protected void fillPointerWithZero() {
        Pointer.memset(this.pointer(), 0, getElementSize() * length());
    }


    @Override
    public void copyAtStride(DataBuffer buf, long n, long stride, long yStride, long offset, long yOffset) {
        Nd4j.getNativeOps().copyBuffer(buf.opaqueBuffer(),n,this.opaqueBuffer(),offset,yOffset);
    }

    @Override
    @Deprecated
    public void removeReferencing(String id) {
    }

    @Override
    @Deprecated
    public Collection<String> references() {
        throw new UnsupportedOperationException();
        //return referencing;
    }

    public abstract Pointer addressPointer();


    @Override
    public long address() {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        return pointer().address();
    }

    @Override
    @Deprecated
    public void addReferencing(String id) {
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
        put(data);
    }

    @Override
    public void setData(float[] data) {
        put(data);
    }

    @Override
    public void setData(double[] data) {
        put(data);
    }

    @Override
    public void setData(long[] data) {
        put(data);
    }

    @Override
    public void setData(byte[] data) {
        put(data);
    }

    @Override
    public void setData(short[] data) {
        put(data);
    }


    @Override
    public void setData(boolean[] data) {
        put(data);
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
        Nd4j.getNativeOps().copyBuffer(ret.opaqueBuffer(),
                length, opaqueBuffer(), 0, 0);


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
        //NOTE: DataOutputStream is big endian
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        val dataType = dataType();
        switch (dataType) {
            case DOUBLE:
                double[] data = asDouble();
                try {
                    for (int i = 0; i < length(); i++) {
                        dos.writeDouble(data[i]);
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                break;
            case FLOAT:
                try {
                    float[] dataFloat = asFloat();
                    for (int i = 0; i < length(); i++) {
                        dos.writeFloat(dataFloat[i]);
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                break;
            case HALF:
                try {
                    float[] dataFloat = asFloat();

                    for (int i = 0; i < length(); i++) {
                        dos.writeShort(HalfIndexer.fromFloat(dataFloat[i]));
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                break;
            case BOOL:
                try {
                    int[] ints = asInt();
                    for (int i = 0; i < length(); i++) {
                        dos.writeByte(ints[i] == 0 ? (byte) 0 : (byte) 1);
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                break;
            case BYTE:
                try {
                    for (int i = 0; i < length(); i++) {
                        dos.writeByte((byte) getShort(i));
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                break;
            case UBYTE:
                try {
                    UByteIndexer u = (UByteIndexer) indexer;
                    for (int i = 0; i < length(); i++) {
                        dos.writeByte(u.get(i));
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                break;
            case SHORT:
                try{
                    for (int i = 0; i < length(); i++) {
                        dos.writeShort(getShort(i));
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                break;
            case INT:
                try {
                    for (int i = 0; i < length(); i++) {
                        dos.writeInt(getInt(i));
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                break;
            case LONG:
                try {
                    for (int i = 0; i < length(); i++) {
                        dos.writeLong(getLong(i));
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                break;
            case BFLOAT16:
            case UINT16:
                //Treat BFloat16 and UINT16 as bytes
                byte[] temp = new byte[(int)(2*length)];
                asNio().get(temp);
                try {
                    if(ByteOrder.nativeOrder().equals(ByteOrder.LITTLE_ENDIAN)) {
                        //Switch endianness to big endian
                        for (int i = 0; i < temp.length / 2; i++) {
                            dos.write(temp[2 * i + 1]);
                            dos.write(temp[2 * i]);
                        }
                    } else {
                        //Keep as big endian
                        dos.write(temp);
                    }
                } catch (IOException e){
                    throw new RuntimeException(e);
                }
                break;
            case UINT64:
                //Treat unsigned long (UINT64) as 8 bytes
                byte[] temp2 = new byte[(int)(8 * length)];
                asNio().get(temp2);
                try {
                    if(ByteOrder.nativeOrder().equals(ByteOrder.LITTLE_ENDIAN)) {
                        //Switch endianness to big endian
                        for (int i = 0; i < temp2.length / 8; i++) {
                            for( int j = 0; j < 8; j++) {
                                dos.write(temp2[8 * i + (7 - j)]);
                            }
                        }
                    } else {
                        //Keep as big endian
                        dos.write(temp2);
                    }
                } catch (IOException e){
                    throw new RuntimeException(e);
                }
                break;
            case UINT32:
                //Treat unsigned integer (UINT32) as 4 bytes
                byte[] temp3 = new byte[(int)(4 * length)];
                asNio().get(temp3);
                try {
                    if(ByteOrder.nativeOrder().equals(ByteOrder.LITTLE_ENDIAN)) {
                        //Switch endianness to big endian
                        for (int i = 0; i < temp3.length / 4; i++) {
                            for( int j = 0; j < 4; j++) {
                                dos.write(temp3[4 * i + (3 - j)]);
                            }
                        }
                    } else {
                        //Keep as big endian
                        dos.write(temp3);
                    }
                } catch (IOException e){
                    throw new RuntimeException(e);
                }
                break;
            case UTF8:
                byte[] temp4 = new byte[(int)length];
                asNio().get(temp4);
                try {
                    dos.write(temp4);
                } catch (IOException e){
                    throw new RuntimeException(e);
                }
                break;
            default:
                throw new UnsupportedOperationException("Unknown data type: [" + dataType + "]");
        }
        return bos.toByteArray();
    }

    @Override
    public boolean[] asBoolean() {
        if(length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        boolean[] ret = new boolean[(int) length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = getIntUnsynced(i) > 0;
        }

        return ret;

    }

    @Override
    public float[] asFloat() {
        if (length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        float[] ret = new float[(int) length];
        for (int i = 0; i < length; i++)
            ret[i] = getFloatUnsynced(i);
        return ret;
    }

    @Override
    public double[] asDouble() {
        if (length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        double[] ret = new double[(int) length];
        for (int i = 0; i < length; i++)
            ret[i] = getDoubleUnsynced(i);
        return ret;
    }

    @Override
    public int[] asInt() {
        if (length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        int[] ret = new int[(int) length];
        for (int i = 0; i < length; i++)
            ret[i] = getIntUnsynced(i);
        return ret;
    }

    @Override
    public long[] asLong() {
        if (length >= Integer.MAX_VALUE)
            throw new IllegalArgumentException("Unable to create array of length " + length);
        long[] ret = new long[(int) length];
        for (int i = 0; i < length; i++)
            ret[i] = getLongUnsynced(i);
        return ret;
    }

    @Override
    public double getDouble(long i) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        if (indexer == null) {
            throw new IllegalStateException("Indexer must never be null");
        }
        switch (dataType()) {
            case FLOAT:
                return ((FloatIndexer) indexer).get(i);
            case UINT32:
                return ((UIntIndexer) indexer).get(i);
            case INT:
                return ((IntIndexer) indexer).get(i);
            case BFLOAT16:
                return ((Bfloat16Indexer) indexer).get(i);
            case HALF:
                return ((HalfIndexer) indexer).get(i);
            case UINT16:
                return ((UShortIndexer) indexer).get(i);
            case SHORT:
                return ((ShortIndexer) indexer).get(i);
            case UINT64:
                return ((ULongIndexer) indexer).get(i).doubleValue();
            case LONG:
                return ((LongIndexer) indexer).get(i);
            case BOOL:
                return ((BooleanIndexer) indexer).get(i) ? 1.0 : 0.0;
            case DOUBLE:
                return ((DoubleIndexer) indexer).get(i);
            case BYTE:
                return ((ByteIndexer) indexer).get(i);
            case UBYTE:
                return ((UByteIndexer) indexer).get(i);
            default:
                throw new UnsupportedOperationException("Cannot get double value from buffer of type " + dataType());
        }
    }

    @Override
    public long getLong(long i) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case FLOAT:
                return (long) ((FloatIndexer) indexer).get(i);
            case DOUBLE:
                return (long) ((DoubleIndexer) indexer).get(i);
            case BFLOAT16:
                return (long) ((Bfloat16Indexer) indexer).get(i);
            case HALF:
                return (long) ((HalfIndexer) indexer).get( i);
            case UINT64:    //Fall through
                return  ((ULongIndexer) indexer).get(i).longValue();
            case LONG:
                return ((LongIndexer) indexer).get(i);
            case UINT32:
                return ((UIntIndexer) indexer).get(i);
            case INT:
                return ((IntIndexer) indexer).get(i);
            case UINT16:
                return ((UShortIndexer) indexer).get(i);
            case SHORT:
                return ((ShortIndexer) indexer).get(i);
            case BYTE:
                return ((ByteIndexer) indexer).get(i);
            case UBYTE:
                return ((UByteIndexer) indexer).get(i);
            case BOOL:
                return  ((BooleanIndexer) indexer).get(i) ? 1L : 0L;
            default:
                throw new UnsupportedOperationException("Cannot get long value from buffer of type " + dataType());
        }
    }

    /**
     * Special method for
     * @param i
     * @return
     */
    protected short getShort(long i) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case DOUBLE:
                return (short) ((DoubleIndexer) indexer).get(i);
            case BFLOAT16:
                return (short) ((Bfloat16Indexer) indexer).get(i);
            case HALF:
                return (short) ((HalfIndexer) indexer).get(i);
            case BOOL:
                return (short) (((BooleanIndexer) indexer).get(i) ? 1 : 0);
            case UINT32:
                return (short) ((UIntIndexer)indexer).get(i);
            case INT:
                return (short) ((IntIndexer) indexer).get(i);
            case UINT16:
            case SHORT:
                return ((ShortIndexer) indexer).get(i);
            case BYTE:
                return ((ByteIndexer) indexer).get(i);
            case UINT64:
                return (short) ((ULongIndexer) indexer).get(i).shortValue();
            case LONG:
                return (short) ((LongIndexer) indexer).get(i);
            case FLOAT:
                return (short) ((FloatIndexer) indexer).get(i);
            default:
                throw new UnsupportedOperationException("Cannot get short value from buffer of type " + dataType());
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
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case DOUBLE:
                return (float) ((DoubleIndexer) indexer).get(i);
            case BOOL:
                return ((BooleanIndexer) indexer).get(i) ? 1.f : 0.f;
            case UINT32:
                return (float) ((UIntIndexer)indexer).get(i);
            case INT:
                return (float) ((IntIndexer) indexer).get(i);
            case UINT16:
                return ((UShortIndexer) indexer).get(i);
            case SHORT:
                return ((ShortIndexer) indexer).get(i);
            case BFLOAT16:
                return ((Bfloat16Indexer) indexer).get(i);
            case HALF:
                return ((HalfIndexer) indexer).get(i);
            case UBYTE:
                return (float) ((UByteIndexer) indexer).get(i);
            case BYTE:
                return ((ByteIndexer) indexer).get(i);
            case UINT64:
                return ((ULongIndexer) indexer).get(i).floatValue();
            case LONG:
                return (float)  ((LongIndexer) indexer).get(i);
            case FLOAT:
                return ((FloatIndexer) indexer).get(i);
            default:
                throw new UnsupportedOperationException("Cannot get float value from buffer of type " + dataType());
        }
    }

    @Override
    public int getInt(long i) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case DOUBLE:
                return (int) ((DoubleIndexer) indexer).get(i);
            case BOOL:
                return ((BooleanIndexer) indexer).get(i) ? 1 : 0;
            case UINT32:
                return (int)((UIntIndexer) indexer).get(i);
            case INT:
                return ((IntIndexer) indexer).get(i);
            case BFLOAT16:
                return (int) ((Bfloat16Indexer) indexer).get(i);
            case HALF:
                return (int) ((HalfIndexer) indexer).get(i);
            case UINT16:
                return ((UShortIndexer) indexer).get(i);
            case SHORT:
                return ((ShortIndexer) indexer).get(i);
            case UBYTE:
                return ((UByteIndexer) indexer).get(i);
            case BYTE:
                return ((ByteIndexer) indexer).get(i);
            case UINT64:
                return ((ULongIndexer) indexer).get(i).intValue();
            case LONG:
                return (int) ((LongIndexer) indexer).get(i);
            case FLOAT:
                return (int) ((FloatIndexer) indexer).get(i);
            default:
                throw new UnsupportedOperationException("Cannot get integer value from buffer of type " + dataType());
        }
    }

    @Override
    public Number getNumber(long i) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        if (dataType() == DataType.DOUBLE)
            return getDouble(i);
        else if (dataType() == DataType.INT || dataType() == DataType.INT32)
            return getInt(i);
        else if (dataType() == DataType.LONG || dataType() == DataType.INT64)
            return getLong(i);
        return getFloat(i);
    }

    public abstract void pointerIndexerByCurrentType(DataType currentType);

    public void putByDestinationType(long i, Number element, DataType globalType) {
        if (globalType == DataType.INT32 || globalType == DataType.INT || type == DataType.INT || globalType == DataType.UINT16 || globalType == DataType.UBYTE || globalType == DataType.SHORT|| globalType == DataType.BYTE || globalType == DataType.BOOL) {
            int anElement = element.intValue();
            put(i, anElement);
        } else if (globalType == DataType.INT64 || globalType == DataType.LONG || type == DataType.LONG || globalType == DataType.UINT32 || globalType == DataType.UINT64) {
            long anElement = element.longValue();
            put(i, anElement);
        } else if (globalType == DataType.FLOAT || globalType == DataType.HALF || globalType == DataType.BFLOAT16) {
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
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(i, element == 0.0 ? false : true);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(i, (byte) element);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(i,  (int) element);
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(i,  (int)element);
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(i,  (short) element);
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(i, (long)element);
                break;
            case INT:
                ((IntIndexer) indexer).put(i, (int) element);
                break;
            case UINT64:
                ((ULongIndexer) indexer).put(i,  BigInteger.valueOf((long) element));
                break;
            case LONG:
                ((LongIndexer) indexer).put(i, (long) element);
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(i,  element);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(i,  element);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(i, element);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(i, element);
                break;
            default:
                throw new IllegalStateException("Unsupported type: " + dataType());
        }
    }

    @Override
    public void put(long i, double element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(i,  element > 0.0);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(i, (byte) element);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(i, (short) element);
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(i,  (int) element);
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(i,  (short) element);
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(i, (long)element);
                break;
            case INT:
                ((IntIndexer) indexer).put(i, (int) element);
                break;
            case UINT64:
                ((ULongIndexer) indexer).put(i,BigInteger.valueOf((long) element));
                break;
            case LONG:
                ((LongIndexer) indexer).put(i, (long) element);
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(i, (float) element);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(i, (float) element);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(i, (float) element);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(i, element);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }



    @Override
    public void put(long i, short element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(i, element == 0 ? false : true);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(i,  (byte) element);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(i,  element);
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(i,  element);
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(i, element);
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(i, element);
                break;
            case INT:
                ((IntIndexer) indexer).put(i, element);
                break;
            case UINT64: //Fall through
                ((ULongIndexer) indexer).put(i,BigInteger.valueOf(element));
                break;
            case LONG:
                ((LongIndexer) indexer).put(i,element);
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(i, element);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(i, element);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(i, element);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(i, element);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }


    @Override
    public void put(long i, int element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(i, element == 0 ? false : true);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(i,  (byte) element);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(i,  element);
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(i,  element);
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(i,  (short) element);
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(i, element);
                break;
            case INT:
                ((IntIndexer) indexer).put(i, element);
                break;
            case UINT64: //Fall through
                ((ULongIndexer) indexer).put(i,BigInteger.valueOf(element));
                break;
            case LONG:
                ((LongIndexer) indexer).put(i,element);
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(i, element);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(i, element);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(i, element);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(i, element);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }

    @Override
    public void put(long i, boolean element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(i, element);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(i, element ? (byte)1 : (byte) 0);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(i, element ? (byte)1 : (byte) 0);
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(i,  element ? 1 : 0);
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(i, element ? (short) 1 : (short) 0);
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(i, element ? 1 : 0);
                break;
            case INT:
                ((IntIndexer) indexer).put(i, element ? 1 : 0);
                break;
            case UINT64:
                ((ULongIndexer) indexer).put(i, BigInteger.valueOf(element ? 1 : 0));
                break;
            case LONG:
                ((LongIndexer) indexer).put(i, element ? 1 : 0);
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(i, element ? 1.0f : 0.0f);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(i, element ? 1.0f : 0.0f);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(i, element ? 1.0f : 0.0f);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(i,  element ? 1.0 : 0.0);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }

    @Override
    public void put(long i,long element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(i, element == 0 ? false : true);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(i, (byte) element);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(i, (short) element);
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(i,  (int) element);
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(i, (short) element);
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(i, element);
                break;
            case INT:
                ((IntIndexer) indexer).put(i, (int) element);
                break;
            case UINT64:
                ((ULongIndexer) indexer).put(i, ULongIndexer.toBigInteger(element));
                break;
            case LONG:
                ((LongIndexer) indexer).put(i, element);
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(i, (float) element);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(i, (float) element);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(i, (float) element);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(i, (double) element);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }

    @Override
    public void put(float[] element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:

                ((BooleanIndexer) indexer).put(0,ArrayUtil.fromFloat(element));
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(0,ArrayUtil.toBytes(element));
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(0,ArrayUtil.toInts(element));
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(0,ArrayUtil.toShorts(element));
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(0,ArrayUtil.toLongArray(element));
                break;
            case INT:
                ((IntIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case UINT64:
                BigInteger[] bigIntegers = ArrayUtil.toBigInteger(element);
                ((ULongIndexer) indexer).put(0,bigIntegers);
                break;
            case LONG:
                ((LongIndexer) indexer).put(0,ArrayUtil.toLongs(element));
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(0,element);
                break;
            case HALF:
                ((HalfIndexer) indexer).put(0,element);
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(0,element);
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(0,ArrayUtil.toDoubleArray(element));
                break;
            default:
                throw new IllegalStateException("Unsupported type: " + dataType());
        }
    }

    @Override
    public void put(double[] element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(0,ArrayUtil.toBooleanArray(element));
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(0,ArrayUtil.toBytes(element));
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(0,ArrayUtil.toInts(element));
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(0,ArrayUtil.toShorts(element));
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(0,ArrayUtil.toLongArray(element));
                break;
            case INT:
                ((IntIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case UINT64:
                BigInteger[] bigIntegers = ArrayUtil.toBigInteger(element);
                ((ULongIndexer) indexer).put(0,bigIntegers);
                break;
            case LONG:
                ((LongIndexer) indexer).put(0,ArrayUtil.toLongArray(element));
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case HALF:
                ((HalfIndexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(0,ArrayUtil.toDoubleArray(element));
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }

    @Override
    public void put(int[] element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(0,ArrayUtil.toBooleanArray(element));
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(0,ArrayUtil.toBytes(element));
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(0,ArrayUtil.toShorts(element));
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(0,ArrayUtil.toLongArray(element));
                break;
            case INT:
                ((IntIndexer) indexer).put(0, element);
                break;
            case UINT64: //Fall through
                BigInteger[] map = ArrayUtil.toBigInteger(element);
                ((ULongIndexer) indexer).put(0,map);
                break;
            case LONG:
                ((LongIndexer) indexer).put(0,ArrayUtil.toLongArray(element));
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(0, ArrayUtil.toFloatArray(element));
                break;
            case HALF:
                ((HalfIndexer) indexer).put(0, ArrayUtil.toFloatArray(element));
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(0,ArrayUtil.toDouble(element));
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }

    @Override
    public void put(boolean[] element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(0, element);
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(0,ArrayUtil.toBytes(element));
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(0,ArrayUtil.toShorts(element));
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(0,ArrayUtil.toLongArray(element));
                break;
            case INT:
                ((IntIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case UINT64:
                ((ULongIndexer) indexer).put(0,ArrayUtil.toBigInteger(element));
                break;
            case LONG:
                ((LongIndexer) indexer).put(0,ArrayUtil.toLongArray(element));
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case HALF:
                ((HalfIndexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(0,ArrayUtil.toDoubleArray(element));
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }


    @Override
    public void put(short[] element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(0, ArrayUtil.toBooleanArray(element));
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(0,ArrayUtil.toBytes(element));
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(0,element);
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(0,ArrayUtil.toLongArray(element));
                break;
            case INT:
                ((IntIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case UINT64:
                ((ULongIndexer) indexer).put(0,ArrayUtil.toBigInteger(element));
                break;
            case LONG:
                ((LongIndexer) indexer).put(0,ArrayUtil.toLongs(element));
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case HALF:
                ((HalfIndexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(0,ArrayUtil.toDoubleArray(element));
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }


    @Override
    public void put(byte[] element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(0, ArrayUtil.toBooleanArray(element));
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(0,element);
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(0,ArrayUtil.toIntArraySimple(element));
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(0,ArrayUtil.toIntArraySimple(element));
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(0,ArrayUtil.toShorts(element));
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(0,ArrayUtil.toLongArray(element));
                break;
            case INT:
                ((IntIndexer) indexer).put(0,ArrayUtil.toIntArraySimple(element));
                break;
            case UINT64:
                ((ULongIndexer) indexer).put(0,ArrayUtil.toBigInteger(element));
                break;
            case LONG:
                ((LongIndexer) indexer).put(0,ArrayUtil.toLongArray(element));
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(0,ArrayUtil.toFloatArraySimple(element));
                break;
            case HALF:
                ((HalfIndexer) indexer).put(0,ArrayUtil.toFloatArraySimple(element));
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(0,ArrayUtil.toFloatArraySimple(element));
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(0,ArrayUtil.toDoubleArraySimple(element));
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
        }
    }

    @Override
    public void put(long[] element) {
        if (released.get())
            throw new IllegalStateException("You can't use DataBuffer once it was released");

        switch (dataType()) {
            case BOOL:
                ((BooleanIndexer) indexer).put(0,ArrayUtil.toBooleanArray(element));
                break;
            case BYTE:
                ((ByteIndexer) indexer).put(0,ArrayUtil.toByteArraySimple(element));
                break;
            case UBYTE:
                ((UByteIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case UINT16:
                ((UShortIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case SHORT:
                ((ShortIndexer) indexer).put(0,ArrayUtil.toShorts(element));
                break;
            case UINT32:
                ((UIntIndexer) indexer).put(0,element);
                break;
            case INT:
                ((IntIndexer) indexer).put(0,ArrayUtil.toIntArray(element));
                break;
            case UINT64:
                ((ULongIndexer) indexer).put(0,ArrayUtil.toBigInteger(element));
                break;
            case LONG:
                ((LongIndexer) indexer).put(0,ArrayUtil.toLongArray(element));
                break;
            case BFLOAT16:
                ((Bfloat16Indexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case HALF:
                ((HalfIndexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case FLOAT:
                ((FloatIndexer) indexer).put(0,ArrayUtil.toFloatArray(element));
                break;
            case DOUBLE:
                ((DoubleIndexer) indexer).put(0,ArrayUtil.toDoubles(element));
                break;
            default:
                throw new UnsupportedOperationException("Unsupported data type: " + dataType());
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
        return wrappedBuffer().asIntBuffer();

    }

    @Override
    public LongBuffer asNioLong() {
        return wrappedBuffer().asLongBuffer();
    }

    @Override
    public DoubleBuffer asNioDouble() {
        return wrappedBuffer().asDoubleBuffer();

    }

    @Override
    public FloatBuffer asNioFloat() {
        return wrappedBuffer().asFloatBuffer();


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
        if (o instanceof DataBuffer) {
            DataBuffer d = (DataBuffer) o;
            if (d.length() != length())
                return false;

            if(d.dataType() != dataType())
                return false;
            OpContext ctx = Nd4j.getExecutioner().buildContext();
            ctx.setInputArrays(Nd4j.create(d),Nd4j.create(this));
            INDArray exec = Nd4j.getExecutioner().exec(new Eps(Nd4j.create(d), Nd4j.create(this), Nd4j.createUninitialized(DataType.BOOL, length())));
            return exec.all();
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


            } else if (savedMode.equals(AllocationMode.LONG_SHAPE)) {
                length = len;
                val currentType = dtype;
                type = currentType;

                if (currentType == DataType.LONG)
                    elementSize = 8;
                else if (currentType == DataType.DOUBLE && currentType != DataType.INT)
                    elementSize = 8;
                else if (currentType == DataType.FLOAT || currentType == DataType.INT)
                    elementSize = 4;
                else if (currentType == DataType.HALF && currentType != DataType.INT)
                    elementSize = 2;

                pointerIndexerByCurrentType(currentType);

                if (currentType != DataType.COMPRESSED)
                    readContent(s, currentType, currentType);
            } else if (allocationMode.equals(AllocationMode.MIXED_DATA_TYPES)) {
                switch (type) {
                    case UINT64:
                    case LONG:
                    case DOUBLE:
                        elementSize = 8;
                        break;
                    case UINT32:
                    case FLOAT:
                    case INT:
                        elementSize = 4;
                        break;
                    case UINT16:
                    case SHORT:
                    case HALF:
                    case BFLOAT16:
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
                    putByDestinationType(i, HalfIndexer.toFloat(aInt.get()), thisType);
                }
            } else if (sourceType == DataType.BFLOAT16) {
                AtomicInteger aInt = new AtomicInteger();
                for (long i = 0; i < length(); i++) {
                    aInt.set(s.readShort());
                    putByDestinationType(i, Bfloat16Indexer.toFloat(aInt.get()), thisType);
                }
            } else if (sourceType == DataType.UINT64) {
                AtomicLong aLong = new AtomicLong();
                for (long i = 0; i < length(); i++) {
                    aLong.set(s.readLong());
                    putByDestinationType(i, aLong, thisType);
                }
            } else if (sourceType == DataType.LONG) {
                AtomicLong aLong = new AtomicLong();
                for (long i = 0; i < length(); i++) {
                    aLong.set(s.readLong());
                    putByDestinationType(i, aLong, thisType);
                }
            } else if (sourceType == DataType.UINT32) {
                AtomicLong aLong = new AtomicLong();
                for (long i = 0; i < length(); i++) {
                    aLong.set(s.readInt());
                    putByDestinationType(i, aLong, thisType);
                }
            } else if (sourceType == DataType.INT ){
                AtomicInteger aInt = new AtomicInteger();
                for (long i = 0; i < length(); i++) {
                    aInt.set(s.readInt());
                    putByDestinationType(i, aInt, thisType);
                }
            } else if (sourceType == DataType.UINT16 ){
                AtomicInteger aInt = new AtomicInteger();
                for (long i = 0; i < length(); i++) {
                    aInt.set(s.readShort());
                    putByDestinationType(i, aInt, thisType);
                }
            } else if (sourceType == DataType.SHORT ){
                AtomicInteger aInt = new AtomicInteger();
                for (long i = 0; i < length(); i++) {
                    aInt.set(s.readShort());
                    putByDestinationType(i, aInt, thisType);
                }
            } else if (sourceType == DataType.UBYTE ){
                AtomicInteger aInt = new AtomicInteger();
                for (long i = 0; i < length(); i++) {
                    aInt.set(s.readByte());
                    putByDestinationType(i, aInt, thisType);
                }
            } else if (sourceType == DataType.BYTE ){
                AtomicInteger aInt = new AtomicInteger();
                for (long i = 0; i < length(); i++) {
                    aInt.set(s.readByte());
                    putByDestinationType(i, aInt, thisType);
                }
            } else if (sourceType == DataType.BOOL ){
                AtomicInteger aInt = new AtomicInteger();
                for (long i = 0; i < length(); i++) {
                    aInt.set(s.readByte());
                    putByDestinationType(i, aInt, thisType);
                }
            } else {
                throw new UnsupportedOperationException("Cannot read type: " + sourceType + " to " + thisType);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    protected abstract double getDoubleUnsynced(long index);
    protected abstract float getFloatUnsynced(long index);
    protected abstract long getLongUnsynced(long index);
    protected abstract int getIntUnsynced(long index);

    @Override
    public void write(DataOutputStream out) throws IOException {
        out.writeUTF(allocationMode.name());
        out.writeLong(length());
        out.writeUTF(dataType().name());
        switch (dataType()) {
            case DOUBLE:
                for (long i = 0; i < length(); i++)
                    out.writeDouble(getDoubleUnsynced(i));
                break;
            case UINT64:
            case LONG:
                for (long i = 0; i < length(); i++)
                    out.writeLong(getLongUnsynced(i));
                break;
            case UINT32:
            case INT:
                for (long i = 0; i < length(); i++)
                    out.writeInt(getIntUnsynced(i));
                break;
            case UINT16:
            case SHORT:
                for (long i = 0; i < length(); i++)
                    out.writeShort((short) getIntUnsynced(i));
                break;
            case UBYTE:
            case BYTE:
                for (long i = 0; i < length(); i++)
                    out.writeByte((byte) getIntUnsynced(i));
                break;
            case BOOL:
                for (long i = 0; i < length(); i++)
                    out.writeByte(getIntUnsynced(i) == 0 ? (byte) 0 : (byte) 1);
                break;
            case BFLOAT16:
                for (long i = 0; i < length(); i++)
                    out.writeShort((short) Bfloat16Indexer.fromFloat(getFloatUnsynced(i)));
                break;
            case HALF:
                for (long i = 0; i < length(); i++)
                    out.writeShort((short) HalfIndexer.fromFloat(getFloatUnsynced(i)));
                break;
            case FLOAT:
                for (long i = 0; i < length(); i++)
                    out.writeFloat(getFloatUnsynced(i));
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
        result = 31 * result + (allocationMode != null ? allocationMode.hashCode() : 0);
        return result;
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
        deallocator().setConstant(reallyConstant);
        this.constant = reallyConstant;
        Nd4j.getDeallocatorService().getReferenceMap().remove(this.deallocationId);

    }

    @Override
    public boolean shouldDeAllocate() {
        return !isConstant() && !released.get();
    }

    @Override
    public int targetDevice() {
        return 0;
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
        if(parentWorkspace != null) {
            return parentWorkspace;
        }


        return null;
    }

    public abstract DataBuffer reallocate(long length);

    /**
     * @return the capacity of the buffer
     * */
    @Override
    public long capacity() {
        return pointer().capacity();
    }

    @Override
    public boolean closeable() {
        if (released.get() || isAttached() || isConstant())
            return false;


        return true;
    }


    @Override
    public void close()  {
        if (!closeable())
            throw new IllegalStateException("Can't release this data buffer");

        release();
    }

    protected void release() {
        this.released.set(true);
        this.indexer = null;
        this.pointer = null;
        Nd4j.getDeallocatorService().getReferenceMap().remove(deallocationId);

    }

    @Override
    public long platformAddress() {
        return address();
    }


    @Override
    public boolean wasClosed() {
        return released.get();
    }


    /**
     * This method synchronizes host memory
     */
    public abstract void syncToPrimary();

    /**
     * This method synchronizes device memory
     */
    public abstract void syncToSpecial();


}
