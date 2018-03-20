package org.datavec.api.writable;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * {@link Writable} type for
 * a byte array.
 *
 * Note that this {@link Writable}
 * should be considered *raw* and *unsafe*
 * for typical use.
 * This writable's primary use case is for low level flexibility
 * and interop for accessing raw content when there are no alternatives.
 *
 * If you want *safe* indexing that you are familiar with, please
 * consider using nd4j's {@link DataBuffer} object
 * and the associated {@link #asNd4jBuffer(DataBuffer.Type, int)}
 *  method below.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class BytesWritable extends ArrayWritable {
    @Getter
    @Setter
    private byte[] content;

    private transient ByteBuffer cached;

    /**
     * Pass in the content for this writable
     * @param content the content for this writable
     */
    public BytesWritable(byte[] content) {
        this.content = content;
    }

    /**
     * Convert the underlying contents of this {@link Writable}
     * to an nd4j {@link DataBuffer}. Note that this is a *copy*
     * of the underlying buffer.
     * Also note that {@link java.nio.ByteBuffer#allocateDirect(int)}
     * is used for allocation.
     * This should be considered an expensive operation.
     *
     * This buffer should be cached when used. Once used, this can be
     * used in standard Nd4j operations.
     *
     * Beyond that, the reason we have to use allocateDirect
     * is due to nd4j data buffers being stored off heap (whether on cpu or gpu)
     * @param type the type of the data buffer
     * @param elementSize the size of each element in the buffer
     * @return the equivalent nd4j data buffer
     */
    public DataBuffer asNd4jBuffer(DataBuffer.Type type,int elementSize) {
        int length = content.length / elementSize;
        DataBuffer ret = Nd4j.createBuffer(ByteBuffer.allocateDirect(content.length),type,length,0);
        for(int i = 0; i < length; i++) {
            switch(type) {
                case DOUBLE:
                    ret.put(i,getDouble(i));
                    break;
                case INT:
                    ret.put(i,getInt(i));
                    break;
                case FLOAT:
                    ret.put(i,getFloat(i));
                   break;
                case LONG:
                    ret.put(i,getLong(i));
                    break;
            }
        }
        return ret;
    }

    @Override
    public long length() {
        return content.length;
    }

    @Override
    public double getDouble(long i) {
        return cachedByteByteBuffer().getDouble((int) i * 8);
    }

    @Override
    public float getFloat(long i) {
        return cachedByteByteBuffer().getFloat((int) i * 4);
    }

    @Override
    public int getInt(long i) {
        return cachedByteByteBuffer().getInt((int) i * 4);
    }

    @Override
    public long getLong(long i) {
        return cachedByteByteBuffer().getLong((int) i * 8);
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.write(content);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        in.readFully(content);
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        out.writeShort(getType().typeIdx());
    }

    @Override
    public WritableType getType() {
        return WritableType.Bytes;
    }

    private ByteBuffer cachedByteByteBuffer() {
        if(cached == null) {
            cached = ByteBuffer.wrap(content);
        }
        return cached;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BytesWritable that = (BytesWritable) o;
        return Arrays.equals(content, that.content);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(content);
    }
}
