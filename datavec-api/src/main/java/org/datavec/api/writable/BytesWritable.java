package org.datavec.api.writable;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * {@link Writable} type for
 * a byte array.
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
public class BytesWritable extends ArrayWritable {
    @Getter
    private byte[] content;

    private transient ByteBuffer cached;

    public BytesWritable(byte[] content) {
        this.content = content;
    }

    @Override
    public long length() {
        return content.length;
    }

    @Override
    public double getDouble(long i) {
        return cachedByteByteBuffer().getDouble((int) i);
    }

    @Override
    public float getFloat(long i) {
        return cachedByteByteBuffer().getFloat((int) i);
    }

    @Override
    public int getInt(long i) {
        return cachedByteByteBuffer().getInt((int) i);
    }

    @Override
    public long getLong(long i) {
        return cachedByteByteBuffer().getLong((int) i);
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
}
