package org.datavec.api.writable;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;

@AllArgsConstructor
public class BytesWritable extends ArrayWritable {
    @Getter
    private byte[] content;



    @Override
    public long length() {
        return content.length;
    }

    @Override
    public double getDouble(long i) {
        return ByteBuffer.wrap(content).getDouble((int) i);
    }

    @Override
    public float getFloat(long i) {
        return ByteBuffer.wrap(content).getFloat((int) i);
    }

    @Override
    public int getInt(long i) {
        return ByteBuffer.wrap(content).getInt((int) i);
    }

    @Override
    public long getLong(long i) {
        return ByteBuffer.wrap(content).getLong((int) i);
    }

    @Override
    public void write(DataOutput out) throws IOException {
        throw new UnsupportedOperationException();
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        throw new UnsupportedOperationException();

    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        throw new UnsupportedOperationException();

    }

    @Override
    public WritableType getType() {
        return WritableType.Image;
    }
}
