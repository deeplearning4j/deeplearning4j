package org.nd4j.linalg.api.buffer;

import org.nd4j.linalg.util.ArrayUtil;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.util.UUID;

/**
 * Int buffer
 * @author Adam Gibson
 */
public class IntBuffer extends BaseDataBuffer {

    private int[] buffer;
    public final static int DATA_TYPE = 2;


    public IntBuffer(int[] buffer) {
        super(buffer.length);
        this.buffer = buffer;
    }

    public IntBuffer(int length) {
        super(length);
    }

    @Override
    public byte[] asBytes() {
        return new byte[0];
    }

    @Override
    public int dataType() {
        return DATA_TYPE;
    }

    @Override
    public float[] asFloat() {
        float[] ret = new float[length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = (float) buffer[i];
        }
        return ret;
    }

    @Override
    public double[] asDouble() {
        double[] ret = new double[length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = (double) buffer[i];
        }
        return ret;
    }

    @Override
    public int[] asInt() {
        return buffer;
    }



    @Override
    public double getDouble(int i) {
        return buffer[i];
    }

    @Override
    public float getFloat(int i) {
        return buffer[i];
    }

    @Override
    public Number getNumber(int i) {
        return buffer[i];
    }

    @Override
    public void put(int i, float element) {
       buffer[i] = (int) element;
    }

    @Override
    public void put(int i, double element) {
       buffer[i] = (int) element;
    }

    @Override
    public void put(int i, int element) {
         buffer[i] = element;
    }

    @Override
    public int getInt(int ix) {
        return buffer[ix];
    }

    @Override
    public DataBuffer dup() {
        return new IntBuffer(ArrayUtil.copy(buffer));
    }
    @Override
    public void flush() {
        path = UUID.randomUUID().toString();
        if(memoryMappedBuffer != null)
            return;
        try {
            memoryMappedBuffer = new RandomAccessFile(path,"rw");
            long size = 8L * length;
            for (long offset = 0; offset < size; offset += MAPPING_SIZE) {
                long size2 = Math.min(size - offset, MAPPING_SIZE);
                mappings.add(memoryMappedBuffer.getChannel().map(FileChannel.MapMode.READ_WRITE, offset, size2));
            }
        } catch (IOException e) {
            try {
                if(memoryMappedBuffer != null)
                    memoryMappedBuffer.close();
            } catch (IOException e1) {
                throw new RuntimeException(e);
            }
            throw new RuntimeException(e);
        }

        buffer = null;
    }

    @Override
    public void destroy() {
        if(buffer != null)
            buffer = null;
        if(memoryMappedBuffer != null) {
            try {
                this.mappings.clear();
                this.memoryMappedBuffer.close();
            } catch (IOException e) {
                throw new RuntimeException(e);

            }
        }
    }
}
