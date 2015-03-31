/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.buffer;

import org.nd4j.linalg.util.ArrayUtil;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.util.UUID;

/**
 * Int buffer
 *
 * @author Adam Gibson
 */
public class IntBuffer extends BaseDataBuffer {

    public final static int DATA_TYPE = 2;
    private int[] buffer;

    public IntBuffer(int[] buffer, boolean copy) {
        super(buffer.length);
        if (!copy)
            this.buffer = buffer;
        else {
            buffer = new int[buffer.length];
            System.arraycopy(buffer, 0, this.buffer, 0, this.buffer.length);
        }

    }

    public IntBuffer(int[] buffer) {
        this(buffer, true);
    }

    public IntBuffer(int length) {
        super(length);
    }

    @Override
    public void setData(int[] data) {
        this.buffer = data;
    }

    @Override
    public void setData(float[] data) {
        this.buffer = ArrayUtil.toInts(data);
    }

    @Override
    public void setData(double[] data) {
        this.buffer = ArrayUtil.toInts(data);
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
        for (int i = 0; i < ret.length; i++) {
            ret[i] = (float) buffer[i];
        }
        return ret;
    }


    @Override
    public int elementSize() {
        return 4;
    }

    @Override
    public void assign(Number value, int offset) {
        for (int i = offset; i < length(); i++) {
            buffer[i] = value.intValue();
        }
    }


    @Override
    public double[] asDouble() {
        double[] ret = new double[length];
        for (int i = 0; i < ret.length; i++) {
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
        if (memoryMappedBuffer != null)
            return;
        try {
            memoryMappedBuffer = new RandomAccessFile(path, "rw");
            long size = 8L * length;
            for (long offset = 0; offset < size; offset += MAPPING_SIZE) {
                long size2 = Math.min(size - offset, MAPPING_SIZE);
                mappings.add(memoryMappedBuffer.getChannel().map(FileChannel.MapMode.READ_WRITE, offset, size2));
            }
        } catch (IOException e) {
            try {
                if (memoryMappedBuffer != null)
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
        if (buffer != null)
            buffer = null;
        if (memoryMappedBuffer != null) {
            try {
                this.mappings.clear();
                this.memoryMappedBuffer.close();
            } catch (IOException e) {
                throw new RuntimeException(e);

            }
        }
    }
}
