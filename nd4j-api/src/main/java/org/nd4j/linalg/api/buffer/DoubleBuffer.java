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


import com.google.common.primitives.Bytes;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.UUID;

/**
 * Double buffer implementation of data buffer
 *
 * @author Adam Gibson
 */
public class DoubleBuffer extends BaseDataBuffer {

    private double[] buffer;

    public DoubleBuffer(int length) {
        super(length);
        this.buffer = new double[length];
        Nd4j.getResourceManager().incrementCurrentAllocatedMemory(elementSize() * length);
    }


    public DoubleBuffer(double[] buffer) {
        this(buffer, true);
    }

    public DoubleBuffer(double[] buffer, boolean copy) {
        super(buffer.length);
        this.buffer = copy ? Arrays.copyOf(buffer, buffer.length) : buffer;
    }


    @Override
    public void setData(int[] data) {
        this.buffer = ArrayUtil.toDoubles(data);
    }

    @Override
    public void setData(float[] data) {
        this.buffer = ArrayUtil.toDoubles(data);
    }

    @Override
    public void setData(double[] data) {
        this.buffer = data;
    }

    @Override
    public byte[] asBytes() {
        byte[][] ret1 = new byte[length][];
        for (int i = 0; i < length; i++) {
            ret1[i] = toByteArray(buffer[i]);
        }

        return Bytes.concat(ret1);
    }

    @Override
    public int dataType() {
        return DataBuffer.DOUBLE;
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
    public double[] asDouble() {
        if (buffer == null) {
            buffer = new double[length];
            for (int i = 0; i < length; i++) {
                buffer[i] = getDouble(i);
            }
            try {
                mappings.clear();
                memoryMappedBuffer.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return buffer;
    }

    @Override
    public int[] asInt() {
        int[] ret = new int[length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = (int) buffer[i];
        }
        return ret;
    }


    @Override
    public double getDouble(int i) {
        if (buffer != null)
            return buffer[i];
        else {
            long p = i * 8;
            int mapN = (int) (p / MAPPING_SIZE);
            int offN = (int) (p % MAPPING_SIZE);
            return mappings.get(mapN).getDouble(offN);
        }
    }


    @Override
    public void assign(Number value, int offset) {
        for (int i = offset; i < length(); i++) {
            buffer[i] = value.doubleValue();
        }
    }

    @Override
    public float getFloat(int i) {
        return (float) getDouble(i);
    }

    @Override
    public Number getNumber(int i) {
        return (int) getDouble(i);
    }


    @Override
    public void put(int i, float element) {
        put(i, (double) element);

    }

    @Override
    public void put(int i, double element) {
        if (buffer != null)
            buffer[i] = element;
        else {
            long p = i * 8;
            int mapN = (int) (p / MAPPING_SIZE);
            int offN = (int) (p % MAPPING_SIZE);
            mappings.get(mapN).putDouble(offN, element);
        }
    }

    @Override
    public void put(int i, int element) {
        put(i, (double) element);
    }


    @Override
    public int getInt(int ix) {
        return (int) buffer[ix];
    }

    @Override
    public DataBuffer dup() {
        return new DoubleBuffer(buffer);
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
    public int elementSize() {
        return 8;
    }

    @Override
    public void destroy() {
        super.destroy();
        if (buffer != null)
            buffer = null;
        if (memoryMappedBuffer != null) {
            try {
                this.memoryMappedBuffer.close();
                mappings.clear();
            } catch (IOException e) {
                throw new RuntimeException(e);

            }
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DoubleBuffer)) return false;

        DoubleBuffer that = (DoubleBuffer) o;

        if (!Arrays.equals(buffer, that.buffer)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return buffer != null ? Arrays.hashCode(buffer) : 0;
    }
}
