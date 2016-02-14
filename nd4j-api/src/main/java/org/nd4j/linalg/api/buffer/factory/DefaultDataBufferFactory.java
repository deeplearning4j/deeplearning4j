/*
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

package org.nd4j.linalg.api.buffer.factory;

import io.netty.buffer.Unpooled;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.nd4j.linalg.util.ArrayUtil;

import java.nio.ByteBuffer;

/**
 * Normal data buffer creation
 *
 * @author Adam Gibson
 */
public class DefaultDataBufferFactory implements DataBufferFactory {
    @Override
    public DataBuffer create(DataBuffer underlyingBuffer, int offset, int length) {
        if(underlyingBuffer.dataType() == DataBuffer.Type.DOUBLE) {
            return new DoubleBuffer(underlyingBuffer,length,offset);
        }
        else if(underlyingBuffer.dataType() == DataBuffer.Type.FLOAT) {
            return new FloatBuffer(underlyingBuffer,length,offset);

        }
        else if(underlyingBuffer.dataType() == DataBuffer.Type.INT) {
            return new IntBuffer(underlyingBuffer,length,offset);
        }
        return null;
    }

    @Override
    public DataBuffer createInt(int offset, ByteBuffer buffer, int length) {
        return new IntBuffer(buffer,length,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, ByteBuffer buffer, int length) {
        return new FloatBuffer(buffer,length,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, ByteBuffer buffer, int length) {
        return new DoubleBuffer(buffer,length,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, int length) {
        return new DoubleBuffer(length,8,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, int length) {
        return new FloatBuffer(length,4,offset);
    }

    @Override
    public DataBuffer createInt(int offset, int length) {
        return new IntBuffer(length,4,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, int[] data) {
        return createDouble(offset,data,true);
    }

    @Override
    public DataBuffer createFloat(int offset, int[] data) {
        FloatBuffer ret = new FloatBuffer(ArrayUtil.toFloats(data),true,offset);
        return ret;
    }

    @Override
    public DataBuffer createInt(int offset, int[] data) {
        return new IntBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, double[] data) {
        return new DoubleBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, byte[] data, int length) {
     return createDouble(offset,ArrayUtil.toDoubleArray(data),true);
    }

    @Override
    public DataBuffer createFloat(int offset, byte[] data, int length) {
        return createFloat(offset,ArrayUtil.toFloatArray(data),true);
    }

    @Override
    public DataBuffer createFloat(int offset, double[] data) {
        return new FloatBuffer(ArrayUtil.toFloats(data),true,offset);
    }

    @Override
    public DataBuffer createInt(int offset, double[] data) {
        return new IntBuffer(ArrayUtil.toInts(data),true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, float[] data) {
        return new DoubleBuffer(ArrayUtil.toDoubles(data),true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, float[] data) {
        return new FloatBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createInt(int offset, float[] data) {
        return new IntBuffer(ArrayUtil.toInts(data),true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, int[] data, boolean copy) {
        return new DoubleBuffer(ArrayUtil.toDoubles(data),true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, int[] data, boolean copy) {
        return new FloatBuffer(ArrayUtil.toFloats(data),copy,offset);
    }

    @Override
    public DataBuffer createInt(int offset, int[] data, boolean copy) {
        return new IntBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, double[] data, boolean copy) {
        return new DoubleBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, double[] data, boolean copy) {
        return new FloatBuffer(ArrayUtil.toFloats(data),copy,offset);
    }

    @Override
    public DataBuffer createInt(int offset, double[] data, boolean copy) {
        return new IntBuffer(ArrayUtil.toInts(data),copy,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, float[] data, boolean copy) {
        return new DoubleBuffer(ArrayUtil.toDoubles(data),copy,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, float[] data, boolean copy) {
        return  new FloatBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createInt(int offset, float[] data, boolean copy) {
        return new IntBuffer(ArrayUtil.toInts(data),copy,offset);
    }

    @Override
    public DataBuffer createInt(ByteBuffer buffer, int length) {
        return new IntBuffer(buffer,length);
    }

    @Override
    public DataBuffer createFloat(ByteBuffer buffer, int length) {
        return new FloatBuffer(buffer,length);
    }

    @Override
    public DataBuffer createDouble(ByteBuffer buffer, int length) {
        return new DoubleBuffer(buffer,length);
    }

    @Override
    public DataBuffer createDouble(int length) {
        return new DoubleBuffer(length);
    }

    @Override
    public DataBuffer createFloat(int length) {
        return new FloatBuffer(length);
    }

    @Override
    public DataBuffer createInt(int length) {
        return new IntBuffer(length);
    }

    @Override
    public DataBuffer createDouble(int[] data) {
        return createDouble(data, true);
    }

    @Override
    public DataBuffer createFloat(int[] data) {
        return createFloat(data, true);
    }

    @Override
    public DataBuffer createInt(int[] data) {
        return createInt(data, true);
    }

    @Override
    public DataBuffer createDouble(double[] data) {
        return createDouble(data, true);
    }

    @Override
    public DataBuffer createDouble(byte[] data,int length) {
        return new FloatBuffer(Unpooled.wrappedBuffer(data),length);
    }

    @Override
    public DataBuffer createFloat(byte[] data,int length) {
        return new DoubleBuffer(Unpooled.wrappedBuffer(data),length);
    }

    @Override
    public DataBuffer createFloat(double[] data) {
        return createFloat(data, true);
    }

    @Override
    public DataBuffer createInt(double[] data) {
        return createInt(data, true);
    }

    @Override
    public DataBuffer createDouble(float[] data) {
        return createDouble(data, true);
    }

    @Override
    public DataBuffer createFloat(float[] data) {
        return createFloat(data, true);
    }

    @Override
    public DataBuffer createInt(float[] data) {
        return createInt(data, true);
    }

    @Override
    public DataBuffer createDouble(int[] data, boolean copy) {

        return new DoubleBuffer(ArrayUtil.toDoubles(data), copy);
    }

    @Override
    public DataBuffer createFloat(int[] data, boolean copy) {
        return new FloatBuffer(ArrayUtil.toFloats(data), copy);
    }

    @Override
    public DataBuffer createInt(int[] data, boolean copy) {
        return new IntBuffer(data, copy);
    }

    @Override
    public DataBuffer createDouble(double[] data, boolean copy) {
        return new DoubleBuffer(data, copy);
    }

    @Override
    public DataBuffer createFloat(double[] data, boolean copy) {
        return new FloatBuffer(ArrayUtil.toFloats(data), copy);
    }

    @Override
    public DataBuffer createInt(double[] data, boolean copy) {
        return new IntBuffer(ArrayUtil.toInts(data), copy);
    }

    @Override
    public DataBuffer createDouble(float[] data, boolean copy) {
        return new FloatBuffer(data, copy);
    }

    @Override
    public DataBuffer createFloat(float[] data, boolean copy) {
        return new FloatBuffer(data, copy);
    }

    @Override
    public DataBuffer createInt(float[] data, boolean copy) {
        return new IntBuffer(ArrayUtil.toInts(data), copy);
    }
}