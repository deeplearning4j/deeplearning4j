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

package org.nd4j.linalg.api.buffer;


import io.netty.buffer.ByteBuf;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteBuffer;

/**
 * Data buffer for floats
 *
 * @author Adam Gibson
 */
public class FloatBuffer extends BaseDataBuffer {
    /**
     * Create a float buffer with the given length
     * @param length the float buffer with the given length
     */
    public FloatBuffer(int length) {
        super(length);
    }

    public FloatBuffer(int length, int elementSize) {
        super(length, elementSize);
    }

    public FloatBuffer(int length, int elementSize, int offset) {
        super(length, elementSize, offset);
    }

    public FloatBuffer(DataBuffer underlyingBuffer, int length, int offset) {
        super(underlyingBuffer, length, offset);
    }

    public FloatBuffer(ByteBuf buf,int length) {
        super(buf,length);
    }

    public FloatBuffer(ByteBuf buf, int length, int offset) {
        super(buf, length, offset);
    }

    public FloatBuffer(float[] data) {
        this(data, Nd4j.copyOnOps);
    }

    public FloatBuffer(int[] data) {
        this(data,Nd4j.copyOnOps);
    }

    public FloatBuffer(double[] data) {
        this(data,Nd4j.copyOnOps);
    }

    public FloatBuffer(int[] data, boolean copyOnOps) {
        super(data, copyOnOps);
    }

    public FloatBuffer(int[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    public FloatBuffer(double[] data, boolean copyOnOps) {
        super(data,copyOnOps);
    }

    public FloatBuffer(double[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    public FloatBuffer(ByteBuffer buffer,int length) {
        super(buffer,length);
    }

    public FloatBuffer(ByteBuffer buffer, int length, int offset) {
        super(buffer, length, offset);
    }

    public FloatBuffer(byte[] data, int length) {
        super(data, length);
    }


    @Override
    public DataBuffer create(ByteBuf buf,int length) {
        return new FloatBuffer(buf,length);
    }

    public FloatBuffer(float[] floats, boolean copy) {
        super(floats, copy);
    }

    public FloatBuffer(float[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    @Override
    public int getElementSize() {
        return 4;
    }



    @Override
    protected DataBuffer create(int length) {
        return new FloatBuffer(length);
    }

    @Override
    public DataBuffer.Type dataType() {
        return DataBuffer.Type.FLOAT;
    }


    @Override
    public DataBuffer create(double[] data) {
        return new FloatBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new FloatBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new FloatBuffer(data);
    }


}