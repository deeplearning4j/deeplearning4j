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
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;

import java.nio.ByteBuffer;

/**
 * Int buffer
 *
 * @author Adam Gibson
 */
public class IntBuffer extends BaseDataBuffer {

    public IntBuffer(long length) {
        super(length);
    }

    protected IntBuffer(ByteBuf buf, int length, int offset) {
        super(buf, length, offset);
    }

    public IntBuffer(ByteBuffer buffer, int length, int offset) {
        super(buffer, length, offset);
    }

    public IntBuffer(byte[] data, int length) {
        super(data, length);
    }

    public IntBuffer(double[] data, boolean copy) {
        super(data, copy);
    }

    public IntBuffer(double[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    public IntBuffer(float[] data, boolean copy) {
        super(data, copy);
    }

    public IntBuffer(float[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    public IntBuffer(int[] data, boolean copy, int offset) {
        super(data, copy, offset);
    }

    public IntBuffer(int length, int elementSize) {
        super(length, elementSize);
    }

    public IntBuffer(int length, int elementSize, int offset) {
        super(length, elementSize, offset);
    }

    public IntBuffer(DataBuffer underlyingBuffer, long length, long offset) {
        super(underlyingBuffer, length, offset);
    }

    public IntBuffer(ByteBuffer buffer,int length) {
        super(buffer,length);
    }

    @Override
    protected DataBuffer create(long length) {
        return new IntBuffer(length);
    }

    public IntBuffer(int[] data) {
        super(data);
    }

    public IntBuffer(double[] data) {
        super(data);
    }

    public IntBuffer(float[] data) {
        super(data);
    }

    @Override
    public DataBuffer create(double[] data) {
        return new IntBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
       return new IntBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
       return new IntBuffer(data);
    }

    public IntBuffer(ByteBuf buf,int length) {
        super(buf,length);
    }

    @Override
    public DataBuffer create(ByteBuf buf,int length) {
        return new IntBuffer(buf,length);
    }

    @Override
    public IComplexFloat getComplexFloat(long i) {
        return null;
    }

    @Override
    public IComplexDouble getComplexDouble(long i) {
        throw new UnsupportedOperationException();

    }

    public IntBuffer(int[] data, boolean copy) {
        super(data, copy);
    }


    @Override
    public DataBuffer.Type dataType() {
        return DataBuffer.Type.INT;
    }



    @Override
    public int getElementSize() {
        return 4;
    }

}
