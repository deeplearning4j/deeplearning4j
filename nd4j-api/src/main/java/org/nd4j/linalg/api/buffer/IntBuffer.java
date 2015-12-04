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

import java.nio.ByteBuffer;

/**
 * Int buffer
 *
 * @author Adam Gibson
 */
public class IntBuffer extends BaseDataBuffer {


    public IntBuffer(int length) {
        super(length);
    }

    public IntBuffer(ByteBuffer buffer,int length) {
        super(buffer,length);
    }

    @Override
    protected DataBuffer create(int length) {
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
