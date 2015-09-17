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



    public FloatBuffer(ByteBuf buf,int length) {
        super(buf,length);
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

    public FloatBuffer(double[] data, boolean copyOnOps) {
        super(data,copyOnOps);
    }



    @Override
    public DataBuffer create(ByteBuf buf,int length) {
        return new FloatBuffer(buf,length);
    }

    public FloatBuffer(float[] floats, boolean copy) {
        super(floats, copy);
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