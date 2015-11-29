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
 * Double buffer implementation of data buffer
 *
 * @author Adam Gibson
 */
public class DoubleBuffer extends BaseDataBuffer {


    public DoubleBuffer(int length) {
        super(length);
    }



    public DoubleBuffer(ByteBuf buf,int length) {
        super(buf,length);
    }

    public DoubleBuffer(double[] data) {
        super(data);
    }

    public DoubleBuffer(int[] data) {
        this(data, Nd4j.copyOnOps);
    }

    public DoubleBuffer(int[] data, boolean copyOnOps) {
        super(data, copyOnOps);
    }

    public DoubleBuffer(float[] data) {
        this(data, Nd4j.copyOnOps);
    }

    public DoubleBuffer(float[] data, boolean copyOnOps) {
        super(data, copyOnOps);
    }

    @Override
    public DataBuffer create(ByteBuf buf,int length) {
        return new DoubleBuffer(buf,length);
    }

    public DoubleBuffer(double[] doubles, boolean copy) {
        super(doubles, copy);
    }


    @Override
    public DataBuffer.Type dataType() {
        return DataBuffer.Type.DOUBLE;
    }




    @Override
    public float getFloat(int i) {
        return (float) getDouble(i);
    }

    @Override
    public Number getNumber(int i) {
        return  getDouble(i);
    }

    @Override
    public DataBuffer create(double[] data) {
        return new DoubleBuffer(data);
    }

    @Override
    public DataBuffer create(float[] data) {
        return new DoubleBuffer(data);
    }

    @Override
    public DataBuffer create(int[] data) {
        return new DoubleBuffer(data);
    }
    @Override
    protected DataBuffer create(int length) {
        return new DoubleBuffer(length);
    }


    @Override
    public void flush() {
    }

    @Override
    public int getElementSize() {
        return 8;
    }



}
