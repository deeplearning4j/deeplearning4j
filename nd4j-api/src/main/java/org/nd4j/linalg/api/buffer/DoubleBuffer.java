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


/**
 * Double buffer implementation of data buffer
 *
 * @author Adam Gibson
 */
public class DoubleBuffer extends BaseDataBuffer {


    public DoubleBuffer(int length) {
        super(length);
    }

    public DoubleBuffer(double[] doubles, boolean copy) {
        super(doubles, copy);
    }


    @Override
    public DataBuffer.Type dataType() {
        return DataBuffer.Type.DOUBLE;
    }

    @Override
    public float[] asFloat() {
        return dataBuffer.nioBuffer().asFloatBuffer().array();
    }

    @Override
    public double[] asDouble() {
        return dataBuffer.nioBuffer().asDoubleBuffer().array();

    }

    @Override
    public int[] asInt() {
        return dataBuffer.nioBuffer().asIntBuffer().array();

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
        dataBuffer.setDouble(i, element);
    }

    @Override
    public void put(int i, int element) {
        put(i, (double) element);
    }


    @Override
    public int getInt(int ix) {
        return dataBuffer.getInt(ix);
    }



    @Override
    public void flush() {
        dataBuffer = null;
    }

    @Override
    public int getElementSize() {
        return 8;
    }



}
