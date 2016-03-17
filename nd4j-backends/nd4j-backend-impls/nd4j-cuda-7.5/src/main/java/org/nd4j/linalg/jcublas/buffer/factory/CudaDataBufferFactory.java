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

package org.nd4j.linalg.jcublas.buffer.factory;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaFloatDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaIntDataBuffer;
import org.nd4j.linalg.util.ArrayUtil;

import java.nio.ByteBuffer;

/**
 * Creates cuda buffers
 *
 * @author Adam Gibson
 */
public class CudaDataBufferFactory implements DataBufferFactory {
    protected DataBuffer.AllocationMode allocationMode;

    @Override
    public void setAllocationMode(DataBuffer.AllocationMode allocationMode) {
        this.allocationMode = allocationMode;
    }

    @Override
    public DataBuffer.AllocationMode allocationMode() {
        if(allocationMode == null) {
            String otherAlloc = System.getProperty("alloc");
            if(otherAlloc.equals("heap"))
                setAllocationMode(DataBuffer.AllocationMode.HEAP);
            else if(otherAlloc.equals("direct"))
                setAllocationMode(DataBuffer.AllocationMode.DIRECT);
            else if(otherAlloc.equals("javacpp"))
                setAllocationMode(DataBuffer.AllocationMode.JAVACPP);
        }
        return allocationMode;
    }

    @Override
    public DataBuffer create(DataBuffer underlyingBuffer, int offset, int length) {
        if(underlyingBuffer.dataType() == DataBuffer.Type.DOUBLE) {
           return new CudaDoubleDataBuffer(underlyingBuffer,length,offset);
        }
        else if(underlyingBuffer.dataType() == DataBuffer.Type.FLOAT) {
            return new CudaFloatDataBuffer(underlyingBuffer,length,offset);

        }
        else if(underlyingBuffer.dataType() == DataBuffer.Type.INT) {
            return new CudaIntDataBuffer(underlyingBuffer,length,offset);

        }
        return null;
    }

    @Override
    public DataBuffer createInt(int offset, ByteBuffer buffer, int length) {
        return new CudaIntDataBuffer(buffer,length,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, ByteBuffer buffer, int length) {
        return new CudaFloatDataBuffer(buffer,length,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, ByteBuffer buffer, int length) {
        return new CudaDoubleDataBuffer(buffer,length,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, int length) {
        return new CudaDoubleDataBuffer(length,8,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, int length) {
        return new CudaFloatDataBuffer(length,4,length);
    }

    @Override
    public DataBuffer createInt(int offset, int length) {
        return new CudaIntDataBuffer(length,4,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, int[] data) {
        return new CudaDoubleDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, int[] data) {
        return new CudaFloatDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createInt(int offset, int[] data) {
        return new CudaIntDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, double[] data) {
        return new CudaDoubleDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, byte[] data, int length) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDoubleArray(data),true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, byte[] data, int length) {
        return new CudaFloatDataBuffer(ArrayUtil.toDoubleArray(data),true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, double[] data) {
        return new CudaFloatDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createInt(int offset, double[] data) {
        return new CudaIntDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, float[] data) {
        return new CudaDoubleDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, float[] data) {
        return new CudaFloatDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createInt(int offset, float[] data) {
        return new CudaIntDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, int[] data, boolean copy) {
        return new CudaDoubleDataBuffer(data,true,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, int[] data, boolean copy) {
        return new CudaFloatDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createInt(int offset, int[] data, boolean copy) {
        return new CudaIntDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, double[] data, boolean copy) {
        return new CudaDoubleDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, double[] data, boolean copy) {
        return new CudaFloatDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createInt(int offset, double[] data, boolean copy) {
        return new CudaIntDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createDouble(int offset, float[] data, boolean copy) {
        return new CudaDoubleDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createFloat(int offset, float[] data, boolean copy) {
        return new CudaFloatDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createInt(int offset, float[] data, boolean copy) {
        return new CudaIntDataBuffer(data,copy,offset);
    }

    @Override
    public DataBuffer createInt(ByteBuffer buffer, int length) {
        return new CudaIntDataBuffer(buffer,length);
    }

    @Override
    public DataBuffer createFloat(ByteBuffer buffer, int length) {
        return new CudaFloatDataBuffer(buffer,length);
    }

    @Override
    public DataBuffer createDouble(ByteBuffer buffer, int length) {
        return new CudaDoubleDataBuffer(buffer,length);
    }

    @Override
    public DataBuffer createDouble(int length) {
        return new CudaDoubleDataBuffer(length);
    }

    @Override
    public DataBuffer createFloat(int length) {
        return new CudaFloatDataBuffer(length);
    }

    @Override
    public DataBuffer createInt(int length) {
        return new CudaIntDataBuffer(length);
    }

    @Override
    public DataBuffer createDouble(int[] data) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDoubles(data));
    }

    @Override
    public DataBuffer createFloat(int[] data) {
        return new CudaFloatDataBuffer(ArrayUtil.toFloats(data));
    }

    @Override
    public DataBuffer createInt(int[] data) {
        return new CudaIntDataBuffer(data);
    }

    @Override
    public DataBuffer createDouble(double[] data) {
        return new CudaDoubleDataBuffer(data);
    }

    @Override
    public DataBuffer createDouble(byte[] data, int length) {
        return new CudaDoubleDataBuffer(data, length);
    }

    @Override
    public DataBuffer createFloat(byte[] data, int length) {
        return new CudaFloatDataBuffer(data, length);
    }

    @Override
    public DataBuffer createFloat(double[] data) {
        return new CudaFloatDataBuffer(ArrayUtil.toFloats(data));
    }

    @Override
    public DataBuffer createInt(double[] data) {
        return new CudaIntDataBuffer(ArrayUtil.toInts(data));
    }

    @Override
    public DataBuffer createDouble(float[] data) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDoubles(data));
    }

    @Override
    public DataBuffer createFloat(float[] data) {
        return new CudaFloatDataBuffer(data);
    }

    @Override
    public DataBuffer createInt(float[] data) {
        return new CudaIntDataBuffer(ArrayUtil.toInts(data));
    }

    @Override
    public DataBuffer createDouble(int[] data, boolean copy) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDouble(data));
    }

    @Override
    public DataBuffer createFloat(int[] data, boolean copy) {
        return new CudaFloatDataBuffer(ArrayUtil.toFloats(data));
    }

    @Override
    public DataBuffer createInt(int[] data, boolean copy) {
        return new CudaIntDataBuffer(data);
    }

    @Override
    public DataBuffer createDouble(double[] data, boolean copy) {
        return new CudaDoubleDataBuffer(data);
    }

    @Override
    public DataBuffer createFloat(double[] data, boolean copy) {
        return new CudaFloatDataBuffer(ArrayUtil.toFloats(data));
    }

    @Override
    public DataBuffer createInt(double[] data, boolean copy) {
        return new CudaIntDataBuffer(ArrayUtil.toInts(data));
    }

    @Override
    public DataBuffer createDouble(float[] data, boolean copy) {
        return new CudaDoubleDataBuffer(ArrayUtil.toDoubles(data));
    }

    @Override
    public DataBuffer createFloat(float[] data, boolean copy) {
        return new CudaFloatDataBuffer(data);
    }

    @Override
    public DataBuffer createInt(float[] data, boolean copy) {
        return new CudaIntDataBuffer(ArrayUtil.toInts(data));
    }
}