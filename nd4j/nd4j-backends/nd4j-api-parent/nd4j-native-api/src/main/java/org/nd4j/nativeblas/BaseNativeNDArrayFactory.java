/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.nativeblas;

import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.LongIndexer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.BaseNDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.MemcpyDirection;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;

/**
 * Base class with {@link NativeOps}
 *
 * @author Adam Gibson
 */
public abstract class BaseNativeNDArrayFactory extends BaseNDArrayFactory {

    protected NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    public BaseNativeNDArrayFactory(DataType dtype, Character order) {
        super(dtype, order);
    }

    public BaseNativeNDArrayFactory(DataType dtype, char order) {
        super(dtype, order);
    }

    public BaseNativeNDArrayFactory() {}



    @Override
    public Pointer convertToNumpy(INDArray array) {
        LongPointer size = new LongPointer(1);
        Pointer header = NativeOpsHolder
                .getInstance().getDeviceNativeOps()
                .numpyHeaderForNd4j(
                        array.data().pointer(),
                        array.shapeInfoDataBuffer().pointer(),
                        array.data().getElementSize()
                        ,size);
        header.capacity(size.get());
        header.position(0);

        char[] magic = {'\\','x','9','3','N','U','M','P','Y','1','0'};

        BytePointer magicPointer = new BytePointer(new String(magic).getBytes());
        BytePointer bytePointer = new BytePointer(magicPointer.capacity() + (int) (size.get() + (array.data().getElementSize() * array.data().length())));
        BytePointer headerCast = new BytePointer(header);
        int pos = 0;
        Pointer.memcpy(bytePointer,magicPointer,magicPointer.capacity());
        pos += (magicPointer.capacity() - 1);
        bytePointer.position(pos);
        Pointer.memcpy(bytePointer,headerCast,headerCast.capacity());
        pos += (headerCast.capacity() - 1);
        bytePointer.position(pos);
        Pointer.memcpy(bytePointer,array.data().pointer(),(array.data().getElementSize() * array.data().length()));
        bytePointer.position(0);
        return bytePointer;
    }

    /**
     * Create from an in memory numpy pointer.
     * Note that this is heavily used
     * in our python library jumpy.
     *
     * @param pointer the pointer to the
     *                numpy array
     * @return an ndarray created from the in memory
     * numpy pointer
     */
    @Override
    public INDArray createFromNpyPointer(Pointer pointer) {
        Pointer dataPointer = nativeOps.dataPointForNumpy(pointer);
        int dataBufferElementSize = nativeOps.elementSizeForNpyArray(pointer);
        DataBuffer data = null;
        Pointer shapeBufferPointer = nativeOps.shapeBufferForNumpy(pointer);
        int length = nativeOps.lengthForShapeBufferPointer(shapeBufferPointer);
        shapeBufferPointer.capacity(8 * length);
        shapeBufferPointer.limit(8 * length);
        shapeBufferPointer.position(0);


        val intPointer = new LongPointer(shapeBufferPointer);
        val newPointer = new LongPointer(length);

        val perfD = PerformanceTracker.getInstance().helperStartTransaction();

        Pointer.memcpy(newPointer, intPointer, shapeBufferPointer.limit());

        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfD, shapeBufferPointer.limit(), MemcpyDirection.HOST_TO_HOST);

        DataBuffer shapeBuffer = Nd4j.createBuffer(
                newPointer,
                DataType.LONG,
                length,
                LongIndexer.create(newPointer));

        dataPointer.position(0);
        dataPointer.limit(dataBufferElementSize * Shape.length(shapeBuffer));
        dataPointer.capacity(dataBufferElementSize * Shape.length(shapeBuffer));


        if(dataBufferElementSize == (Float.SIZE / 8)) {
            FloatPointer dPointer = new FloatPointer(dataPointer.limit() / dataBufferElementSize);

            val perfX = PerformanceTracker.getInstance().helperStartTransaction();

            Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

            PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

            data = Nd4j.createBuffer(dPointer,
                    DataType.FLOAT,
                    Shape.length(shapeBuffer),
                    FloatIndexer.create(dPointer));
        }
        else if(dataBufferElementSize == (Double.SIZE / 8)) {
            DoublePointer dPointer = new DoublePointer(dataPointer.limit() / dataBufferElementSize);

            val perfX = PerformanceTracker.getInstance().helperStartTransaction();

            Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

            PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

            data = Nd4j.createBuffer(dPointer,
                    DataType.DOUBLE,
                    Shape.length(shapeBuffer),
                    DoubleIndexer.create(dPointer));
        }

        INDArray ret = Nd4j.create(data,
                Shape.shape(shapeBuffer),
                Shape.strideArr(shapeBuffer),
                0,
                Shape.order(shapeBuffer));

        return ret;
    }

    @Override
    public INDArray createFromNpyHeaderPointer(Pointer pointer) {
        Pointer dataPointer = nativeOps.dataPointForNumpyHeader(pointer);
        int dataBufferElementSize = nativeOps.elementSizeForNpyArrayHeader(pointer);
        DataBuffer data = null;
        Pointer shapeBufferPointer = nativeOps.shapeBufferForNumpyHeader(pointer);
        int length = nativeOps.lengthForShapeBufferPointer(shapeBufferPointer);
        shapeBufferPointer.capacity(8 * length);
        shapeBufferPointer.limit(8 * length);
        shapeBufferPointer.position(0);


        val intPointer = new LongPointer(shapeBufferPointer);
        val newPointer = new LongPointer(length);

        val perfD = PerformanceTracker.getInstance().helperStartTransaction();

        Pointer.memcpy(newPointer, intPointer, shapeBufferPointer.limit());

        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfD, shapeBufferPointer.limit(), MemcpyDirection.HOST_TO_HOST);

        DataBuffer shapeBuffer = Nd4j.createBuffer(
                newPointer,
                DataType.LONG,
                length,
                LongIndexer.create(newPointer));

        dataPointer.position(0);
        dataPointer.limit(dataBufferElementSize * Shape.length(shapeBuffer));
        dataPointer.capacity(dataBufferElementSize * Shape.length(shapeBuffer));


        if(dataBufferElementSize == (Float.SIZE / 8)) {
            FloatPointer dPointer = new FloatPointer(dataPointer.limit() / dataBufferElementSize);

            val perfX = PerformanceTracker.getInstance().helperStartTransaction();

            Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

            PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

            data = Nd4j.createBuffer(dPointer,
                    DataType.FLOAT,
                    Shape.length(shapeBuffer),
                    FloatIndexer.create(dPointer));
        }
        else if(dataBufferElementSize == (Double.SIZE / 8)) {
            DoublePointer dPointer = new DoublePointer(dataPointer.limit() / dataBufferElementSize);

            val perfX = PerformanceTracker.getInstance().helperStartTransaction();

            Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

            PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

            data = Nd4j.createBuffer(dPointer,
                    DataType.DOUBLE,
                    Shape.length(shapeBuffer),
                    DoubleIndexer.create(dPointer));
        }

        INDArray ret = Nd4j.create(data,
                Shape.shape(shapeBuffer),
                Shape.strideArr(shapeBuffer),
                0,
                Shape.order(shapeBuffer));

        return ret;
    }


    /**
     * Create from a given numpy file.
     *
     * @param file the file to create the ndarray from
     * @return the created ndarray
     */
    @Override
    public INDArray createFromNpyFile(File file) {
        byte[] pathBytes = file.getAbsolutePath().getBytes(Charset.forName("UTF-8"));
        ByteBuffer directBuffer = ByteBuffer.allocateDirect(pathBytes.length).order(ByteOrder.nativeOrder());
        directBuffer.put(pathBytes);
        directBuffer.rewind();
        directBuffer.position(0);
        Pointer pointer = nativeOps.numpyFromFile(new BytePointer(directBuffer));

        INDArray result = createFromNpyPointer(pointer);

        // releasing original pointer here
        nativeOps.releaseNumpy(pointer);
        return result;
    }

}
