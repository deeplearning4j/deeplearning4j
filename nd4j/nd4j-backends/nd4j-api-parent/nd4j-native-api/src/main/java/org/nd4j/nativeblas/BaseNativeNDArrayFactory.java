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

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.BaseNDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.MemcpyDirection;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Map;

/**
 * Base class with {@link NativeOps}
 *
 * @author Adam Gibson
 */
@Slf4j
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
        val size = new LongPointer(1);
        Pointer header = NativeOpsHolder
                .getInstance().getDeviceNativeOps()
                .numpyHeaderForNd4j(
                        array.data().pointer(),
                        array.shapeInfoDataBuffer().pointer(),
                        array.data().getElementSize()
                        ,size);

        val headerSize = size.get() - 1;
        header.capacity(headerSize);
        header.position(0);



        BytePointer bytePointer = new BytePointer((int) (headerSize + (array.data().getElementSize() * array.data().length())));
        BytePointer headerCast = new BytePointer(header);
        val indexer = ByteIndexer.create(headerCast);
        int pos = 0;
        bytePointer.position(pos);
        Pointer.memcpy(bytePointer, headerCast,headerCast.capacity());
        pos += (headerCast.capacity());
        bytePointer.position(pos);

        // make sure data is copied to the host memory
        Nd4j.getAffinityManager().ensureLocation(array, AffinityManager.Location.HOST);

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

        val jvmShapeInfo = shapeBuffer.asLong();
        val dtype = ArrayOptionsHelper.dataType(jvmShapeInfo);

        switch (dtype) {
            case UBYTE: {
                val dPointer = new BytePointer(dataPointer.limit() / dataBufferElementSize);
                val perfX = PerformanceTracker.getInstance().helperStartTransaction();

                Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

                data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        UByteIndexer.create(dPointer));
            }
            break;
            case BYTE: {
                val dPointer = new BytePointer(dataPointer.limit() / dataBufferElementSize);
                val perfX = PerformanceTracker.getInstance().helperStartTransaction();

                Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

                data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        ByteIndexer.create(dPointer));
            }
            break;
            case UINT64:
            case LONG: {
                val dPointer = new LongPointer(dataPointer.limit() / dataBufferElementSize);
                val perfX = PerformanceTracker.getInstance().helperStartTransaction();

                Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

                data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        LongIndexer.create(dPointer));
            }
            break;
            case UINT32:
            case INT: {
                val dPointer = new IntPointer(dataPointer.limit() / dataBufferElementSize);
                val perfX = PerformanceTracker.getInstance().helperStartTransaction();

                Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

                data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        IntIndexer.create(dPointer));
            }
            break;
            case UINT16: {
                val dPointer = new ShortPointer(dataPointer.limit() / dataBufferElementSize);
                val perfX = PerformanceTracker.getInstance().helperStartTransaction();

                Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

                data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        UShortIndexer.create(dPointer));
            }
            break;
            case SHORT: {
                val dPointer = new ShortPointer(dataPointer.limit() / dataBufferElementSize);
                val perfX = PerformanceTracker.getInstance().helperStartTransaction();

                Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

                data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        ShortIndexer.create(dPointer));
            }
            break;
            case BFLOAT16:
            case HALF: {
                val dPointer = new ShortPointer(dataPointer.limit() / dataBufferElementSize);
                val perfX = PerformanceTracker.getInstance().helperStartTransaction();

                Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

                data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        HalfIndexer.create(dPointer));
            }
            break;
            case FLOAT: {
                val dPointer = new FloatPointer(dataPointer.limit() / dataBufferElementSize);
                val perfX = PerformanceTracker.getInstance().helperStartTransaction();

                Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

                data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        FloatIndexer.create(dPointer));
            }
            break;
            case DOUBLE: {
                val dPointer = new DoublePointer(dataPointer.limit() / dataBufferElementSize);
                val perfX = PerformanceTracker.getInstance().helperStartTransaction();

                Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

                data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        DoubleIndexer.create(dPointer));
            }
            break;
        }

        INDArray ret = Nd4j.create(data,
                Shape.shape(shapeBuffer),
                Shape.strideArr(shapeBuffer),
                0,
                Shape.order(shapeBuffer));

        Nd4j.getAffinityManager().tagLocation(ret, AffinityManager.Location.DEVICE);

        return ret;
    }

    @Override
    public INDArray createFromNpyHeaderPointer(Pointer pointer) {
        val dtype = DataType.fromInt(nativeOps.dataTypeFromNpyHeader(pointer));

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

        val perfX = PerformanceTracker.getInstance().helperStartTransaction();

        switch (dtype) {
            case BYTE: {
                    val dPointer = new BytePointer(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.limit(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.capacity(dataPointer.limit() / dataBufferElementSize);
                    Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                    data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        ByteIndexer.create(dPointer));
                }
                break;
            case SHORT: {
                    val dPointer = new ShortPointer(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.limit(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.capacity(dataPointer.limit() / dataBufferElementSize);
                    Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                    data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        ShortIndexer.create(dPointer));
                }
                break;
            case INT: {
                    val dPointer = new IntPointer(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.limit(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.capacity(dataPointer.limit() / dataBufferElementSize);
                    Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                    data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        IntIndexer.create(dPointer));
                }
                break;
            case LONG: {
                    val dPointer = new LongPointer(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.limit(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.capacity(dataPointer.limit() / dataBufferElementSize);
                    Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                    data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        LongIndexer.create(dPointer));
                }
                break;
            case UBYTE: {
                    val dPointer = new BytePointer(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.limit(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.capacity(dataPointer.limit() / dataBufferElementSize);
                    Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                    data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        UByteIndexer.create(dPointer));
                }
                break;
            case UINT16: {
                    val dPointer = new ShortPointer(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.limit(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.capacity(dataPointer.limit() / dataBufferElementSize);
                    Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                    data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        UShortIndexer.create(dPointer));
                }
                break;
            case UINT32: {
                    val dPointer = new IntPointer(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.limit(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.capacity(dataPointer.limit() / dataBufferElementSize);
                    Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                    data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        IntIndexer.create(dPointer));
                }
                break;
            case UINT64: {
                    val dPointer = new LongPointer(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.limit(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.capacity(dataPointer.limit() / dataBufferElementSize);
                    Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                    data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        LongIndexer.create(dPointer));
                }
                break;
            case HALF: {
                    val dPointer = new ShortPointer(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.limit(dataPointer.limit() / dataBufferElementSize);
                    //dPointer.capacity(dataPointer.limit() / dataBufferElementSize);
                    Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                    data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        HalfIndexer.create(dPointer));
                }
                break;
            case FLOAT: {
                // TODO: we might want to skip copy, and use existing pointer/data here
                val dPointer = new FloatPointer(dataPointer.limit() / dataBufferElementSize);
                Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        FloatIndexer.create(dPointer));
                }
                break;
            case DOUBLE: {
                // TODO: we might want to skip copy, and use existing pointer/data here
                val dPointer = new DoublePointer(dataPointer.limit() / dataBufferElementSize);
                Pointer.memcpy(dPointer, dataPointer, dataPointer.limit());

                data = Nd4j.createBuffer(dPointer,
                        dtype,
                        Shape.length(shapeBuffer),
                        DoubleIndexer.create(dPointer));
            }
            break;
            default:
                throw new RuntimeException("Unsupported data type: [" + dtype + "]");
        }

        PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, dataPointer.limit(), MemcpyDirection.HOST_TO_HOST);

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

    @Override
    public Map<String, INDArray> createFromNpzFile(File file) throws Exception{

        // TODO error checks
        HashMap<String, INDArray> map = new HashMap<>();
        InputStream is = new FileInputStream(file);
        while(true){
            byte[] localHeader = new byte[30];
            is.read(localHeader);
            if ((int)localHeader[2] != 3 || (int)localHeader[3] != 4){
                if(map.isEmpty()) {
                    throw new IllegalStateException("Found malformed NZP file header: File is not a npz file? " + file.getPath());
                } else {
                    break;
                }
            }
            int fNameLength = localHeader[26];
            byte[] fNameBytes = new byte[fNameLength];
            is.read(fNameBytes);
            String fName = "";
            for (int i=0; i < fNameLength - 4; i++){
                fName += (char)fNameBytes[i];
            }
            int extraFieldLength = localHeader[28];
            if (extraFieldLength > 0){
                is.read(new byte[extraFieldLength]);
            }
            is.read(new byte[11]);
            
            String headerStr = "";
            int b;
            while((b = is.read()) != ((int)'\n')){
                headerStr += (char)b;
            }

            int idx;
            String typeStr;
            if(headerStr.contains("<")){
                idx = headerStr.indexOf("'<") + 2;
            } else {
                idx = headerStr.indexOf("'|") + 2;
            }
            typeStr = headerStr.substring(idx, idx + 2);

            int elemSize;
            DataType dt;
            if (typeStr.equals("f8")){
                elemSize = 8;
                dt = DataType.DOUBLE;
            } else if (typeStr.equals("f4")){
                elemSize = 4;
                dt = DataType.FLOAT;
            } else if(typeStr.equals("f2")){
                elemSize = 2;
                dt = DataType.HALF;
            } else if(typeStr.equals("i8")){
                elemSize = 8;
                dt = DataType.LONG;
            } else if (typeStr.equals("i4")){
                elemSize = 4;
                dt = DataType.INT;
            } else if(typeStr.equals("i2")){
                elemSize = 2;
                dt = DataType.SHORT;
            } else if(typeStr.equals("i1")){
                elemSize = 1;
                dt = DataType.BYTE;
            } else if(typeStr.equals("u1")){
                elemSize = 1;
                dt = DataType.UBYTE;
            } else{
                throw new Exception("Unsupported data type: " + typeStr);
            }
            idx = headerStr.indexOf("'fortran_order': ");
            char order = (headerStr.charAt(idx + "'fortran_order': ".length()) == 'F')? 'c' : 'f';

            String shapeStr = headerStr.substring(headerStr.indexOf("(") + 1, headerStr.indexOf(")"));

            shapeStr = shapeStr.replace(" ", "");
            String[] dims = shapeStr.split(",");
            long[] shape = new long[dims.length];
            long size = 1;
            for (int i =0; i < dims.length; i++){
                long d = Long.parseLong(dims[i]);
                shape[i] = d;
                size *= d;
            }


            // TODO support long shape

            int numBytes = (int)(size * elemSize);
            byte[] data = new byte[numBytes];
            is.read(data);
            ByteBuffer bb = ByteBuffer.wrap(data);

            if (dt == DataType.DOUBLE){
                double[] doubleData = new double[(int)size];
                for (int i=0; i<size; i++){
                    long l = bb.getLong(8*i);
                    l = Long.reverseBytes(l);
                    doubleData[i] = Double.longBitsToDouble(l);
                }
                map.put(fName, Nd4j.create(doubleData, shape, order));
            } else if(dt == DataType.FLOAT){
                float[] floatData = new float[(int)size];
                for (int i=0; i<size; i++){
                    int i2 = bb.getInt(4*i);
                    i2 = Integer.reverseBytes(i2);
                    float f = Float.intBitsToFloat(i2);
                    floatData[i] = f;
                }
                map.put(fName, Nd4j.create(floatData, shape, order));
            } else if(dt == DataType.HALF){
                INDArray arr = Nd4j.create(DataType.HALF, size);
                ByteBuffer bb2 = arr.data().pointer().asByteBuffer();
                for( int i=0; i<size; i++ ) {
                    short s = bb.getShort(2*i);
                    bb2.put((byte)((s >> 8) & 0xff));
                    bb2.put((byte)(s & 0xff));
                }
                Nd4j.getAffinityManager().tagLocation(arr, AffinityManager.Location.HOST);
                map.put(fName, arr.reshape(order, shape));
            } else if(dt == DataType.LONG){
                long[] d = new long[(int)size];
                for (int i=0; i<size; i++){
                    long l = bb.getLong(8*i);
                    l = Long.reverseBytes(l);
                    d[i] = l;
                }
                map.put(fName, Nd4j.createFromArray(d).reshape(order, shape));
            } else if(dt == DataType.INT){
                int[] d = new int[(int)size];
                for (int i=0; i<size; i++){
                    int l = bb.getInt(4*i);
                    l = Integer.reverseBytes(l);
                    d[i] = l;
                }
                map.put(fName, Nd4j.createFromArray(d).reshape(order, shape));
            } else if(dt == DataType.SHORT){
                short[] d = new short[(int)size];
                for (int i=0; i<size; i++){
                    short l = bb.getShort(2*i);
                    l = Short.reverseBytes(l);
                    d[i] = l;
                }
                map.put(fName, Nd4j.createFromArray(d).reshape(order, shape));
            } else if(dt == DataType.BYTE){
                map.put(fName, Nd4j.createFromArray(data).reshape(order, shape));
            } else if(dt == DataType.UBYTE){
                short[] d = new short[(int)size];
                for (int i=0; i<size; i++){
                    short l = ((short) (bb.get(i) & (short) 0xff));
                    d[i] = l;
                }
                map.put(fName, Nd4j.createFromArray(d).reshape(order, shape).castTo(DataType.UBYTE));
            }

        }

        return map;

    }
    public Map<String, INDArray> _createFromNpzFile(File file) throws Exception{

        // TODO: Fix libnd4j implementation
        byte[] pathBytes = file.getAbsolutePath().getBytes(Charset.forName("UTF-8"));
        ByteBuffer directBuffer = ByteBuffer.allocateDirect(pathBytes.length).order(ByteOrder.nativeOrder());
        directBuffer.put(pathBytes);
        directBuffer.rewind();
        directBuffer.position(0);
        Pointer pointer = nativeOps.mapFromNpzFile(new BytePointer(directBuffer));
        int n = nativeOps.getNumNpyArraysInMap(pointer);
        HashMap<String, INDArray> map = new HashMap<>();

        for (int i=0; i<n; i++){
            String arrName = nativeOps.getNpyArrayNameFromMap(pointer, i);
            Pointer arrPtr = nativeOps.getNpyArrayFromMap(pointer, i);
            int ndim = nativeOps.getNpyArrayRank(arrPtr);
            long[] shape = new long[ndim];
            LongPointer shapePtr = nativeOps.getNpyArrayShape(arrPtr);

            long length = 1;
            for (int j=0; j<ndim; j++){
                shape[j] = shapePtr.get(j);
                length *= shape[j];
            }

            int numBytes = nativeOps.getNpyArrayElemSize(arrPtr);

            int elemSize = numBytes * 8;

            char order = nativeOps.getNpyArrayOrder(arrPtr);

            Pointer dataPointer = nativeOps.dataPointForNumpyStruct(arrPtr);


            dataPointer.position(0);

            long size = elemSize * length;
            dataPointer.limit(size);
            dataPointer.capacity(size);

            INDArray arr;
            if (elemSize == Float.SIZE){
                FloatPointer dPointer = new FloatPointer(dataPointer.limit() / elemSize);
                DataBuffer data = Nd4j.createBuffer(dPointer,
                        DataType.FLOAT,
                        length,
                        FloatIndexer.create(dPointer));

                arr = Nd4j.create(data, shape, Nd4j.getStrides(shape, order), 0, order, DataType.FLOAT);

            }
            else if (elemSize == Double.SIZE){
                DoublePointer dPointer = new DoublePointer(dataPointer.limit() / elemSize);
                DataBuffer data = Nd4j.createBuffer(dPointer,
                        DataType.DOUBLE,
                        length,
                       DoubleIndexer.create(dPointer));
                arr = Nd4j.create(data, shape, Nd4j.getStrides(shape, order), 0, order, DataType.DOUBLE);
            }

            else{
                throw new Exception("Unsupported data type: " + String.valueOf(elemSize));
            }


            map.put(arrName, arr);
        }

        return map;

    }

}
