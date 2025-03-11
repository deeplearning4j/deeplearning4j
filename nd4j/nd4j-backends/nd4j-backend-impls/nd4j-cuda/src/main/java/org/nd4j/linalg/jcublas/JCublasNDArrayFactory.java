/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.jcublas;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.DataTypeEx;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.ops.custom.Flatten;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.linalg.compression.CompressionUtils;
import org.nd4j.linalg.jcublas.buffer.*;
import org.bytedeco.javacpp.*;
import org.nd4j.jita.allocator.enums.CudaConstants;
import org.nd4j.jita.allocator.impl.AllocationPoint;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.blas.*;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.nativeblas.*;

import java.util.*;

/**
 * Jcublas ndarray factory. Handles creation of
 * jcuda.jcublas ndarrays.
 *
 * @author mjk
 */
@Slf4j
public class JCublasNDArrayFactory extends BaseNativeNDArrayFactory {


    public JCublasNDArrayFactory() { }

    public JCublasNDArrayFactory(DataType dtype, Character order) {
        super(dtype, order);
    }

    public JCublasNDArrayFactory(DataType dtype, char order) {
        super(dtype, order);
        AtomicAllocator.getInstance();
    }

    @Override
    public void createBlas() {
        blas = new CudaBlas();
        PointerPointer functions = new PointerPointer(13);
        functions.put(0, Loader.addressof("cublasSgemv_v2"));
        functions.put(1, Loader.addressof("cublasDgemv_v2"));
        functions.put(2, Loader.addressof("cublasHgemm"));
        functions.put(3, Loader.addressof("cublasSgemm_v2"));
        functions.put(4, Loader.addressof("cublasDgemm_v2"));
        functions.put(5, Loader.addressof("cublasSgemmEx"));
        functions.put(6, Loader.addressof("cublasHgemmBatched"));
        functions.put(7, Loader.addressof("cublasSgemmBatched"));
        functions.put(8, Loader.addressof("cublasDgemmBatched"));
        functions.put(9, Loader.addressof("cusolverDnSgesvd_bufferSize"));
        functions.put(10, Loader.addressof("cusolverDnDgesvd_bufferSize"));
        functions.put(11, Loader.addressof("cusolverDnSgesvd"));
        functions.put(12, Loader.addressof("cusolverDnDgesvd"));
        nativeOps.initializeFunctions(functions);

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

    }

    @Override
    public void createLevel1() {
        level1 = new JcublasLevel1();
    }

    @Override
    public void createLevel2() {
        level2 = new JcublasLevel2();
    }

    @Override
    public void createLevel3() {
        level3 = new JcublasLevel3();
    }

    @Override
    public void createLapack() {
        lapack = new JcublasLapack();
    }

    @Override
    public INDArray create(int[] shape, DataBuffer buffer) {
        return new JCublasNDArray(shape, buffer);
    }


    @Override
    public INDArray create(DataBuffer buffer, LongShapeDescriptor longShapeDescriptor) {
        return new JCublasNDArray(buffer, longShapeDescriptor);
    }

    /**
     * Create an ndarray with the given data layout
     *
     * @param data the data to create the ndarray with
     * @return the ndarray with the given data layout
     */
    @Override
    public INDArray create(double[][] data) {
        return new JCublasNDArray(data);
    }

    @Override
    public INDArray create(double[][] data, char ordering) {
        return new JCublasNDArray(data, ordering);
    }

    @Override
    public INDArray create(DataBuffer data) {
        return new JCublasNDArray(data);
    }

    @Override
    public INDArray create(DataBuffer data, long rows, long columns, int[] stride, long offset) {
        // FIXME: int cast
        return new JCublasNDArray(data, new long[] {rows, columns}, ArrayUtil.toLongArray(stride), Nd4j.order(), data.dataType());
    }

    @Override
    public INDArray create(int[] shape, char ordering) {
        return new JCublasNDArray(shape, ordering);
    }

    @Override
    public INDArray createUninitialized(int[] shape, char ordering) {
        return new JCublasNDArray(shape, Nd4j.getStrides(shape, ordering), 0, ordering, false);
    }

    @Override
    public INDArray create(DataBuffer data, int[] newShape, int[] newStride, long offset, char ordering) {
        return new JCublasNDArray(data, newShape, newStride, offset, ordering);
    }

    @Override
    public INDArray create(float[] data, int[] shape, long offset, Character order) {
        return new JCublasNDArray(data, shape, offset, order);
    }

    @Override
    public INDArray create(float[] data, long rows, long columns, int[] stride, long offset, char ordering) {
        return new JCublasNDArray(data, new long[] {rows, columns}, ArrayUtil.toLongArray(stride), offset, ordering);
    }

    @Override
    public INDArray create(double[] data, int[] shape, char ordering) {
        return new JCublasNDArray(data, shape, ordering);
    }

    @Override
    public INDArray create(double[] data, long[] shape, char ordering) {
        return new JCublasNDArray(data, shape, ordering);
    }

    @Override
    public INDArray create(LongShapeDescriptor longShapeDescriptor) {
        return new JCublasNDArray(longShapeDescriptor);
    }

    @Override
    public INDArray create(Collection<String> strings, long[] shape, char order) {
        val pairShape = Nd4j.getShapeInfoProvider().createShapeInformation(shape, order, DataType.UTF8);
        val buffer = new CudaUtf8Buffer(strings);
        val list = new ArrayList<>(strings);
        return Nd4j.createArrayFromShapeBuffer(buffer, pairShape);
    }

    @Override
    public INDArray createUninitialized(DataType dataType, long[] shape, long[] strides, char ordering, MemoryWorkspace currentWorkspace) {
        return null;
    }

    @Override
    public INDArray create(List<INDArray> list, int[] shape, char ordering) {
        return new JCublasNDArray(list, shape, ordering);
    }

    @Override
    public INDArray create(double[] data, int[] shape, long offset) {
        return new JCublasNDArray(data, shape, (char) offset);
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, long offset, char ordering) {
        return new JCublasNDArray(data, shape, stride, offset, ordering);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param data
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, long offset) {
        return new JCublasNDArray(data, shape, stride, offset);
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param data
     * @param shape  the shape of the ndarray
     * @param stride the stride for the ndarray
     * @param offset the offset of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, long offset) {
        return new JCublasNDArray(data, shape, stride, offset);
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape) {
        return new JCublasNDArray(data, shape);
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape, int[] stride, long offset) {
        return new JCublasNDArray(data, ArrayUtil.toLongArray(shape), ArrayUtil.toLongArray(stride), Nd4j.order(), data.dataType());
    }

    /**
     * Creates an ndarray with the specified shape
     *
     * @param list
     * @param shape the shape of the ndarray
     * @return the instance
     */
    @Override
    public INDArray create(List<INDArray> list, int[] shape) {
        if (order == FORTRAN)
            return new JCublasNDArray(list, shape, ArrayUtil.calcStridesFortran(shape));
        else
            return new JCublasNDArray(list, shape);
    }

    @Override
    public INDArray create(float[] data, int[] shape, long offset) {
        return new JCublasNDArray(data, shape, offset);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType, workspace), shape, stride, order, dataType);
    }

    @Override
    public INDArray create(long[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride, order, dataType);
    }

    @Override
    public INDArray create(int[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride, order, dataType);
    }

    @Override
    public INDArray create(short[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride, order, dataType);
    }

    @Override
    public INDArray create(byte[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride, order, dataType);
    }

    @Override
    public INDArray create(boolean[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride, order, dataType);
    }

    @Override
    public INDArray create(float[][] floats) {
        return new JCublasNDArray(floats);
    }

    @Override
    public INDArray create(float[][] data, char ordering) {
        return new JCublasNDArray(data, ordering);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, long offset, char ordering) {
        return new JCublasNDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(DataBuffer buffer, int[] shape, long offset) {
        return new JCublasNDArray(buffer, shape, offset);
    }


    @Override
    public INDArray toFlattened(Collection<INDArray> matrices) {
        return this.toFlattened(order(), matrices);
    }

    @Override
    public INDArray toFlattened(char order, Collection<INDArray> matrices) {
        if (Nd4j.getExecutioner() instanceof GridExecutioner)
            ((GridExecutioner) Nd4j.getExecutioner()).flushQueue();

        return Nd4j.exec(new Flatten(order, matrices.toArray(new INDArray[0])))[0];
    }

    @Override
    public INDArray concat(int dimension, INDArray... toConcat) {
        Nd4j.getExecutioner().push();

        return Nd4j.exec(new Concat(dimension, toConcat))[0];
    }

    @Override
    public INDArray specialConcat(int dimension, INDArray... toConcat) {
        return concat(dimension,toConcat );
    }


    /**
     * This method produces concatenated array, that consist from tensors, fetched from source array, against some dimension and specified indexes
     *
     * @param source          source tensor
     * @param sourceDimension dimension of source tensor
     * @param indexes         indexes from source array
     * @return
     */
    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, int[] indexes) {
        return pullRows(source, sourceDimension, indexes, Nd4j.order());
    }

    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, long[] indexes) {
        return pullRows(source, sourceDimension, ArrayUtil.toInts(indexes));
    }

    /**
     * This method produces concatenated array, that consist from tensors, fetched from source array, against some dimension and specified indexes
     *
     * @param source          source tensor
     * @param sourceDimension dimension of source tensor
     * @param indexes         indexes from source array
     * @return
     */
    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, int[] indexes, char order) {
        if (indexes == null || indexes.length < 1)
            throw new IllegalStateException("Indexes can't be null or zero-length");


        long[] shape;
        if (source.rank() == 1) {
            shape = new long[]{indexes.length};
        } else if (sourceDimension == 1)
            shape = new long[] {indexes.length, source.shape()[sourceDimension]};
        else if (sourceDimension == 0)
            shape = new long[] {source.shape()[sourceDimension], indexes.length};
        else
            throw new UnsupportedOperationException("2D input is expected");

        return pullRows(source, Nd4j.createUninitialized(source.dataType(), shape, order), sourceDimension, indexes);
    }

    @Override
    public INDArray pullRows(INDArray source, INDArray destination, int sourceDimension, int[] indexes) {
        Nd4j.getExecutioner().push();

        if (indexes == null || indexes.length < 1)
            throw new IllegalStateException("Indexes can't be null or zero-length");

        Preconditions.checkArgument(source.dataType() == destination.dataType(), "Source and Destination data types must be the same");

        long[] shape = null;
        if (source.rank() == 1) {
            shape = new long[]{indexes.length};
        } else if (sourceDimension == 1)
            shape = new long[] {indexes.length, source.shape()[sourceDimension]};
        else if (sourceDimension == 0)
            shape = new long[] {source.shape()[sourceDimension], indexes.length};
        else
            throw new UnsupportedOperationException("2D input is expected");

        INDArray ret = destination;
        if(ret == null){
            ret = Nd4j.createUninitialized(source.dataType(), shape, order);
        } else {
            if(!Arrays.equals(shape, destination.shape())){
                throw new IllegalStateException("Cannot pull rows into destination array: expected destination array of" +
                        " shape " + Arrays.toString(shape) + " but got destination array of shape " + Arrays.toString(destination.shape()));
            }
        }

        AtomicAllocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(ret, source);

        OpaqueNDArray sourceOpaque = OpaqueNDArray.fromINDArray(source);
        OpaqueNDArray retOpaque = OpaqueNDArray.fromINDArray(ret);

        val tempIndexes = new CudaLongDataBuffer(indexes.length);
        AtomicAllocator.getInstance().memcpyBlocking(tempIndexes, new LongPointer(ArrayUtil.toLongArray(indexes)), indexes.length * 8, 0);

        OpaqueNDArray indexOpaque = OpaqueNDArray.fromINDArray(Nd4j.createFromArray(indexes));

        PointerPointer extras = new PointerPointer(null, // not used
                context.getOldStream(), allocator.getDeviceIdPointer());

        nativeOps.pullRows(extras,
                sourceOpaque, retOpaque,
                indexes.length,
                indexOpaque,
                sourceDimension);

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        allocator.registerAction(context, ret, source);

        return ret;
    }


    /**
     * In place shuffle of an ndarray
     * along a specified set of dimensions
     *
     * @param array     the ndarray to shuffle
     * @param dimension the dimension to do the shuffle
     * @return
     */
    @Override
    public void shuffle(INDArray array, Random rnd, long... dimension) {
        shuffle(Collections.singletonList(array), rnd, dimension);
    }

    /**
     * Symmetric in place shuffle of an ndarray
     * along a specified set of dimensions. Each array in list should have it's own dimension at the same index of dimensions array
     *
     * @param arrays      the ndarrays to shuffle
     * @param dimensions the dimensions to do the shuffle
     * @return
     */
    @Override
    public void shuffle(List<INDArray> arrays, Random rnd, List<long[]> dimensions) {
        // no dimension - no shuffle
        if (dimensions == null || dimensions.size() == 0)
            throw new RuntimeException("Dimension can't be null or 0-length");

        if (arrays == null || arrays.size() == 0)
            throw new RuntimeException("No input arrays provided");

        if (dimensions.size() > 1 && arrays.size() != dimensions.size())
            throw new IllegalStateException("Number of dimensions do not match number of arrays to shuffle");

        Nd4j.getExecutioner().push();

        // first we build TAD for input array and dimensions

        AtomicAllocator allocator = AtomicAllocator.getInstance();

        CudaContext context = null;

        for (int x = 0; x < arrays.size(); x++) {
            context = allocator.getFlowController().prepareAction(arrays.get(x));
        }

        val zero = arrays.get(0);
        int tadLength = 1;
        if (zero.rank() > 1)
            for (int i = 0; i < dimensions.get(0).length; i++) {
                tadLength *= zero.size(dimensions.get(0)[i]);
            }

        val numTads = zero.length() / tadLength;

        val map = ArrayUtil.buildInterleavedVector(rnd, (int) numTads);


        // Create a long[][] array for dimensions
        long[][] dimArray = new long[dimensions.size()][];
        for (int i = 0; i < dimensions.size(); i++) {
            dimArray[i] = dimensions.get(i);
        }

        // Create an INDArray from the long[][] array
        INDArray dimINDArray = Nd4j.createFromArray(dimArray);

        // Convert the INDArray to OpaqueNDArray
        OpaqueNDArray dimensionArr = OpaqueNDArray.fromINDArray(dimINDArray);

        val extras = new PointerPointer(null, // not used
                context.getOldStream(), allocator.getDeviceIdPointer());

        // Create an array of OpaqueNDArray
        OpaqueNDArray[] xOpaqueArray = new OpaqueNDArray[arrays.size()];
        for (int i = 0; i < arrays.size(); i++) {
            val array = arrays.get(i);

            //we have to sync manually here as we are calling the method with raw cuda pointers
            AllocationPoint point = allocator.getAllocationPoint(array);
            if(point.isActualOnHostSide()) {
                AtomicAllocator.getInstance().getFlowController().synchronizeToDevice(point);
                point.tickDeviceWrite();
            }

            xOpaqueArray[i] = OpaqueNDArray.fromINDArray(array);
        }

        // Create OpaqueNDArrayArr from the array of OpaqueNDArray
        OpaqueNDArrayArr xArr = new OpaqueNDArrayArr(xOpaqueArray);

        nativeOps.shuffle(extras, xArr, null, arrays.size(), dimensionArr, OpaqueNDArray.fromINDArray(Nd4j.createFromArray(map)));

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        for (int f = 0; f < arrays.size(); f++) {
            allocator.getFlowController().registerAction(context, arrays.get(f));
        }
    }

    /**
     * Symmetric in place shuffle of an ndarray
     * along a specified set of dimensions. All arrays
     *
     * @param sourceArrays     the ndarray to shuffle
     * @param dimension the dimension to do the shuffle
     * @return
     */
    @Override
    public void shuffle(Collection<INDArray> sourceArrays, Random rnd, long... dimension) {
        shuffle(new ArrayList<INDArray>(sourceArrays), rnd, Collections.singletonList(dimension));
    }

    /*
    public DataBuffer convertToHalfs(DataBuffer buffer) {
        DataBuffer halfsBuffer = new CudaHalfDataBuffer(buffer.length());
    
        AtomicAllocator allocator = AtomicAllocator.getInstance();
    
        AllocationPoint pointSrc = allocator.getAllocationPoint(buffer);
        AllocationPoint pointDst = allocator.getAllocationPoint(halfsBuffer);
    
        CudaContext context =  allocator.getFlowController().prepareAction(pointDst, pointSrc);
    
        PointerPointer extras = new PointerPointer(
                null, // not used for conversion
                context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer());
    
        Pointer x = AtomicAllocator.getInstance().getPointer(buffer, context);
        Pointer z = AtomicAllocator.getInstance().getPointer(halfsBuffer, context);
    
        if (buffer.dataType() == DataType.FLOAT) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().convertFloatsToHalfs(extras, x, (int) buffer.length(), z);
            pointDst.tickDeviceWrite();
        } else if (buffer.dataType() == DataType.DOUBLE) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().convertDoublesToHalfs(extras, x, (int) buffer.length(), z);
            pointDst.tickDeviceWrite();
        } else if (buffer.dataType() == DataType.HALF) {
            log.info("Buffer is already HALF-precision");
            return buffer;
        } else {
            throw new UnsupportedOperationException("Conversion INT->HALF isn't supported yet.");
        }
    
        allocator.getFlowController().registerAction(context, pointDst, pointSrc);
    
        return halfsBuffer;
    }
    
    public DataBuffer restoreFromHalfs(DataBuffer buffer) {
        if (buffer.dataType() != DataType.HALF)
            throw new IllegalStateException("Input DataBuffer should contain Halfs");
    
        DataBuffer outputBuffer = null;
    
    
    
        if (Nd4j.dataType() == DataType.FLOAT) {
            outputBuffer = new CudaFloatDataBuffer(buffer.length());
    
        } else if (Nd4j.dataType() == DataType.DOUBLE) {
            outputBuffer = new CudaDoubleDataBuffer(buffer.length());
    
        } else throw new UnsupportedOperationException("DataType ["+Nd4j.dataType()+"] isn't supported yet");
    
        AtomicAllocator allocator = AtomicAllocator.getInstance();
    
        AllocationPoint pointSrc = allocator.getAllocationPoint(buffer);
        AllocationPoint pointDst = allocator.getAllocationPoint(outputBuffer);
    
        CudaContext context =  allocator.getFlowController().prepareAction(pointDst, pointSrc);
    
        PointerPointer extras = new PointerPointer(
                null, // not used for conversion
                context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer());
    
        Pointer x = AtomicAllocator.getInstance().getPointer(buffer, context);
        Pointer z = AtomicAllocator.getInstance().getPointer(outputBuffer, context);
    
        if (Nd4j.dataType() == DataType.FLOAT) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().convertHalfsToFloats(extras, x, (int) buffer.length(), z);
            pointDst.tickDeviceWrite();
        } else if (Nd4j.dataType() == DataType.DOUBLE) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().convertHalfsToDoubles(extras, x, (int) buffer.length(), z);
            pointDst.tickDeviceWrite();
        } else if (Nd4j.dataType() == DataType.HALF) {
            log.info("Buffer is already HALF-precision");
            return buffer;
        }
    
        allocator.getFlowController().registerAction(context, pointDst, pointSrc);
    
        return outputBuffer;
    }
    */

    /**
     * This method converts Single/Double precision databuffer to Half-precision databuffer
     *
     * @param typeSrc
     * @param source
     * @param typeDst @return
     */
    @Override
    public INDArray convertDataEx(DataTypeEx typeSrc, INDArray source, DataTypeEx typeDst) {
        if (source.isView())
            throw new UnsupportedOperationException("Impossible to compress View. Consider using dup() before. ");

        DataBuffer buffer = convertDataEx(typeSrc, source.data(), typeDst);
        source.setData(buffer);

        if (buffer instanceof CompressedDataBuffer)
            source.markAsCompressed(true);
        else
            source.markAsCompressed(false);

        return source;
    }



    @Override
    public void convertDataEx(DataTypeEx typeSrc, Pointer source, DataTypeEx typeDst, Pointer target, long length) {
        val stream = AtomicAllocator.getInstance().getDeviceContext().getOldStream();

        val p = new PointerPointer<>(new Pointer[]{null, stream});

        nativeOps.convertTypes(p, typeSrc.ordinal(), source, length, typeDst.ordinal(), target);

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());
    }

    @Override
    public void convertDataEx(DataTypeEx typeSrc, Pointer source, DataTypeEx typeDst, DataBuffer buffer) {
        Pointer srcPtr = null;
        Pointer dstPtr = null;
        long size = 0;
        long ssize = 0;
        val stream = AtomicAllocator.getInstance().getDeviceContext().getOldStream();
        if (buffer instanceof CompressedDataBuffer) {
            // compressing
            size = ((CompressedDataBuffer) buffer).getCompressionDescriptor().getCompressedLength();
            ssize = ((CompressedDataBuffer) buffer).getCompressionDescriptor().getOriginalLength();

            srcPtr = nativeOps.mallocDevice(ssize, 0, 0);
            dstPtr = nativeOps.mallocDevice(size, 0, 0);

            if (nativeOps.lastErrorCode() != 0)
                throw new RuntimeException(nativeOps.lastErrorMessage());

            nativeOps.memcpyAsync(srcPtr, source, ssize, CudaConstants.cudaMemcpyHostToDevice, stream);

            if (nativeOps.lastErrorCode() != 0)
                throw new RuntimeException(nativeOps.lastErrorMessage());
        } else {
            // decompressing
            throw new UnsupportedOperationException();
        }

        convertDataEx(typeSrc, srcPtr, typeDst, dstPtr, buffer.length());
        nativeOps.memcpyAsync(buffer.addressPointer(), dstPtr, size, CudaConstants.cudaMemcpyHostToHost, stream);

        stream.synchronize();

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        if (buffer instanceof CompressedDataBuffer) {
            nativeOps.freeDevice(srcPtr, 0);
            nativeOps.freeDevice(dstPtr, 0);

            if (nativeOps.lastErrorCode() != 0)
                throw new RuntimeException(nativeOps.lastErrorMessage());
        }
    }

    @Override
    public void convertDataEx(DataTypeEx typeSrc, DataBuffer source, DataTypeEx typeDst, DataBuffer target) {

        val stream = AtomicAllocator.getInstance().getDeviceContext().getOldStream();
        Pointer srcPtr = null;
        Pointer dstPtr = null;

        // we have to replace pointer here, temporary
        if (Nd4j.getWorkspaceManager().anyWorkspaceActiveForCurrentThread()) {
            val ws = Nd4j.getMemoryManager().getCurrentWorkspace();
            // if true - we're decompressing from host memory
            if (source instanceof CompressedDataBuffer) {
                val size = ((CompressedDataBuffer) source).getCompressionDescriptor().getCompressedLength();
                srcPtr = ws.alloc(size, MemoryKind.DEVICE, DataType.HALF, false);
                nativeOps.memcpyAsync(srcPtr, source.addressPointer(), size, CudaConstants.cudaMemcpyHostToHost, stream);

                if (nativeOps.lastErrorCode() != 0)
                    throw new RuntimeException(nativeOps.lastErrorMessage());
            }

            // if true - we're compressing into host memory
            if (target instanceof CompressedDataBuffer) {
                val size = ((CompressedDataBuffer) target).getCompressionDescriptor().getCompressedLength();
                dstPtr = ws.alloc(size, MemoryKind.DEVICE, DataType.HALF, false);
            }
        } else {
            // if true - we're decompressing from host memory
            if (source instanceof CompressedDataBuffer) {
                log.info("Replacing source ptr");
                val size = ((CompressedDataBuffer) source).getCompressionDescriptor().getCompressedLength();
                srcPtr = nativeOps.mallocDevice(size, 0, 0);
                nativeOps.memcpyAsync(srcPtr, source.addressPointer(), size, CudaConstants.cudaMemcpyHostToHost, stream);
                stream.synchronize();

                if (nativeOps.lastErrorCode() != 0)
                    throw new RuntimeException(nativeOps.lastErrorMessage());
            } else
                srcPtr = AtomicAllocator.getInstance().getPointer(source);

            // if true - we're compressing into host memory
            if (target instanceof CompressedDataBuffer) {
                log.info("Replacing target ptr");
                val size = ((CompressedDataBuffer) target).getCompressionDescriptor().getCompressedLength();
                dstPtr = nativeOps.mallocDevice(size, 0, 0);

                if (nativeOps.lastErrorCode() != 0)
                    throw new RuntimeException(nativeOps.lastErrorMessage());
            } else
                dstPtr = AtomicAllocator.getInstance().getPointer(target);
        }


        convertDataEx(typeSrc, srcPtr, typeDst, dstPtr, target.length());

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        Nd4j.getExecutioner().commit();


        // we were compressing something into temporary buffer
        if (target instanceof CompressedDataBuffer) {
            nativeOps.memcpyAsync(target.addressPointer(), dstPtr, target.capacity(),  CudaConstants.cudaMemcpyHostToHost, stream);

            if (Nd4j.getWorkspaceManager().anyWorkspaceActiveForCurrentThread()) {
                // no-op, workspace was used
            } else
                nativeOps.freeDevice(dstPtr, 0);
        }

        // we were decompressing something from host memory
        if (source instanceof CompressedDataBuffer) {
            if (Nd4j.getWorkspaceManager().anyWorkspaceActiveForCurrentThread()) {
                // no-op, workspace was used
            } else
                nativeOps.freeDevice(srcPtr, 0);

        }

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        Nd4j.getExecutioner().commit();
    }

    @Override
    public DataBuffer convertDataEx(DataTypeEx typeSrc, DataBuffer source, DataTypeEx typeDst) {
        int elementSize = 0;
        if (typeDst.ordinal() <= 2)
            elementSize = 1;
        else if (typeDst.ordinal() <= 5)
            elementSize = 2;
        else if (typeDst.ordinal() == 6)
            elementSize = 4;
        else if (typeDst.ordinal() == 7)
            elementSize = 8;
        else
            throw new UnsupportedOperationException("Unknown target TypeEx: " + typeDst.name());

        // flushQueue should be blocking here, because typeConversion happens on cpu side
        Nd4j.getExecutioner().commit();

        DataBuffer buffer = null;

        if (!(source instanceof CompressedDataBuffer))
            AtomicAllocator.getInstance().synchronizeHostData(source);

        if (CompressionUtils.goingToCompress(typeSrc, typeDst)) {
            // all types below 8 are compression modes
            Pointer pointer = new BytePointer(source.length() * elementSize);
            CompressionDescriptor descriptor = new CompressionDescriptor(source, typeDst.name());
            descriptor.setCompressionType(CompressionType.LOSSY);
            descriptor.setCompressedLength(source.length() * elementSize);
            buffer = new CompressedDataBuffer(pointer, descriptor);
        } else {
            CompressedDataBuffer compressed = (CompressedDataBuffer) source;
            CompressionDescriptor descriptor = compressed.getCompressionDescriptor();
            // decompression mode
            buffer = Nd4j.createBuffer(descriptor.getNumberOfElements(), false);

            AllocationPoint point = AtomicAllocator.getInstance().getAllocationPoint(buffer);
            point.tickDeviceWrite();
        }

        convertDataEx(typeSrc, source, typeDst, buffer);

        return buffer;
    }


    @Override
    public INDArray sort(INDArray x, boolean descending) {
        if (x.isScalar())
            return x;

        Nd4j.getExecutioner().push();

        CudaContext context = AtomicAllocator.getInstance().getFlowController().prepareAction(x);

        Pointer ptr = AtomicAllocator.getInstance().getHostPointer(x.shapeInfoDataBuffer());

        PointerPointer extraz = new PointerPointer(ptr, // 0
                context.getOldStream(), // 1
                AtomicAllocator.getInstance().getDeviceIdPointer(), // 2
                null, // 3
                context.getBufferReduction(), // 4
                context.getBufferScalar(), // 5
                null, // 6
                ptr, // 7
                AtomicAllocator.getInstance().getHostPointer(x.shapeInfoDataBuffer()), // 8
                ptr, // 9
                ptr, // 10
                ptr, // 11
                ptr, // 12
                ptr, // 13
                ptr, // 14
                ptr, // special pointer for IsMax  // 15
                ptr, // special pointer for IsMax  // 16
                ptr, // special pointer for IsMax // 17
                new CudaPointer(0));

        // we're sending > 10m elements to radixSort
        boolean isRadix = !x.isView() && (x.length() > 1024 * 1024 * 10);
        INDArray tmpX = x;

        // we need to guarantee all threads are finished here
        if (isRadix)
            Nd4j.getExecutioner().commit();

        OpaqueNDArray x2 = OpaqueNDArray.fromINDArray(x);

        nativeOps.sort(extraz,
                x2,
                descending
        );

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        AtomicAllocator.getInstance().getFlowController().registerAction(context, x);

        return x;
    }

    @Override
    public INDArray empty(DataType type) {
        long extras  = ArrayOptionsHelper.setOptionBit(0L, ArrayType.EMPTY);
        extras = ArrayOptionsHelper.setOptionBit(extras, type);
        val shape = Nd4j.getShapeInfoProvider().createShapeInformation(new long[0], new long[0], 1, 'c', extras);
        return new JCublasNDArray(null, (CudaLongDataBuffer) shape.getFirst(), shape.getSecond());
    }


    @Override
    public INDArray sort(INDArray x, boolean descending, long... dimension) {
        if (x.isScalar())
            return x;

        Arrays.sort(dimension);

        Nd4j.getExecutioner().push();

        val tadBuffers = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x, dimension);

        val context = AtomicAllocator.getInstance().getFlowController().prepareAction(x);

        val extraz = new PointerPointer(AtomicAllocator.getInstance().getHostPointer(x.shapeInfoDataBuffer()), // not used
                context.getOldStream(), AtomicAllocator.getInstance().getDeviceIdPointer());


        val dimensionPointer = AtomicAllocator.getInstance()
                .getHostPointer(AtomicAllocator.getInstance().getConstantBuffer(dimension));

        OpaqueNDArray x2 = OpaqueNDArray.fromINDArray(x);
        nativeOps.sortTad(extraz,
                x2,
                new LongPointer(dimensionPointer),
                dimension.length,
                (LongPointer) AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context),
                new LongPointerWrapper(AtomicAllocator.getInstance().getPointer(tadBuffers.getSecond(), context)),
                descending
        );

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        AtomicAllocator.getInstance().getFlowController().registerAction(context, x);

        return x;
    }


    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, long offset) {
        return new JCublasNDArray(data, shape, stride, offset, Nd4j.order());
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, char order, DataType dataType) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride,  order, dataType);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, long offset) {
        return new JCublasNDArray(data, shape, stride, offset, Nd4j.order());
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType, workspace), shape, stride,  Nd4j.order(), dataType);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType, workspace), shape, stride,  Nd4j.order(), dataType);
    }

    @Override
    public INDArray create(long[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride,  Nd4j.order(), dataType);
    }

    @Override
    public INDArray create(int[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride,  Nd4j.order(), dataType);
    }

    @Override
    public INDArray create(short[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride,  Nd4j.order(), dataType);
    }

    @Override
    public INDArray create(byte[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride,  Nd4j.order(), dataType);
    }

    @Override
    public INDArray create(boolean[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride,  Nd4j.order(), dataType);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createTypedBuffer(data, dataType, workspace), shape, stride,  order, dataType);
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape) {
        return new JCublasNDArray(data, shape);
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride, long offset) {
        return new JCublasNDArray(data, shape, stride, offset, Nd4j.order(), data.dataType());
    }

    @Override
    public INDArray create(List<INDArray> list, long[] shape) {
        return new JCublasNDArray(list, shape);
    }

    @Override
    public INDArray create(long rows, long columns, long[] stride, long offset) {
        return create(new long[] {rows, columns}, stride, offset, Nd4j.order());
    }

    @Override
    public INDArray create(long[] shape, char ordering) {
        return new JCublasNDArray(shape, 0, ordering);
    }

    @Override
    public INDArray create(DataType dataType, long[] shape, char ordering, MemoryWorkspace workspace) {
        return create(dataType, shape, Nd4j.getStrides(shape, ordering), ordering, workspace);
    }

    @Override
    public INDArray create(DataType dataType, long[] shape, long[] strides, char ordering, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createBuffer(dataType, Shape.lengthOf(shape), true, workspace), shape, strides, ordering, dataType);
    }

    @Override
    public INDArray createUninitialized(long[] shape, char ordering) {
        return new JCublasNDArray(shape, Nd4j.getStrides(shape, ordering), 0, ordering, false);
    }

    @Override
    public INDArray createUninitialized(DataType dataType, long[] shape, char ordering, MemoryWorkspace workspace) {
        return new JCublasNDArray(Nd4j.createBuffer(dataType, Shape.lengthOf(shape), false), shape, Nd4j.getStrides(shape, ordering), ordering, dataType);
    }

    @Override
    public INDArray createUninitializedDetached(DataType dataType, char ordering, long... shape) {
        return new JCublasNDArray(Nd4j.createBufferDetached(shape, dataType), shape, Nd4j.getStrides(shape, order), order, dataType);
    }

    @Override
    public INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, char ordering) {
        return new JCublasNDArray(data, newShape, newStride, offset, ordering, data.dataType());
    }

    @Override
    public INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, long ews, char ordering) {
        return new JCublasNDArray(data, newShape, newStride, offset, ews, ordering, data.dataType());
    }

    @Override
    public INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, long ews, char ordering, boolean isView) {
        return new JCublasNDArray(data,newShape,newStride,offset,ews,ordering,data.dataType(),isView);
    }

    @Override
    public INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, char ordering, DataType dataType) {
        return new JCublasNDArray(data, newShape, newStride, offset, ordering, dataType);
    }

    @Override
    public INDArray create(List<INDArray> list, long[] shape, char ordering) {
        return new JCublasNDArray(list, shape, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, char order, long offset) {
        return new JCublasNDArray(data, shape, stride, offset, order);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, long offset, char ordering) {
        return new JCublasNDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, long offset, char ordering) {
        return new JCublasNDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long offset, Character order) {
        return new JCublasNDArray(data, shape, Nd4j.getStrides(shape, order), offset, order);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long offset, Character order) {
        return new JCublasNDArray(data, shape, Nd4j.getStrides(shape, order), offset, order);
    }

    @Override
    public INDArray create(float[] data, long[] shape, char ordering) {
        return new JCublasNDArray(data, shape, Nd4j.getStrides(shape, order), 0, ordering);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @Override
    public INDArray sortCooIndices(INDArray x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray create(DataType dataType, long[] shape, long[] paddings, long[] paddingOffsets, char ordering,
                           MemoryWorkspace workspace) {
        return new JCublasNDArray(dataType, shape, paddings, paddingOffsets, ordering, workspace);
    }
}
