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

package org.nd4j.linalg.cpu.nativecpu;


import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.LongBuffer;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.linalg.compression.CompressionUtils;
import org.nd4j.linalg.memory.MemcpyDirection;
import org.nd4j.linalg.primitives.Pair;
import org.bytedeco.javacpp.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.compression.CompressionUtils;
import org.nd4j.linalg.cpu.nativecpu.blas.*;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.BaseNativeNDArrayFactory;
import org.nd4j.nativeblas.LongPointerWrapper;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.*;

/**
 * {@link org.nd4j.linalg.factory.NDArrayFactory}
 * for cpus and the nd4j-native backend.
 *
 * @author Adam Gibson
 */
@Slf4j
public class CpuNDArrayFactory extends BaseNativeNDArrayFactory {

    protected ThreadLocal<PointerPointer> extrazA = new ThreadLocal<>();
    protected ThreadLocal<PointerPointer> extrazB = new ThreadLocal<>();
    protected ThreadLocal<Integer> extrazSize = new ThreadLocal<>();

    public CpuNDArrayFactory() {}

    static {
        //invoke the override
        Nd4j.getBlasWrapper();
    }


    public CpuNDArrayFactory(DataBuffer.Type dtype, Character order) {
        super(dtype, order);
    }

    public CpuNDArrayFactory(DataBuffer.Type dtype, char order) {
        super(dtype, order);
    }

    @Override
    public void createBlas() {
        String lib = System.getProperty(ND4JSystemProperties.ND4J_CPU_LOAD_OPENBLAS,
                     System.getProperty(ND4JSystemProperties.ND4J_CPU_LOAD_OPENBLAS_NOLAPACK, "")).toLowerCase();
        if (lib.trim().length() == 0) {
            // try to load by default the LAPACK-less version of MKL bundled with MKL-DNN
            System.setProperty(ND4JSystemProperties.ND4J_CPU_LOAD_OPENBLAS_NOLAPACK, "mklml");
        }

        blas = new CpuBlas();

        // TODO: add batched gemm here

        PointerPointer functions = new PointerPointer(10);
        functions.put(0, Loader.addressof("cblas_sgemv"));
        functions.put(1, Loader.addressof("cblas_dgemv"));
        functions.put(2, Loader.addressof("cblas_sgemm"));
        functions.put(3, Loader.addressof("cblas_dgemm"));
        functions.put(4, Loader.addressof("cblas_sgemm_batch"));
        functions.put(5, Loader.addressof("cblas_dgemm_batch"));
        functions.put(6, Loader.addressof("LAPACKE_sgesvd"));
        functions.put(7, Loader.addressof("LAPACKE_dgesvd"));
        functions.put(8, Loader.addressof("LAPACKE_sgesdd"));
        functions.put(9, Loader.addressof("LAPACKE_dgesdd"));
        nativeOps.initializeFunctions(functions);
    }

    @Override
    public void createLevel1() {
        level1 = new CpuLevel1();
    }

    @Override
    public void createLevel2() {
        level2 = new CpuLevel2();
    }

    @Override
    public void createLevel3() {
        level3 = new CpuLevel3();
    }

    @Override
    public void createLapack() {
        lapack = new CpuLapack();
    }

    @Override
    public INDArray create(int[] shape, DataBuffer buffer) {
        return new NDArray(shape, buffer);
    }

    /**
     * Create an ndarray with the given data layout
     *
     * @param data the data to create the ndarray with
     * @return the ndarray with the given data layout
     */
    @Override
    public INDArray create(double[][] data) {
        return new NDArray(data);
    }

    @Override
    public INDArray create(double[][] data, char ordering) {
        return new NDArray(data, ordering);
    }

    @Override
    public INDArray create(DataBuffer data) {
        return new NDArray(data);
    }

    @Override
    public INDArray create(DataBuffer data, long rows, long columns, int[] stride, long offset) {
        return create(data, new long[]{rows, columns}, ArrayUtil.toLongArray(stride), offset);
    }

    @Override
    public INDArray create(long rows, long columns, long[] stride, long offset) {
        return create(new long[]{rows, columns}, stride, offset);
    }

    @Override
    public INDArray create(int[] shape, char ordering) {
        return new NDArray(shape, Nd4j.getStrides(shape, ordering), 0, ordering);
    }

    @Override
    public INDArray create(long[] shape, char ordering) {
        return new NDArray(shape, Nd4j.getStrides(shape, ordering), 0, ordering);
    }

    @Override
    public INDArray createUninitialized(int[] shape, char ordering) {
        return new NDArray(shape, Nd4j.getStrides(shape, ordering), 0, ordering, false);
    }

    @Override
    public INDArray createUninitialized(long[] shape, char ordering) {
        return new NDArray(shape, Nd4j.getStrides(shape, ordering), 0, ordering, false);
    }

    @Override
    public INDArray createUninitializedDetached(int[] shape, char ordering) {
        MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        INDArray ret = new NDArray(shape, Nd4j.getStrides(shape, ordering), 0, ordering, false);
        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);
        return ret;
    }

    @Override
    public INDArray createUninitializedDetached(long[] shape, char ordering) {
        MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        INDArray ret = new NDArray(shape, Nd4j.getStrides(shape, ordering), 0, ordering, false);
        Nd4j.getMemoryManager().setCurrentWorkspace(workspace);
        return ret;
    }

    @Override
    public INDArray create(DataBuffer data, int[] newShape, int[] newStride, long offset, char ordering) {
        return new NDArray(data, newShape, newStride, offset, ordering);
    }

    @Override
    public INDArray create(float[] data, int[] shape, long offset, Character order) {
        return new NDArray(data, shape, offset, order);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long offset, Character order) {
        return new NDArray(data, shape, offset, order);
    }

    @Override
    public INDArray create(float[] data, long rows, long columns, int[] stride, long offset, char ordering) {
        return create(data, new long[]{rows, columns}, ArrayUtil.toLongArray(stride), offset, ordering);
    }

    @Override
    public INDArray create(double[] data, int[] shape, char ordering) {
        return new NDArray(Nd4j.createBuffer(data), shape, ordering);
    }

    @Override
    public INDArray create(double[] data, long[] shape, char ordering) {
        return create(data, shape, (Character) ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, char ordering) {
        return create(data, shape, (Character) ordering);
    }

    @Override
    public INDArray create(List<INDArray> list, int[] shape, char ordering) {
        return new NDArray(list, shape, ordering);
    }



    @Override
    public INDArray create(List<INDArray> list, long[] shape, char ordering) {
        return new NDArray(list, shape, ordering);
    }

    @Override
    public INDArray create(double[] data, int[] shape, long offset) {
        return new NDArray(Nd4j.createBuffer(data), shape, offset);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long offset, Character order) {
        return new NDArray(data, shape, offset, order.charValue());
    }



    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, long offset, char ordering) {
        return new NDArray(Nd4j.createBuffer(data), shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, long offset, char ordering) {
        return new NDArray(Nd4j.createBuffer(data), shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, long offset, char ordering) {
        return new NDArray(Nd4j.createBuffer(data), shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, long offset) {
        return new NDArray(data, shape, stride, offset, Nd4j.order());
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, long offset) {
        return new NDArray(data, shape, stride, offset, Nd4j.order());
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape) {
        return new NDArray(data, shape);
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride, long offset) {
        return create(data, shape, stride, offset, Nd4j.order());
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride, long offset, char ordering) {
        return new NDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, char order, long offset) {
        return new NDArray(data, shape, stride, offset, order);
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
        return new NDArray(data, shape, stride, offset);
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
        return new NDArray(data, shape, stride, offset);
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape) {
        return new NDArray(data, shape);
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape, int[] stride, long offset) {
        return new NDArray(data, shape, stride, offset, Nd4j.order());
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
        return new NDArray(list, shape, Nd4j.getStrides(shape));
    }

    @Override
    public INDArray create(List<INDArray> list, long[] shape) {
        return new NDArray(list, shape, Nd4j.getStrides(shape));
    }

    @Override
    public INDArray empty(DataBuffer.Type type) {
        long extras  = ArrayOptionsHelper.setOptionBit(0L, ArrayType.EMPTY);
        extras = ArrayOptionsHelper.setOptionBit(extras, type);
        val shape = Nd4j.getShapeInfoProvider().createShapeInformation(new int[0], new int[0],0,1,'c', extras);
        return new NDArray(null, (LongBuffer) shape.getFirst(), shape.getSecond());
    }

    @Override
    public INDArray create(float[][] floats) {
        return new NDArray(floats);
    }

    @Override
    public INDArray create(float[][] data, char ordering) {
        return new NDArray(data, ordering);
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, long offset, char ordering) {
        return new NDArray(data, shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(DataBuffer buffer, int[] shape, long offset) {
        return new NDArray(buffer, shape, Nd4j.getStrides(shape), offset);
    }

    @Override
    public INDArray create(float[] data, int[] shape, long offset) {
        return new NDArray(data, shape, offset);
    }

    @Override
    public INDArray toFlattened(char order, Collection<INDArray> matrices) {
        int length = 0;
        for (INDArray m : matrices)
            length += m.length();
        INDArray ret = Nd4j.create(new int[] {1, length}, order);
        int linearIndex = 0;
        PointerPointer dummy = new PointerPointer(new Pointer[] {null});
        for (INDArray m : matrices) {
            Nd4j.getCompressor().autoDecompress(m);

            if (m.ordering() == order && m.data().allocationMode() == DataBuffer.AllocationMode.HEAP
                    && Shape.strideDescendingCAscendingF(m) && Shape.isContiguousInBuffer(m)) {
                //Can do array copy
                int retFrom = linearIndex;
                long mFrom = m.offset();
                Object arr = m.data().array();
                if (arr instanceof float[]) {
                    float[] mData = (float[]) arr;
                    float[] retData = (float[]) ret.data().array();

                    // FIXME: LONG
                    // FIXME: int cast
                    System.arraycopy(mData, (int) mFrom, retData, retFrom, (int) m.length());
                } else {
                    double[] mData = (double[]) arr;
                    double[] retData = (double[]) ret.data().array();

                    // FIXME: LONG
                    // FIXME: int cast
                    System.arraycopy(mData, (int) mFrom, retData, retFrom, (int) m.length());
                }
                linearIndex += m.length();
            } else {

                    nativeOps.flatten(dummy, linearIndex, order, ret.data().addressPointer(),
                            (LongPointer) ret.shapeInfoDataBuffer().addressPointer(),
                            m.data().addressPointer(),
                            (LongPointer) m.shapeInfoDataBuffer().addressPointer());

                //Works for all cases...

                /* NdIndexIterator iter = new NdIndexIterator(order, m.shape());
                while (iter.hasNext()) {
                    ret.putScalar(linearIndex++, m.getDouble(iter.next()));
                }*/

                linearIndex += m.length();

            }
        }
        return ret;
    }

    @Override
    public INDArray[] tear(INDArray tensor, int... dimensions) {
        if (tensor.isCompressed())
            Nd4j.getCompressor().decompressi(tensor);

        Arrays.sort(dimensions);

        Pair<DataBuffer, DataBuffer> tadBuffers = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(tensor, dimensions);

        long tadLength = 1;
        long[] shape = new long[dimensions.length];
        for (int i = 0; i < dimensions.length; i++) {
            tadLength *= tensor.shape()[dimensions[i]];
            shape[i] = tensor.shape()[dimensions[i]];
        }



        int numTads = (int)(tensor.lengthLong() / tadLength);
        INDArray[] result = new INDArray[numTads];

        PointerPointer targets = new PointerPointer(numTads);

        for (int x = 0; x < numTads; x++) {
            result[x] = Nd4j.createUninitialized(shape);

            targets.put(x, result[x].data().pointer());
        }

            nativeOps.tear(null,
                    tensor.data().pointer(),
                    (LongPointer) tensor.shapeInfoDataBuffer().pointer(),
                    targets,
                    (LongPointer) result[0].shapeInfoDataBuffer().pointer(),
                    (LongPointer) tadBuffers.getFirst().pointer(),
                    new LongPointerWrapper(tadBuffers.getSecond().pointer())
            );

        return result;
    }

    /**
     * concatenate ndarrays along a dimension
     *
     * @param dimension the dimension to concatenate along
     * @param toConcat  the ndarrays to concatenate
     * @return the concatenate ndarrays
     */
    @Override
    public INDArray concat(int dimension, INDArray... toConcat) {
        if (toConcat == null || toConcat.length == 0)
            throw new ND4JIllegalStateException("Can't concatenate 0 arrays");

        if (toConcat.length == 1)
            return toConcat[0];

        // if reusable var wasn't created for this thread, or is smaller then needed - set it to new value
        if (extrazA.get() == null || extrazB.get() == null || extrazSize.get() == null || extrazSize.get() < toConcat.length) {
            extrazA.set(new PointerPointer(toConcat.length));
            extrazB.set(new PointerPointer(toConcat.length));
            extrazSize.set(toConcat.length);
        }

        PointerPointer shapeInfoPointers = extrazA.get();
        PointerPointer dataPointers = extrazB.get();

        int sumAlongDim = 0;

        long[] outputShape = ArrayUtil.copy(toConcat[0].shape());


        for (int i = 0; i < toConcat.length; i++) {
            if (toConcat[i].isCompressed())
                Nd4j.getCompressor().decompressi(toConcat[i]);

            shapeInfoPointers.put(i, toConcat[i].shapeInfoDataBuffer().addressPointer());
            dataPointers.put(i, toConcat[i].data().addressPointer());
            sumAlongDim += toConcat[i].size(dimension);
            for (int j = 0; j < toConcat[i].rank(); j++)
                if (j != dimension && toConcat[i].size(j) != outputShape[j]) {
                    throw new IllegalArgumentException(
                            "Illegal concatenation at array " + i + " and shape element " + j);
                }


            //log.info("Shape[{}]: {}", i, Arrays.toString(toConcat[i].shapeInfoDataBuffer().asInt()));
        }

        outputShape[dimension] = sumAlongDim;

        //PointerPointer dummy = new PointerPointer(new Pointer[] {null});

        INDArray ret = Nd4j.createUninitialized(outputShape, Nd4j.order());

        nativeOps.concat(null, dimension, toConcat.length, dataPointers, shapeInfoPointers,
                    ret.data().addressPointer(),
                    (LongPointer) ret.shapeInfoDataBuffer().addressPointer(),
                    //new PointerPointer(new Pointer[] {null}), new PointerPointer(new Pointer[] {null}));
                    null, null);

        return ret;
        // return super.concat(dimension,toConcat);
    }


    /**
     * For CPU backend this method is equal to concat()
     *
     * @param dimension the dimension to concatneate along
     * @param toConcat  the ndarrays to concateneate
     * @return
     */
    @Override
    public INDArray specialConcat(int dimension, INDArray... toConcat) {
        return concat(dimension, toConcat);
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
        return pullRows(source, sourceDimension, ArrayUtil.toLongArray(indexes));
    }

    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, long[] indexes) {
        return pullRows(source, sourceDimension, indexes, Nd4j.order());
    }

    /**
     * This method produces concatenated array, that consist from tensors, fetched from source array, against some dimension and specified indexes
     *
     * @param source          source tensor
     * @param sourceDimension dimension of source tensor
     * @param indexes         indexes from source array
     * @return
     */

    public INDArray pullRows(INDArray source, int sourceDimension, long[] indexes, char order) {
        if (indexes == null || indexes.length < 1)
            throw new IllegalStateException("Indexes can't be null or zero-length");

        long[] shape;
        if (sourceDimension == 1)
            shape = new long[] {indexes.length, source.shape()[sourceDimension]};
        else if (sourceDimension == 0)
            shape = new long[] {source.shape()[sourceDimension], indexes.length};
        else
            throw new UnsupportedOperationException("2D input is expected");
        return pullRows(source, Nd4j.createUninitialized(shape, order), sourceDimension, indexes);
    }

    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, int[] indexes, char order) {
        return pullRows(source, sourceDimension, ArrayUtil.toLongArray(indexes), order);
    }

    @Override
    public INDArray pullRows(INDArray source, INDArray destination, int sourceDimension, int[] indexes) {
        return pullRows(source, destination, sourceDimension, ArrayUtil.toLongArray(indexes));
    }

    public INDArray pullRows(INDArray source, INDArray destination, int sourceDimension, long[] indexes) {
        if (indexes == null || indexes.length < 1)
            throw new IllegalStateException("Indexes can't be null or zero-length");

        long[] shape = null;
        if (sourceDimension == 1)
            shape = new long[] {indexes.length, source.shape()[sourceDimension]};
        else if (sourceDimension == 0)
            shape = new long[] {source.shape()[sourceDimension], indexes.length};
        else
            throw new UnsupportedOperationException("2D input is expected");

        INDArray ret = destination;
        if(ret == null){
            ret = Nd4j.createUninitialized(shape, order);
        } else {
            if(!Arrays.equals(shape, destination.shape())){
                throw new IllegalStateException("Cannot pull rows into destination array: expected destination array of" +
                        " shape " + Arrays.toString(shape) + " but got destination array of shape " + Arrays.toString(destination.shape()));
            }
        }

        Nd4j.getCompressor().autoDecompress(source);

        PointerPointer dummy = new PointerPointer(new Pointer[] {null});

        TADManager tadManager = Nd4j.getExecutioner().getTADManager();

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(source, new int[] {sourceDimension});

        Pair<DataBuffer, DataBuffer> zTadBuffers = tadManager.getTADOnlyShapeInfo(ret, new int[] {sourceDimension});

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();

        Pointer zTadShapeInfo = zTadBuffers.getFirst().addressPointer();

        LongPointer pIndex = new LongPointer(indexes);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer hostTadOffsets = offsets == null ? null : offsets.addressPointer();

        DataBuffer zOffsets = zTadBuffers.getSecond();

        Pointer zTadOffsets = zOffsets == null ? null : zOffsets.addressPointer();


        nativeOps.pullRows(dummy, source.data().addressPointer(),
                    (LongPointer) source.shapeInfoDataBuffer().addressPointer(),
                    ret.data().addressPointer(),
                    (LongPointer) ret.shapeInfoDataBuffer().addressPointer(), indexes.length, pIndex,
                    (LongPointer) hostTadShapeInfo, new LongPointerWrapper(hostTadOffsets), (LongPointer) zTadShapeInfo,
                    new LongPointerWrapper(zTadOffsets));


        return ret;
    }

    public INDArray accumulate(INDArray target, INDArray... arrays) {

        if (arrays == null || arrays.length == 0)
            throw new RuntimeException("Input arrays are missing");

        if (arrays.length == 1)
            return target.addi(arrays[0]);

        long len = target.lengthLong();

        PointerPointer dataPointers = new PointerPointer(arrays.length);

        for (int i = 0; i < arrays.length; i++) {
            Nd4j.getCompressor().autoDecompress(arrays[i]);

            if (arrays[i].elementWiseStride() != 1)
                throw new ND4JIllegalStateException("Native accumulation is applicable only to continuous INDArrays");

            if (arrays[i].lengthLong() != len)
                throw new ND4JIllegalStateException("All arrays should have equal length for accumulation");

            dataPointers.put(i, arrays[i].data().addressPointer());
        }


        nativeOps.accumulate(null, dataPointers, (DoublePointer) target.data().addressPointer(), arrays.length, len);

        return target;
    }

    /**
     * This method averages input arrays, and returns averaged array
     *
     * @param target
     * @param arrays
     * @return
     */
    @Override
    public INDArray average(INDArray target, INDArray[] arrays) {
        if (arrays == null || arrays.length == 0)
            throw new RuntimeException("Input arrays are missing");

        if (arrays.length == 1)
            return target.assign(arrays[0]);

        long len = target != null ? target.lengthLong() : arrays[0].length();

        PointerPointer dataPointers = new PointerPointer(arrays.length);

        for (int i = 0; i < arrays.length; i++) {
            Nd4j.getCompressor().autoDecompress(arrays[i]);

            if (arrays[i].elementWiseStride() != 1)
                throw new ND4JIllegalStateException("Native averaging is applicable only to continuous INDArrays");

            if (arrays[i].lengthLong() != len)
                throw new ND4JIllegalStateException("All arrays should have equal length for averaging");

            dataPointers.put(i, arrays[i].data().addressPointer());
        }


        nativeOps.average(null, dataPointers, target == null ? null : target.data().addressPointer(), arrays.length,
                    len, true);

        return target;
    }

    /**
     * This method averages input arrays, and returns averaged array
     *
     * @param target
     * @param arrays
     * @return
     */
    @Override
    public INDArray average(INDArray target, Collection<INDArray> arrays) {
        return average(target, arrays.toArray(new INDArray[0]));
    }

    @Override
    public INDArray average(INDArray[] arrays) {
        if (arrays == null || arrays.length == 0)
            throw new RuntimeException("Input arrays are missing");

        INDArray ret = Nd4j.createUninitialized(arrays[0].shape(), arrays[0].ordering());

        return average(ret, arrays);
    }

    @Override
    public INDArray average(Collection<INDArray> arrays) {
        return average(arrays.toArray(new INDArray[0]));
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
    public void shuffle(INDArray array, Random rnd, int... dimension) {
        shuffle(Collections.singletonList(array), rnd, dimension);
    }

    /**
     * Symmetric in place shuffle of an ndarray
     * along a specified set of dimensions. All arrays
     *
     * @param array     the ndarray to shuffle
     * @param dimension the dimension to do the shuffle
     * @return
     */
    @Override
    public void shuffle(Collection<INDArray> array, Random rnd, int... dimension) {
        shuffle(new ArrayList<INDArray>(array), rnd, Collections.singletonList(dimension));
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
    public void shuffle(List<INDArray> arrays, Random rnd, List<int[]> dimensions) {
        if (dimensions == null || dimensions.size() == 0)
            throw new RuntimeException("Dimension can't be null or 0-length");

        if (arrays == null || arrays.size() == 0)
            throw new RuntimeException("No input arrays provided");

        if (dimensions.size() > 1 && arrays.size() != dimensions.size())
            throw new IllegalStateException("Number of dimensions do not match number of arrays to shuffle");

        int tadLength = 1;
        for (int i = 0; i < dimensions.get(0).length; i++) {
            tadLength *= arrays.get(0).shape()[dimensions.get(0)[i]];
        }

        long numTads = arrays.get(0).length() / tadLength;

        val map = ArrayUtil.buildInterleavedVector(rnd, (int) numTads);

        PointerPointer dataPointers = new PointerPointer(arrays.size());
        PointerPointer shapePointers = new PointerPointer(arrays.size());
        PointerPointer tadPointers = new PointerPointer(arrays.size());
        PointerPointer offsetPointers = new PointerPointer(arrays.size());

        PointerPointer dummy = new PointerPointer(new Pointer[] {null});

        List<Pair<DataBuffer, DataBuffer>> list = new ArrayList<>();

        TADManager tadManager = Nd4j.getExecutioner().getTADManager();

        val ptrMap = new IntPointer(map);

        long[] ptrs = new long[arrays.size()];


        for (int i = 0; i < arrays.size(); i++) {
            INDArray array = arrays.get(i);

            Nd4j.getCompressor().autoDecompress(array);


            int[] dimension = dimensions.size() > 1 ? dimensions.get(i) : dimensions.get(0);

            Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(array, dimension);
            list.add(tadBuffers);

            Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();

            DataBuffer offsets = tadBuffers.getSecond();

            if (offsets.length() != numTads)
                throw new ND4JIllegalStateException("Can't symmetrically shuffle arrays with non-equal number of TADs");

            if (offsets == null)
                throw new ND4JIllegalStateException("Offsets for shuffle can't be null");


            dataPointers.put(i, array.data().addressPointer());
            shapePointers.put(i, array.shapeInfoDataBuffer().addressPointer());
            offsetPointers.put(i, offsets.addressPointer());
            tadPointers.put(i, tadBuffers.getFirst().addressPointer());
        }


        nativeOps.shuffle(dummy, dataPointers, shapePointers, dataPointers, shapePointers, arrays.size(),
                    ptrMap, tadPointers, offsetPointers);


        dataPointers.address();
        shapePointers.address();
        tadPointers.address();
        offsetPointers.address();
    }


    /**
     * This method converts Half-precision databuffer to current dType buffer.
     *
     * @param buffer
     * @return
     */
    /*
    @Override
    public DataBuffer restoreFromHalfs(DataBuffer buffer) {
        if (buffer.dataType() != DataBuffer.Type.COMPRESSED)
            throw new IllegalStateException("DataBuffer contains wrong data: " + buffer.dataType());
    
        CompressedDataBuffer comp = (CompressedDataBuffer) buffer;
        CompressionDescriptor descriptor = comp.getCompressionDescriptor();
    
        DataBuffer targetBuffer = Nd4j.createBuffer(descriptor.getCompressedLength() / 2);
    
        if (Nd4j.dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.convertHalfsToDoubles(
                    null,
                    comp.addressPointer(),
                    (int) descriptor.getCompressedLength() / 2,
                    targetBuffer.addressPointer()
            );
        } else if (Nd4j.dataType() == DataBuffer.Type.FLOAT) {
            nativeOps.convertHalfsToFloats(
                    null,
                    comp.addressPointer(),
                    (int) descriptor.getCompressedLength() / 2,
                    targetBuffer.addressPointer()
            );
        } else {
            throw new UnsupportedOperationException("Target dtype isn't supported: " + Nd4j.dataType());
        }
    
        return targetBuffer;
    }
    */

    /**
     * This method converts Single/Double precision databuffer to Half-precision databuffer
     *
     * @param buffer
     * @return
     */
    /*@Override
    public DataBuffer convertToHalfs(DataBuffer buffer) {
        // we allocate pointer
        ShortPointer pointer = new ShortPointer(buffer.length());
    
        if (buffer.dataType() == DataBuffer.Type.DOUBLE) {
            nativeOps.convertDoublesToHalfs(
                    null,
                    buffer.addressPointer(),
                    (int) buffer.length(),
                    pointer
            );
        } else if (buffer.dataType() == DataBuffer.Type.FLOAT) {
            nativeOps.convertFloatsToHalfs(
                    null,
                    buffer.addressPointer(),
                    (int) buffer.length(),
                    pointer
            );
        } else {
            throw new UnsupportedOperationException("Source dtype isn't supported: " + buffer.dataType());
        }
    
        CompressionDescriptor descriptor = new CompressionDescriptor(buffer, new Float16());
        descriptor.setCompressedLength(buffer.length() * 2);
    
    
        CompressedDataBuffer result = new CompressedDataBuffer(pointer, descriptor);
        return result;
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
    public INDArray convertDataEx(DataBuffer.TypeEx typeSrc, INDArray source, DataBuffer.TypeEx typeDst) {
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
    public DataBuffer convertDataEx(DataBuffer.TypeEx typeSrc, DataBuffer source, DataBuffer.TypeEx typeDst) {
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

        DataBuffer buffer = null;


        if (CompressionUtils.goingToCompress(typeSrc, typeDst)) {
            // all types below 6 are compression modes
            BytePointer pointer = new BytePointer(source.length() * elementSize);
            CompressionDescriptor descriptor = new CompressionDescriptor(source, typeDst.name());
            descriptor.setCompressionType(CompressionType.LOSSY);
            descriptor.setCompressedLength(source.length() * elementSize);
            buffer = new CompressedDataBuffer(pointer, descriptor);
        } else {
            CompressedDataBuffer compressed = (CompressedDataBuffer) source;
            CompressionDescriptor descriptor = compressed.getCompressionDescriptor();

            // decompression mode
            buffer = Nd4j.createBuffer(descriptor.getNumberOfElements(), true);
        }

        convertDataEx(typeSrc, source, typeDst, buffer);

        return buffer;
    }

    @Override
    public void convertDataEx(DataBuffer.TypeEx typeSrc, Pointer source, DataBuffer.TypeEx typeDst, Pointer target,
                              long length) {
        nativeOps.convertTypes(null, typeSrc.ordinal(), source, length, typeDst.ordinal(), target);
    }

    @Override
    public void convertDataEx(DataBuffer.TypeEx typeSrc, Pointer source, DataBuffer.TypeEx typeDst, DataBuffer buffer) {
        convertDataEx(typeSrc, source, typeDst, buffer.addressPointer(), buffer.length());
    }

    @Override
    public void convertDataEx(DataBuffer.TypeEx typeSrc, DataBuffer source, DataBuffer.TypeEx typeDst,
                              DataBuffer target) {
        convertDataEx(typeSrc, source.addressPointer(), typeDst, target.addressPointer(), target.length());
    }



    @Override
    public INDArray createSparseCSR(double[] data, int[] columns, int[] pointerB, int[] pointerE, long[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray createSparseCSR(float[] data, int[] columns, int[] pointerB, int[] pointerE, long[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray createSparseCSR(DataBuffer data, int[] columns, int[] pointerB, int[] pointerE, long[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray createSparseCOO(double[] values, int[][] indices, long[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray createSparseCOO(float[] values, int[][] indices, long[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray createSparseCOO(double[] values, long[][] indices, long[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray createSparseCOO(float[] values, long[][] indices, long[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray createSparseCOO(DataBuffer values, DataBuffer indices, long[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray createSparseCOO(DataBuffer values, DataBuffer indices, DataBuffer sparseInformation, long[] shape) {
        throw new UnsupportedOperationException();
    }


    @Override
    public INDArray createSparseCOO(DataBuffer values, DataBuffer indices, long[] sparseOffsets, int[] flags, int[] hiddenDimensions, int underlyingRank, long[] shape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray sort(INDArray x, boolean descending) {
        if (x.isScalar())
            return x;


        NativeOpsHolder.getInstance().getDeviceNativeOps().sort(null, x.data().addressPointer(), (LongPointer) x.shapeInfoDataBuffer().addressPointer(), descending);

        return x;
    }

    @Override
    public INDArray sort(INDArray x, boolean descending, int... dimension) {
        if (x.isScalar())
            return x;

        Arrays.sort(dimension);
        Pair<DataBuffer, DataBuffer> tadBuffers = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x, dimension);


        NativeOpsHolder.getInstance().getDeviceNativeOps().sortTad(null,
                    x.data().addressPointer(),
                    (LongPointer) x.shapeInfoDataBuffer().addressPointer(),
                    (IntPointer) Nd4j.getConstantHandler().getConstantBuffer(dimension).addressPointer(),
                    dimension.length,
                    (LongPointer) tadBuffers.getFirst().addressPointer(),
                    new LongPointerWrapper(tadBuffers.getSecond().addressPointer()),
                    descending);


        return x;
    }

    @Override
    public INDArray sortCooIndices(INDArray x) {
        throw new UnsupportedOperationException("Not an COO ndarray");
    }
}
