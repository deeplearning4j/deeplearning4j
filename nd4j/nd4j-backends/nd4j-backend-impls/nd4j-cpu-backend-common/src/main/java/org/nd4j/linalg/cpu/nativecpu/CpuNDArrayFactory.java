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

package org.nd4j.linalg.cpu.nativecpu;


import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.*;
import org.nd4j.linalg.api.ops.custom.Flatten;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.linalg.compression.CompressionUtils;
import org.nd4j.linalg.cpu.nativecpu.buffer.*;
import org.nd4j.common.primitives.Pair;
import org.bytedeco.javacpp.*;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.cpu.nativecpu.blas.*;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.nativeblas.*;

import java.util.*;

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


    public CpuNDArrayFactory(DataType dtype, Character order) {
        super(dtype, order);
    }

    public CpuNDArrayFactory(DataType dtype, char order) {
        super(dtype, order);
    }

    @Override
    public void createBlas() {
        // we'll check hardware support first
        if (!nativeOps.isMinimalRequirementsMet()) {
            // this means cpu binary was built for some arch support, we don't have on this box

            val binaryLevel = nativeOps.binaryLevel();
            val optimalLevel = nativeOps.optimalLevel();

            String binLevel = cpuBinaryLevelToName(binaryLevel);
            String optLevel = cpuBinaryLevelToName(optimalLevel);

            log.warn("*********************************** CPU Feature Check Failed ***********************************");
            log.error("Error initializing ND4J: Attempting to use " + binLevel + " ND4J binary on a CPU with only " + optLevel + " support");
            log.error( binLevel + " binaries cannot be run on a CPU without these instructions. See https://deeplearning4j.konduit.ai/multi-project/explanation/configuration/backends/performance-issues#step-11-check-cpu-support-for-hardware-extensions-avx-etc for more details");
            log.error("ND4J will now exit.");
            log.warn("************************************************************************************************");
            System.exit(1);
        }

        val binaryLevel = nativeOps.binaryLevel();
        val optimalLevel = nativeOps.optimalLevel();

        String binLevel = cpuBinaryLevelToName(binaryLevel);
        String optLevel = cpuBinaryLevelToName(optimalLevel);
        log.info("Binary level " + binLevel + " optimization level " + optLevel);
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

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());
    }

    private static String cpuBinaryLevelToName(int level) {
        switch (level){
            case 3:
                return "AVX512";
            case 2:
                return "AVX/AVX2";
            case 1:
            case 0:
            default:
                return "Generic x86";
        }
    }

    @Override
    public INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, long ews, char ordering, boolean isView) {
        return new NDArray(data,newShape,newStride,offset,ews,ordering,isView);
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

    @Override
    public INDArray create(DataBuffer buffer, LongShapeDescriptor longShapeDescriptor) {
        return new NDArray(buffer, longShapeDescriptor);
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
    public INDArray create(DataType dataType, long[] shape, char ordering, MemoryWorkspace workspace) {
        return new NDArray(dataType, shape, Nd4j.getStrides(shape, ordering), 0, ordering, workspace);
    }

    @Override
    public INDArray create(DataType dataType, long[] shape, long[] strides,  char ordering, MemoryWorkspace workspace ) {
        return new NDArray(dataType, shape, strides, 0, ordering);
    }

    @Override
    public INDArray createUninitialized(DataType dataType, long[] shape, char ordering, MemoryWorkspace workspace) {
        return new NDArray(dataType, shape, Nd4j.getStrides(shape, ordering), 0, ordering, false, workspace);
    }

    @Override
    public INDArray createUninitialized(DataType dataType, long[] shape, long[] strides, char ordering) {
        return super.createUninitialized(dataType, shape, strides, ordering);
    }

    @Override
    public INDArray createUninitializedDetached(DataType dataType, char ordering, long... shape){
        MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        INDArray ret = new NDArray(dataType, shape, Nd4j.getStrides(shape, ordering), 0, ordering, false);
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
        boolean hasZeros = false;
        for (long v : shape) {
            if (v == 0) {
                hasZeros = true;
                break;
            }
        }
        return new NDArray(hasZeros ? null : Nd4j.createBuffer(data), shape, ordering);
    }

    @Override
    public INDArray create(double[] data, long[] shape, char ordering) {
        return create(data, shape, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, char ordering) {
        return create(data, shape, ordering);
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
        boolean hasZeros = false;
        for (long v : shape) {
            if (v == 0) {
                hasZeros = true;
                break;
            }
        }
        return new NDArray(hasZeros ? null : Nd4j.createTypedBuffer(data, DataType.DOUBLE), shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, long offset, char ordering) {
        return new NDArray(Nd4j.createTypedBuffer(data, DataType.DOUBLE), shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(LongShapeDescriptor longShapeDescriptor) {
        return new NDArray(longShapeDescriptor);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, long offset, char ordering) {
        return new NDArray(Nd4j.createTypedBuffer(data, DataType.FLOAT), shape, stride, offset, ordering);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, long offset) {
        return new NDArray(data, shape, stride, offset, Nd4j.order());
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, char order, DataType dataType) {
        return new NDArray(data, shape, stride, 0, order);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, long offset) {
        return new NDArray(data, shape, stride, offset, Nd4j.order());
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride,  Nd4j.order(), dataType);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType), shape, stride,  order, dataType, workspace);
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  order, dataType, workspace);
    }

    @Override
    public INDArray create(long[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  order, dataType,workspace);
    }

    @Override
    public INDArray create(int[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  order, dataType,workspace);
    }

    @Override
    public INDArray create(short[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  order, dataType,workspace);
    }

    @Override
    public INDArray create(byte[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  order, dataType,workspace);
    }

    @Override
    public INDArray create(boolean[] data, long[] shape, long[] stride, char order, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  order, dataType,workspace);
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  Nd4j.order(), dataType,workspace);
    }

    @Override
    public INDArray create(long[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  Nd4j.order(), dataType,workspace);
    }

    @Override
    public INDArray create(int[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  Nd4j.order(), dataType,workspace);
    }

    @Override
    public INDArray create(short[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  Nd4j.order(), dataType,workspace);
    }

    @Override
    public INDArray create(boolean[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  Nd4j.order(), dataType,workspace);
    }

    @Override
    public INDArray create(byte[] data, long[] shape, long[] stride, DataType dataType, MemoryWorkspace workspace) {
        return new NDArray(Nd4j.createTypedBuffer(data, dataType,workspace), shape, stride,  Nd4j.order(), dataType,workspace);
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
    public INDArray create(DataBuffer data, long[] shape, long[] stride, long offset, long ews, char ordering) {
        return new NDArray(data, shape, stride, offset, ews, ordering);
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride, long offset, char ordering, DataType dataType) {
        return new NDArray(data, shape, stride, offset, ordering, dataType);
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
    public INDArray empty(DataType type) {
        long extras  = ArrayOptionsHelper.setOptionBit(0L, ArrayType.EMPTY);
        extras = ArrayOptionsHelper.setOptionBit(extras, type);
        val shape = Nd4j.getShapeInfoProvider().createShapeInformation(new long[0], new long[0],1,'c', extras);
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
        Preconditions.checkArgument(matrices.size() > 0, "toFlattened expects > 0 operands");

        return Nd4j.exec(new Flatten(order, matrices.toArray(new INDArray[matrices.size()])))[0];
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

        return Nd4j.exec(new Concat(dimension, toConcat))[0];
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

        return pullRows(source, Nd4j.createUninitialized(source.dataType(), shape, order), sourceDimension, indexes);
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
            ret = Nd4j.createUninitialized(source.dataType(), shape, order);
        } else {
            if(!Arrays.equals(shape, destination.shape())) {
                throw new IllegalStateException("Cannot pull rows into destination array: expected destination array of" +
                        " shape " + Arrays.toString(shape) + " but got destination array of shape " + Arrays.toString(destination.shape()));
            }
        }

        Nd4j.getCompressor().autoDecompress(source);

        OpaqueNDArray sourceOpaque = OpaqueNDArray.fromINDArray(source);
        OpaqueNDArray retOpaque = OpaqueNDArray.fromINDArray(ret);
        OpaqueNDArray indexOpaque = OpaqueNDArray.fromINDArray(Nd4j.createFromArray(indexes));


        nativeOps.pullRows(null,
                sourceOpaque, retOpaque,
                indexes.length, indexOpaque,
                sourceDimension);

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        return ret;
    }

    public INDArray accumulate(INDArray target, INDArray... arrays) {

        if (arrays == null || arrays.length == 0)
            throw new RuntimeException("Input arrays are missing");

        if (arrays.length == 1)
            return target.addi(arrays[0]);

        long len = target.length();

        OpaqueNDArray targetOpaque = OpaqueNDArray.fromINDArray(target);
        OpaqueNDArrayArr arraysOpaque = new OpaqueNDArrayArr(Arrays.stream(arrays)
                .map(OpaqueNDArray::fromINDArray)
                .toArray(OpaqueNDArray[]::new));


        nativeOps.accumulate(null,
                arraysOpaque, targetOpaque,
                arrays.length, len);

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

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

        if (arrays.length == 1) {
            // Edge case - average 1 array - no op
            if (target == null) {
                return null;
            }
            return target.assign(arrays[0]);
        }

        long len = target != null ? target.length() : arrays[0].length();

        OpaqueNDArray targetOpaque = OpaqueNDArray.fromINDArray(target);
        OpaqueNDArrayArr arraysOpaque = new OpaqueNDArrayArr(Arrays.stream(arrays)
                .map(OpaqueNDArray::fromINDArray)
                .toArray(OpaqueNDArray[]::new));


        nativeOps.average(null,
                arraysOpaque, targetOpaque,
                arrays.length, len, true);

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

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

        INDArray ret = Nd4j.createUninitialized(arrays[0].dataType(), arrays[0].shape(), arrays[0].ordering());

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
    public void shuffle(INDArray array, Random rnd, long... dimension) {
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
    public void shuffle(Collection<INDArray> array, Random rnd, long... dimension) {
        shuffle(new ArrayList<>(array), rnd, Collections.singletonList(dimension));
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
        if (dimensions == null || dimensions.size() == 0)
            throw new RuntimeException("Dimension can't be null or 0-length");

        if (arrays == null || arrays.size() == 0)
            throw new RuntimeException("No input arrays provided");

        if (dimensions.size() > 1 && arrays.size() != dimensions.size())
            throw new IllegalStateException("Number of dimensions do not match number of arrays to shuffle");

        val zero = arrays.get(0);
        int tadLength = 1;
        if (zero.rank() > 1)
            for (int i = 0; i < dimensions.get(0).length; i++) {
                tadLength *= zero.size(dimensions.get(0)[i]);
            }

        long numTads = zero.length() / tadLength;

        val map = ArrayUtil.buildInterleavedVector(rnd, (int) numTads);

        OpaqueNDArrayArr arraysOpaque = new OpaqueNDArrayArr(arrays.stream()
                .map(OpaqueNDArray::fromINDArray)
                .toArray(OpaqueNDArray[]::new));

        INDArray mapArray = Nd4j.createFromArray(map);
        OpaqueNDArray ptrMap = OpaqueNDArray.fromINDArray(mapArray);



        // Convert List<long[]> to long[][]
        long[][] dimensionsArray = new long[dimensions.size()][];
        for (int i = 0; i < dimensions.size(); i++) {
            dimensionsArray[i] = dimensions.get(i);
        }

        // Create an INDArray from the long[][]
        INDArray dimensionsINDArray = Nd4j.createFromArray(dimensionsArray);

        // Convert the INDArray to an OpaqueNDArray
        OpaqueNDArray dimensionsOpaque = OpaqueNDArray.fromINDArray(dimensionsINDArray);


        nativeOps.shuffle(null,
                arraysOpaque, arraysOpaque,
                arrays.size(),
                dimensionsOpaque, ptrMap);

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());
    }

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
        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        source.setData(buffer);

        if (buffer instanceof CompressedDataBuffer)
            source.markAsCompressed(true);
        else
            source.markAsCompressed(false);

        return source;
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

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        return buffer;
    }

    @Override
    public void convertDataEx(DataTypeEx typeSrc, Pointer source, DataTypeEx typeDst, Pointer target,
                              long length) {
        nativeOps.convertTypes(null, typeSrc.ordinal(), source, length, typeDst.ordinal(), target);

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());
    }

    @Override
    public void convertDataEx(DataTypeEx typeSrc, Pointer source, DataTypeEx typeDst, DataBuffer buffer) {
        convertDataEx(typeSrc, source, typeDst, buffer.addressPointer(), buffer.length());
    }

    @Override
    public void convertDataEx(DataTypeEx typeSrc, DataBuffer source, DataTypeEx typeDst,
                              DataBuffer target) {
        convertDataEx(typeSrc, source.addressPointer(), typeDst, target.addressPointer(), target.length());
    }

    @Override
    public INDArray sort(INDArray x, boolean descending) {
        if (x.isScalar())
            return x;

        OpaqueNDArray xOpaque = OpaqueNDArray.fromINDArray(x);

        nativeOps.sort(null,
                xOpaque,
                descending);

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        return x;
    }

    @Override
    public INDArray sort(INDArray x, boolean descending, long... dimension) {
        if (x.isScalar())
            return x;

        Arrays.sort(dimension);
        OpaqueNDArray xOpaque = OpaqueNDArray.fromINDArray(x);
        OpaqueNDArray dimensionOpaque = OpaqueNDArray.fromINDArray(Nd4j.createFromArray(dimension));

        nativeOps.sort(null,
                xOpaque,
                descending);

        if (nativeOps.lastErrorCode() != 0)
            throw new RuntimeException(nativeOps.lastErrorMessage());

        return x;
    }
    @Override
    public INDArray sortCooIndices(INDArray x) {
        throw new UnsupportedOperationException("Not an COO ndarray");
    }


    @Override
    public INDArray create(Collection<String> strings, long[] shape, char order) {
        val pairShape = Nd4j.getShapeInfoProvider().createShapeInformation(shape, order, DataType.UTF8);
        val buffer = new Utf8Buffer(strings);
        return Nd4j.createArrayFromShapeBuffer(buffer, pairShape);
    }

    @Override
    public INDArray createUninitialized(DataType dataType, long[] shape, long[] strides, char ordering, MemoryWorkspace currentWorkspace) {
        return new NDArray(dataType,shape,strides,0,ordering,currentWorkspace);
    }

    @Override
    public INDArray create(DataType dataType, long[] shape, long[] paddings, long[] paddingOffsets, char ordering,
                           MemoryWorkspace workspace) {
        return new NDArray(dataType, shape, paddings, paddingOffsets, ordering, workspace);
    }
}
