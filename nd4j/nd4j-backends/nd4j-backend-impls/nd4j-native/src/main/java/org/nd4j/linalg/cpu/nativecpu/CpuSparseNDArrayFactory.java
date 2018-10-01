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
import org.bytedeco.javacpp.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.DataTypeEx;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArrayCOO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.cpu.nativecpu.blas.*;
import org.nd4j.linalg.factory.BaseSparseNDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.nativeblas.LongPointerWrapper;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * @author Audrey Loeffel
 */

// TODO : Implement the methods

@Slf4j
public class CpuSparseNDArrayFactory extends BaseSparseNDArrayFactory {

    public CpuSparseNDArrayFactory(){}

    @Override
    public INDArray createSparseCSR(double[] data, int[] columns, int[] pointerB, int[] pointerE, long[] shape){
        return new SparseNDArrayCSR(data, columns, pointerB, pointerE, shape);
    }
    @Override
    public INDArray createSparseCSR(float[] data, int[] columns, int[] pointerB, int[] pointerE, long[] shape){
        return new SparseNDArrayCSR(data, columns, pointerB, pointerE, shape);
    }
    @Override
    public INDArray createSparseCSR(DataBuffer data, int[] columns, int[] pointerB, int[] pointerE, long[] shape){
        return new SparseNDArrayCSR(data, columns, pointerB, pointerE, shape);
    }

    @Override
    public INDArray createSparseCOO(double[] values, int[][] indices, long[] shape){
        return new SparseNDArrayCOO(values, indices, shape);
    }

    @Override
    public INDArray createSparseCOO(float[] values, int[][] indices, long[] shape){
        return new SparseNDArrayCOO(values, indices, shape);
    }

    @Override
    public INDArray create(DataType dataType, long[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray createUninitialized(DataType dataType, long[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray createSparseCOO(double[] values, long[][] indices, long[] shape) {
        return new SparseNDArrayCOO(values, indices, shape);
    }

    @Override
    public INDArray createSparseCOO(float[] values, long[][] indices, long[] shape) {
        return new SparseNDArrayCOO(values, indices, shape);
    }

    @Override
    public INDArray createSparseCOO(DataBuffer values, DataBuffer indices, long[] shape){
        return new SparseNDArrayCOO(values, indices, shape);
    }

    @Override
    public INDArray createSparseCOO(DataBuffer values, DataBuffer indices, long[] sparseOffsets, int[] flags, int[] hiddenDimensions, int underlyingRank, long[] shape) {
        return new SparseNDArrayCOO(values, indices, sparseOffsets, flags, hiddenDimensions, underlyingRank, shape);
    }

    @Override
    public INDArray createSparseCOO(DataBuffer values, DataBuffer indices, DataBuffer sparseInformation, long[] shape) {
        return new SparseNDArrayCOO(values, indices, sparseInformation, shape);
    }
    //  TODO ->


    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, long[] indexes) {
        return null;
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, DataType dataType) {
        return null;
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, DataType dataType) {
        return null;
    }

    @Override
    public INDArray create(long[] data, long[] shape, long[] stride, DataType dataType) {
        return null;
    }

    @Override
    public INDArray create(int[] data, long[] shape, long[] stride, DataType dataType) {
        return null;
    }

    @Override
    public INDArray create(short[] data, long[] shape, long[] stride, DataType dataType) {
        return null;
    }

    @Override
    public INDArray create(byte[] data, long[] shape, long[] stride, DataType dataType) {
        return null;
    }

    @Override
    public INDArray create(boolean[] data, long[] shape, long[] stride, DataType dataType) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(List<INDArray> list, long[] shape) {
        return null;
    }

    @Override
    public INDArray create(long rows, long columns, long[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(long[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray createUninitialized(long[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray createUninitializedDetached(long[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, char ordering) {
        return null;
    }

    @Override
    public INDArray create(List<INDArray> list, long[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, char order, long offset) {
        return null;
    }

    @Override
    public INDArray trueScalar(Number value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray specialConcat(int dimension, INDArray... toConcat) {
        return null;
    }

    @Override
    public INDArray pullRows(INDArray source, INDArray destination, int sourceDimension, int[] indexes) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(List<INDArray> list, int[] shape) {
        return null;
    }

    static{
        Nd4j.getBlasWrapper();
    }
    // contructors ?

    @Override
    public void createBlas(){ blas = new SparseCpuBlas();}

    @Override
    public void createLevel1() {
        level1 = new SparseCpuLevel1();
    }

    @Override
    public void createLevel2() {
        level2 = new SparseCpuLevel2();
    }

    @Override
    public void createLevel3() { level3 = new SparseCpuLevel3(); }

    @Override
    public void createLapack() {
        lapack = new SparseCpuLapack();
    }

    @Override
    public INDArray create(int[] shape, DataBuffer buffer) {
        return null;
    }

    @Override
    public INDArray toFlattened(char order, Collection<INDArray> matrices) {
        return null;
    }

    @Override
    public INDArray create(double[][] data) {
        return null;
    }

    @Override
    public INDArray create(double[][] data, char ordering) {
        return null;
    }

    @Override
    public void shuffle(INDArray array, Random rnd, int... dimension) {

    }

    @Override
    public void shuffle(Collection<INDArray> array, Random rnd, int... dimension) {

    }

    @Override
    public void shuffle(List<INDArray> array, Random rnd, List<int[]> dimensions) {

    }

    @Override
    public INDArray average(INDArray target, INDArray[] arrays) {
        return null;
    }

    @Override
    public INDArray average(INDArray[] arrays) {
        return null;
    }

    @Override
    public INDArray average(Collection<INDArray> arrays) {
        return null;
    }

    @Override
    public INDArray accumulate(INDArray target, INDArray... arrays) {
        return null;
    }

    @Override
    public INDArray average(INDArray target, Collection<INDArray> arrays) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, long rows, long columns, int[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape, int[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(float[][] floats) {
        return null;
    }

    @Override
    public INDArray create(float[][] data, char ordering) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, long offset, char ordering) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer buffer, int[] shape, long offset) {
        return null;
    }

    @Override
    public INDArray create(int[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray createUninitialized(int[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray createUninitializedDetached(int[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, int[] newShape, int[] newStride, long offset, char ordering) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int[] shape, long offset, Character order) {
        return null;
    }

    @Override
    public INDArray create(float[] data, long rows, long columns, int[] stride, long offset, char ordering) {
        return null;
    }

    @Override
    public INDArray create(double[] data, int[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray create(List<INDArray> list, int[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray create(double[] data, int[] shape, long offset) {
        return null;
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, long offset, char ordering) {
        return null;
    }

    @Override
    public INDArray convertDataEx(DataTypeEx typeSrc, INDArray source, DataTypeEx typeDst) {
        return null;
    }

    @Override
    public DataBuffer convertDataEx(DataTypeEx typeSrc, DataBuffer source, DataTypeEx typeDst) {
        return null;
    }

    @Override
    public void convertDataEx(DataTypeEx typeSrc, DataBuffer source, DataTypeEx typeDst, DataBuffer target) {

    }

    @Override
    public void convertDataEx(DataTypeEx typeSrc, Pointer source, DataTypeEx typeDst, Pointer target, long length) {

    }

    @Override
    public void convertDataEx(DataTypeEx typeSrc, Pointer source, DataTypeEx typeDst, DataBuffer buffer) {

    }

    @Override
    public INDArray createFromNpyPointer(Pointer pointer) {
        return null;
    }

    @Override
    public INDArray createFromNpyHeaderPointer(Pointer pointer) {
        return null;
    }

    @Override
    public INDArray createFromNpyFile(File file) {
        return null;
    }

    @Override
    public Pointer convertToNumpy(INDArray array) {
        return null;
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, long offset, char ordering) {
        return null;
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, long offset, char ordering) {
        return null;
    }

    @Override
    public INDArray[] tear(INDArray tensor, int... dimensions) {
        return new INDArray[0];
    }

    @Override
    public INDArray sort(INDArray x, boolean descending) {
        if (x.isScalar())
            return x;

        NativeOpsHolder.getInstance().getDeviceNativeOps().sort(null,  x.data().addressPointer(), (LongPointer) x.shapeInfoDataBuffer().addressPointer(), descending);

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
                    new IntPointer(dimension),
                    dimension.length,
                    (LongPointer) tadBuffers.getFirst().addressPointer(),
                    new LongPointerWrapper(tadBuffers.getSecond().addressPointer()),
                    descending);


        return x;
    }

    @Override
    public INDArray sortCooIndices(INDArray x) {

        if(x.getFormat() != SparseFormat.COO){
            throw new UnsupportedOperationException("Not a COO ndarray");
        }
        BaseSparseNDArrayCOO array = (BaseSparseNDArrayCOO) x;
        DataBuffer val = array.getValues();
        DataBuffer idx = array.getIndices();
        long length = val.length();
        int rank = array.underlyingRank();

        NativeOpsHolder.getInstance().getDeviceNativeOps().sortCooIndices(null, (LongPointer) idx.addressPointer(), val.addressPointer(), length, rank);


        return array;
    }

    @Override
    public INDArray create(float[] data, long[] shape, long offset, Character order) {
        return null;
    }

    @Override
    public INDArray create(double[] data, long[] shape, long offset, Character order) {
        return null;
    }

    @Override
    public INDArray create(float[] data, long[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray create(double[] data, long[] shape, char ordering) {
        return null;
    }


    @Override
    public INDArray empty(DataType type) {
        throw new UnsupportedOperationException();
    }
}
