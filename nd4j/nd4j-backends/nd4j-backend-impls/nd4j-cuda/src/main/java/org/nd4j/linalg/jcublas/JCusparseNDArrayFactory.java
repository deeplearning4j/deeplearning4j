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

package org.nd4j.linalg.jcublas;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.ISparseNDArray;
import org.nd4j.linalg.factory.BaseSparseNDArrayFactory;
import org.nd4j.linalg.jcublas.blas.*;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * @author Audrey Loeffel
 */
@Slf4j
public class JCusparseNDArrayFactory extends BaseSparseNDArrayFactory{

    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    public JCusparseNDArrayFactory(){}

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(float[] data, long[] shape, long[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(double[] data, long[] shape, long[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(List<INDArray> list, int[] shape) {
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
    public INDArray empty(DataBuffer.Type type) {
        throw new IllegalStateException();
    }

    @Override
    public void createBlas() {
        blas = new SparseCudaBlas();
    }

    @Override
    public void createLevel1() {
        level1 = new JcusparseLevel1();
    }

    @Override
    public void createLevel2() {
        level2 = new JcusparseLevel2();
    }

    @Override
    public void createLevel3() {
        level3 = new JcusparseLevel3();
    }

    @Override
    public void createLapack() {
        lapack = new JcusparseLapack();
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
    public INDArray specialConcat(int dimension, INDArray... toConcat) {
        return null;
    }

    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, long[] indexes) {
        return null;
    }

    @Override
    public INDArray pullRows(INDArray source, INDArray destination, int sourceDimension, int[] indexes) {
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
    public INDArray create(DataBuffer data, long[] shape) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape, int[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, long[] shape, long[] stride, long offset) {
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
    public INDArray create(long[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray createUninitialized(int[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray createUninitialized(long[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray createUninitializedDetached(int[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray createUninitializedDetached(long[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, int[] newShape, int[] newStride, long offset, char ordering) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, long[] newShape, long[] newStride, long offset, char ordering) {
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
    public INDArray create(List<INDArray> list, long[] shape, char ordering) {
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
    public INDArray create(float[] data, long[] shape, long[] stride, char order, long offset) {
        return null;
    }

    @Override
    public INDArray convertDataEx(DataBuffer.TypeEx typeSrc, INDArray source, DataBuffer.TypeEx typeDst) {
        return null;
    }

    @Override
    public DataBuffer convertDataEx(DataBuffer.TypeEx typeSrc, DataBuffer source, DataBuffer.TypeEx typeDst) {
        return null;
    }

    @Override
    public void convertDataEx(DataBuffer.TypeEx typeSrc, DataBuffer source, DataBuffer.TypeEx typeDst, DataBuffer target) {

    }

    @Override
    public void convertDataEx(DataBuffer.TypeEx typeSrc, Pointer source, DataBuffer.TypeEx typeDst, Pointer target, long length) {

    }

    @Override
    public void convertDataEx(DataBuffer.TypeEx typeSrc, Pointer source, DataBuffer.TypeEx typeDst, DataBuffer buffer) {

    }

    @Override
    public INDArray createFromNpyPointer(Pointer pointer) {
        return null;
    }

    @Override
    public INDArray createFromNpyFile(File file) {
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
        return null;
    }

    @Override
    public INDArray sort(INDArray x, boolean descending, int... dimensions) {
        return null;
    }

    @Override
    public INDArray sortCooIndices(INDArray x) {
        //TODO
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray ravelCooIndices(INDArray x, char clipMode) {
        //TODO
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray unravelCooIndices(INDArray x, DataBuffer shapeInfo) {
        //TODO
        throw new UnsupportedOperationException();
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
    public ISparseNDArray createSparseCSR(double[] data, int[] columns, int[] pointerB, int[] pointerE, long[] shape) {
        return new JcusparseNDArrayCSR(data, columns, pointerB, pointerE, shape);
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
    public INDArray createSparseCOO(double[] values, long[][] indices, long[] shape) {
        return new JCusparseNDArrayCOO(values, indices, shape);
    }

    @Override
    public INDArray createSparseCOO(float[] values, long[][] indices, long[] shape) {
        return new JCusparseNDArrayCOO(values, indices, shape);
    }

    @Override
    public INDArray createSparseCOO(double[] values, int[][] indices, long[] shape) {
        return new JCusparseNDArrayCOO(values, indices, shape);
    }

    @Override
    public INDArray createSparseCOO(float[] values, int[][] indices, long[] shape) {
        return new JCusparseNDArrayCOO(values, indices, shape);
    }

    @Override
    public INDArray createSparseCOO(DataBuffer values, DataBuffer indices, long[] shape) {
        return new JCusparseNDArrayCOO(values, indices, shape);
    }

    @Override
    public INDArray createSparseCOO(DataBuffer values, DataBuffer indices, DataBuffer sparseInformation, long[] shape) {
        return new JCusparseNDArrayCOO(values, indices, sparseInformation, shape);
    }

    @Override
    public INDArray createSparseCOO(DataBuffer values, DataBuffer indices, long[] sparseOffsets, int[] flags, int[] hiddenDimensions, int underlyingRank, long[] shape) {
        return new JCusparseNDArrayCOO(values, indices, sparseOffsets, flags, hiddenDimensions, underlyingRank, shape);
    }


}
