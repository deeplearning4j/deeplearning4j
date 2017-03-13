package org.nd4j.linalg.factory;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.blas.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.rng.distribution.Distribution;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Created by audrey on 3/2/17.
 */
public abstract class ISparseMatrixFactory implements NDArrayFactory {

    protected char order;
    protected Blas blas;
    protected Level1 level1;
    protected Level2 level2;
    protected Level3 level3;
    protected Lapack lapack;

    @Override
    public Blas blas() {
        return null;
    }

    @Override
    public Lapack lapack() {
        return null;
    }

    @Override
    public Level1 level1() {
        return null;
    }

    @Override
    public Level2 level2() {
        return null;
    }

    @Override
    public Level3 level3() {
        return null;
    }

    @Override
    public void createBlas() {

    }

    @Override
    public void createLevel1() {

    }

    @Override
    public void createLevel2() {

    }

    @Override
    public void createLevel3() {

    }

    @Override
    public void createLapack() {

    }

    @Override
    public IComplexNDArray complexValueOf(int num, IComplexNumber value) {
        return null;
    }

    @Override
    public IComplexNDArray complexValueOf(int[] shape, IComplexNumber value) {
        return null;
    }

    @Override
    public IComplexNDArray complexValueOf(int num, double value) {
        return null;
    }

    @Override
    public IComplexNDArray complexValueOf(int[] shape, double value) {
        return null;
    }

    @Override
    public void setOrder(char order) {

    }

    @Override
    public void setDType(DataBuffer.Type dtype) {

    }

    @Override
    public INDArray create(int[] shape, DataBuffer buffer) {
        return null;
    }

    @Override
    public char order() {
        return 0;
    }

    @Override
    public DataBuffer.Type dtype() {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(int rows, int columns, int[] stride, int offset) {
        return null;
    }

    @Override
    public INDArray linspace(int lower, int upper, int num) {
        return null;
    }

    @Override
    public INDArray toFlattened(Collection<INDArray> matrices) {
        return null;
    }

    @Override
    public INDArray toFlattened(int length, Iterator<? extends INDArray>[] matrices) {
        return null;
    }

    @Override
    public INDArray toFlattened(char order, Collection<INDArray> matrices) {
        return null;
    }

    @Override
    public INDArray bilinearProducts(INDArray curr, INDArray in) {
        return null;
    }

    @Override
    public INDArray toFlattened(INDArray... matrices) {
        return null;
    }

    @Override
    public INDArray toFlattened(char order, INDArray... matrices) {
        return null;
    }

    @Override
    public INDArray eye(int n) {
        return null;
    }

    @Override
    public void rot90(INDArray toRotate) {

    }

    @Override
    public INDArray rot(INDArray reverse) {
        return null;
    }

    @Override
    public INDArray reverse(INDArray reverse) {
        return null;
    }

    @Override
    public INDArray arange(double begin, double end) {
        return null;
    }

    @Override
    public IComplexFloat createFloat(float real, float imag) {
        return null;
    }

    @Override
    public IComplexDouble createDouble(double real, double imag) {
        return null;
    }

    @Override
    public void copy(INDArray a, INDArray b) {

    }

    @Override
    public INDArray rand(int[] shape, float min, float max, Random rng) {
        return null;
    }

    @Override
    public INDArray rand(int rows, int columns, float min, float max, Random rng) {
        return null;
    }

    @Override
    public INDArray appendBias(INDArray... vectors) {
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
    public IComplexNDArray createComplex(INDArray arr) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(List<IComplexNDArray> arrs, int[] shape) {
        return null;
    }

    @Override
    public INDArray concat(int dimension, INDArray... toConcat) {
        return null;
    }

    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, int[] indexes) {
        return null;
    }

    @Override
    public INDArray pullRows(INDArray source, int sourceDimension, int[] indexes, char order) {
        return null;
    }

    @Override
    public void shuffle(INDArray array, java.util.Random rnd, int... dimension) {

    }

    @Override
    public void shuffle(Collection<INDArray> array, java.util.Random rnd, int... dimension) {

    }

    @Override
    public void shuffle(List<INDArray> array, java.util.Random rnd, List<int[]> dimensions) {

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
    public INDArray average(INDArray target, Collection<INDArray> arrays) {
        return null;
    }

    @Override
    public IComplexNDArray concat(int dimension, IComplexNDArray... toConcat) {
        return null;
    }

    @Override
    public INDArray rand(int rows, int columns, Random r) {
        return null;
    }

    @Override
    public INDArray rand(int rows, int columns, long seed) {
        return null;
    }

    @Override
    public INDArray rand(int rows, int columns) {
        return null;
    }

    @Override
    public INDArray rand(char order, int rows, int columns) {
        return null;
    }

    @Override
    public INDArray randn(int rows, int columns, Random r) {
        return null;
    }

    @Override
    public INDArray randn(int rows, int columns) {
        return null;
    }

    @Override
    public INDArray randn(char order, int rows, int columns) {
        return null;
    }

    @Override
    public INDArray randn(int rows, int columns, long seed) {
        return null;
    }

    @Override
    public INDArray rand(int[] shape, Distribution r) {
        return null;
    }

    @Override
    public INDArray rand(int[] shape, Random r) {
        return null;
    }

    @Override
    public INDArray rand(int[] shape, long seed) {
        return null;
    }

    @Override
    public INDArray rand(int[] shape) {
        return null;
    }

    @Override
    public INDArray rand(char order, int[] shape) {
        return null;
    }

    @Override
    public INDArray randn(int[] shape, Random r) {
        return null;
    }

    @Override
    public INDArray randn(int[] shape) {
        return null;
    }

    @Override
    public INDArray randn(char order, int[] shape) {
        return null;
    }

    @Override
    public INDArray randn(int[] shape, long seed) {
        return null;
    }

    @Override
    public INDArray create(double[] data) {
        return null;
    }

    @Override
    public INDArray create(float[] data) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(double[] data) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data) {
        return null;
    }

    @Override
    public INDArray create(int columns) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(int columns) {
        return null;
    }

    @Override
    public INDArray zeros(int rows, int columns) {
        return null;
    }

    @Override
    public IComplexNDArray complexZeros(int rows, int columns) {
        return null;
    }

    @Override
    public INDArray zeros(int columns) {
        return null;
    }

    @Override
    public IComplexNDArray complexZeros(int columns) {
        return null;
    }

    @Override
    public INDArray valueArrayOf(int[] shape, double value) {
        return null;
    }

    @Override
    public INDArray valueArrayOf(int rows, int columns, double value) {
        return null;
    }

    @Override
    public INDArray ones(int rows, int columns) {
        return null;
    }

    @Override
    public IComplexNDArray complexOnes(int rows, int columns) {
        return null;
    }

    @Override
    public INDArray ones(int columns) {
        return null;
    }

    @Override
    public IComplexNDArray complexOnes(int columns) {
        return null;
    }

    @Override
    public INDArray hstack(INDArray... arrs) {
        return null;
    }

    @Override
    public INDArray vstack(INDArray... arrs) {
        return null;
    }

    @Override
    public INDArray zeros(int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray complexZeros(int[] shape) {
        return null;
    }

    @Override
    public INDArray ones(int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray complexOnes(int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int rows, int columns, int[] stride, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, int rows, int columns, int[] stride, int offset) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, int rows, int columns, int[] stride, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, int[] shape, int[] stride, int offset) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int rows, int columns, int[] stride, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, int offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, char ordering) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, int offset) {
        return null;
    }

    @Override
    public INDArray create(double[] data, int[] shape) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(double[] data, int rows, int columns, int[] stride, int offset) {
        return null;
    }

    @Override
    public INDArray create(double[] data, int rows, int columns, int[] stride, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset) {
        return null;
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, int offset) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, int[] shape, int[] stride) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, int[] shape, int[] stride, int offset) {
        return null;
    }

    @Override
    public INDArray create(List<INDArray> list, int[] shape) {
        return null;
    }

    @Override
    public INDArray create(int rows, int columns, int[] stride, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(int[] shape, int[] stride, int offset) {
        return null;
    }

    @Override
    public INDArray create(int[] shape, int[] stride, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(int rows, int columns, int[] stride) {
        return null;
    }

    @Override
    public INDArray create(int rows, int columns, int[] stride) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(int[] shape, int[] stride) {
        return null;
    }

    @Override
    public INDArray create(int[] shape, int[] stride) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(int rows, int columns) {
        return null;
    }

    @Override
    public INDArray create(int rows, int columns) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(int[] shape) {
        return null;
    }

    @Override
    public INDArray create(int[] shape) {
        return null;
    }

    @Override
    public INDArray scalar(Number value, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray complexScalar(Number value, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray complexScalar(Number value) {
        return null;
    }

    @Override
    public INDArray scalar(float value, int offset) {
        return null;
    }

    @Override
    public INDArray scalar(double value, int offset) {
        return null;
    }

    @Override
    public INDArray scalar(int value, int offset) {
        return null;
    }

    @Override
    public INDArray scalar(Number value) {
        return null;
    }

    @Override
    public INDArray scalar(float value) {
        return null;
    }

    @Override
    public INDArray scalar(double value) {
        return null;
    }

    @Override
    public IComplexNDArray scalar(IComplexNumber value, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray scalar(IComplexFloat value) {
        return null;
    }

    @Override
    public IComplexNDArray scalar(IComplexDouble value) {
        return null;
    }

    @Override
    public IComplexNDArray scalar(IComplexNumber value) {
        return null;
    }

    @Override
    public IComplexNDArray scalar(IComplexFloat value, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray scalar(IComplexDouble value, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, int offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer buffer, int[] shape, int offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer buffer, int[] shape, int offset) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int[] shape, int offset) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int[] shape, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset, char ordering) {
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
    public IComplexNDArray createComplex(float[] dim) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, int offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray complexFlatten(List<IComplexNDArray> flatten) {
        return null;
    }

    @Override
    public IComplexNDArray complexFlatten(IComplexNDArray[] flatten) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer buffer, int[] shape, int offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, int offset) {
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
    public INDArray create(DataBuffer data, int[] newShape, int[] newStride, int offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, int[] newDims, int[] newStrides, int offset, char ordering) {
        return null;
    }

    @Override
    public INDArray rand(int rows, int columns, double min, double max, Random rng) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, Character order) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int[] shape, int offset, Character order) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int rows, int columns, int[] stride, int offset, char ordering) {
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
    public INDArray create(double[] data, int[] shape, int offset) {
        return null;
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, int offset, char ordering) {
        return null;
    }

    @Override
    public INDArray rand(int[] shape, double min, double max, Random rng) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(int[] ints, int[] ints1, int[] stride, int offset) {
        return null;
    }

    @Override
    public INDArray create(int[] ints, int[] ints1, int[] stride, int offset) {
        return null;
    }

    @Override
    public INDArray create(int[] shape, int[] ints1, int[] stride, char order, int offset) {
        return null;
    }

    @Override
    public INDArray create(int rows, int columns, char ordering) {
        return null;
    }

    @Override
    public INDArray create(int[] shape, DataBuffer.Type dataType) {
        return null;
    }

    @Override
    public INDArray create(float[] data, char order) {
        return null;
    }

    @Override
    public INDArray create(float[] data, int[] shape, int[] stride, char order, int offset) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer buffer, int[] shape, int[] stride, char order, int offset) {
        return null;
    }

    @Override
    public INDArray create(double[] data, char order) {
        return null;
    }

    @Override
    public INDArray create(double[] data, int[] shape, int[] stride, char order, int offset) {
        return null;
    }

    @Override
    public INDArray create(int[] shape, int[] stride, int offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(int[] shape, int[] complexStrides, int offset, char ordering) {
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
}
