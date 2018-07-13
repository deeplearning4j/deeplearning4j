package org.nd4j.linalg.cpu.nativecpu;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
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
    public INDArray createUninitialized(int[] shape, char ordering, DataBuffer.Type dType) {
        return null;
    }

    @Override
    public INDArray createUninitialized(long[] shape, char ordering, DataBuffer.Type dType) {
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
    public IComplexFloat createFloat(float real, float imag) {
        return null;
    }

    @Override
    public IComplexDouble createDouble(double real, double imag) {
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
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, long offset) {
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

    @Override
    public IComplexNDArray createComplex(double[] data, int[] shape, int[] stride, long offset, char ordering) {
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
    public IComplexNDArray createComplex(DataBuffer data) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, long rows, long columns, int[] stride, long offset) {
        return null;
    }

    @Override
    public INDArray create(DataBuffer data, long rows, long columns, int[] stride, long offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer data, int[] shape, int[] stride, long offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, long offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, long offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, int[] stride, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, long offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(IComplexNumber[] data, int[] shape, char ordering) {
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
    public INDArray create(DataBuffer data, int[] shape, int[] stride, long offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer buffer, int[] shape, long offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(DataBuffer buffer, int[] shape, long offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, long offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, long offset) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, int[] shape, int[] stride, long offset, char ordering) {
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
    public IComplexNDArray createComplex(DataBuffer data, int[] newDims, int[] newStrides, long offset, char ordering) {
        return null;
    }

    @Override
    public IComplexNDArray createComplex(float[] data, Character order) {
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
        if (x.isScalar())
            return x;

        if (x.data().dataType() == DataBuffer.Type.FLOAT) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().sortFloat(null, (FloatPointer) x.data().addressPointer(), (LongPointer) x.shapeInfoDataBuffer().addressPointer(), descending);
        } else if (x.data().dataType() == DataBuffer.Type.DOUBLE) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().sortDouble(null, (DoublePointer) x.data().addressPointer(), (LongPointer) x.shapeInfoDataBuffer().addressPointer(), descending);
        } else {
            throw new UnsupportedOperationException("Unknown dataype " + x.data().dataType());
        }
        return x;
    }

    @Override
    public INDArray sort(INDArray x, boolean descending, int... dimension) {
        if (x.isScalar())
            return x;

        Arrays.sort(dimension);
        Pair<DataBuffer, DataBuffer> tadBuffers = Nd4j.getExecutioner().getTADManager().getTADOnlyShapeInfo(x, dimension);

        if (x.data().dataType() == DataBuffer.Type.FLOAT) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().sortTadFloat(null,
                    (FloatPointer) x.data().addressPointer(),
                    (LongPointer) x.shapeInfoDataBuffer().addressPointer(),
                    new IntPointer(dimension),
                    dimension.length,
                    (LongPointer) tadBuffers.getFirst().addressPointer(),
                    new LongPointerWrapper(tadBuffers.getSecond().addressPointer()),
                    descending);
        } else if (x.data().dataType() == DataBuffer.Type.DOUBLE) {
            NativeOpsHolder.getInstance().getDeviceNativeOps().sortTadDouble(null,
                    (DoublePointer) x.data().addressPointer(),
                    (LongPointer) x.shapeInfoDataBuffer().addressPointer(),
                    new IntPointer(dimension),
                    dimension.length,
                    (LongPointer) tadBuffers.getFirst().addressPointer(),
                    new LongPointerWrapper(tadBuffers.getSecond().addressPointer()),
                    descending);
        } else {
            throw new UnsupportedOperationException("Unknown datatype " + x.data().dataType());
        }

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
        switch(val.dataType()){
            case FLOAT:
                NativeOpsHolder.getInstance().getDeviceNativeOps().sortCooIndicesFloat(null, (LongPointer) idx.addressPointer(), (FloatPointer) val.addressPointer(), length, rank);
                break;
            case DOUBLE:
                NativeOpsHolder.getInstance().getDeviceNativeOps().sortCooIndicesDouble(null, (LongPointer) idx.addressPointer(), (DoublePointer) val.addressPointer(), length, rank);
                break;
            default:
                throw new UnsupportedOperationException("Unknown datatype " + x.data().dataType());
        }

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
    public INDArray empty(DataBuffer.Type type) {
        throw new UnsupportedOperationException();
    }
}
