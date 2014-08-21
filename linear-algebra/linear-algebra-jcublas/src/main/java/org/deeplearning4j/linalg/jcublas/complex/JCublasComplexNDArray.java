package org.deeplearning4j.linalg.jcublas.complex;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.api.ndarray.SliceOp;
import org.deeplearning4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.linalg.ops.reduceops.Ops;

/**
 * Created by mjk on 8/20/14.
 */
public class JCublasComplexNDArray implements IComplexNDArray {

    @Override
    public IComplexNDArray cumsumi(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray cumsum(int dimension) {
        return null;
    }

    @Override
    public INDArray assign(INDArray arr) {
        return null;
    }

    @Override
    public int vectorsAlongDimension(int dimension) {
        return 0;
    }

    @Override
    public IComplexNDArray vectorAlongDimension(int index, int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray assign(IComplexNDArray arr) {
        return null;
    }

    @Override
    public IComplexNDArray put(NDArrayIndex[] indices, IComplexNumber element) {
        return null;
    }

    @Override
    public IComplexNDArray put(NDArrayIndex[] indices, IComplexNDArray element) {
        return null;
    }

    @Override
    public IComplexNDArray put(NDArrayIndex[] indices, Number element) {
        return null;
    }

    @Override
    public IComplexNDArray putScalar(int i, IComplexNumber value) {
        return null;
    }

    @Override
    public IComplexNDArray putScalar(int i, Number value) {
        return null;
    }

    @Override
    public INDArray putScalar(int[] i, Number value) {
        return null;
    }

    @Override
    public IComplexNDArray lt(Number other) {
        return null;
    }

    @Override
    public IComplexNDArray lti(Number other) {
        return null;
    }

    @Override
    public IComplexNDArray eq(Number other) {
        return null;
    }

    @Override
    public IComplexNDArray eqi(Number other) {
        return null;
    }

    @Override
    public IComplexNDArray gt(Number other) {
        return null;
    }

    @Override
    public IComplexNDArray gti(Number other) {
        return null;
    }

    @Override
    public IComplexNDArray lt(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray lti(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray eq(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray eqi(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray gt(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray gti(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray neg() {
        return null;
    }

    @Override
    public IComplexNDArray negi() {
        return null;
    }

    @Override
    public IComplexNDArray rdiv(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray rdivi(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray rsub(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray rsubi(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray div(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray divi(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray mul(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray muli(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray sub(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray subi(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray add(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray addi(Number n) {
        return null;
    }

    @Override
    public IComplexNDArray get(NDArrayIndex... indexes) {
        return null;
    }

    @Override
    public IComplexNDArray getColumns(int[] columns) {
        return null;
    }

    @Override
    public IComplexNDArray getRows(int[] rows) {
        return null;
    }


    @Override
    public IComplexNDArray min(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray max(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray put(int i, int j, INDArray element) {
        return null;
    }

    @Override
    public INDArray put(int i, int j, Number element) {
        return null;
    }

    @Override
    public IComplexNDArray put(int[] indices, INDArray element) {
        return null;
    }

    @Override
    public IComplexNDArray putSlice(int slice, INDArray put) {
        return null;
    }

    @Override
    public void iterateOverDimension(int dimension, SliceOp op, boolean modify) {

    }

    @Override
    public IComplexNDArray reduce(Ops.DimensionOp op, int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray getScalar(int... indexes) {
        return null;
    }

    @Override
    public void checkDimensions(INDArray other) {

    }

    @Override
    public int[] endsForSlices() {
        return new int[0];
    }

    @Override
    public IComplexNDArray assign(Number value) {
        return null;
    }

    @Override
    public int linearIndex(int i) {
        return 0;
    }

    @Override
    public void iterateOverAllRows(SliceOp op) {

    }

    @Override
    public IComplexNDArray rdiv(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray rdivi(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray rdiv(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray rdivi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray rsub(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray rsub(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray rsubi(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray rsubi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray hermitian() {
        return null;
    }

    @Override
    public IComplexNDArray conj() {
        return null;
    }

    @Override
    public IComplexNDArray conji() {
        return null;
    }

    @Override
    public INDArray getReal() {
        return null;
    }

    @Override
    public IComplexNDArray repmat(int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray putRow(int row, INDArray toPut) {
        return null;
    }

    @Override
    public IComplexNDArray putColumn(int column, INDArray toPut) {
        return null;
    }

    @Override
    public IComplexNDArray getScalar(int row, int column) {
        return null;
    }

    @Override
    public IComplexNDArray getScalar(int i) {
        return null;
    }

    @Override
    public double squaredDistance(INDArray other) {
        return 0;
    }

    @Override
    public double distance2(INDArray other) {
        return 0;
    }

    @Override
    public double distance1(INDArray other) {
        return 0;
    }

    @Override
    public INDArray put(NDArrayIndex[] indices, INDArray element) {
        return null;
    }

    @Override
    public IComplexNDArray put(int i, INDArray element) {
        return null;
    }

    @Override
    public IComplexNDArray diviColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray divColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray diviRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray divRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray muliColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray mulColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray muliRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray mulRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray subiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray subColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray subiRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray subRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray addiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray addColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public IComplexNDArray addiRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray addRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public IComplexNDArray mmul(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray mmul(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray div(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray div(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray mul(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray mul(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray sub(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray sub(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray add(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray add(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray mmuli(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray mmuli(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray divi(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray divi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray muli(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray muli(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray subi(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray subi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray addi(INDArray other) {
        return null;
    }

    @Override
    public IComplexNDArray addi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray normmax(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray norm2(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray norm1(int dimension) {
        return null;
    }

    @Override
    public INDArray std(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray prod(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray mean(int dimension) {
        return null;
    }

    @Override
    public INDArray var(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray sum(int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray get(int[] indices) {
        return null;
    }

    @Override
    public IComplexNDArray dup() {
        return null;
    }

    @Override
    public IComplexNDArray ravel() {
        return null;
    }

    @Override
    public int slices() {
        return 0;
    }

    @Override
    public IComplexNDArray slice(int i, int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray slice(int i) {
        return null;
    }

    @Override
    public int offset() {
        return 0;
    }

    @Override
    public IComplexNDArray reshape(int[] newShape) {
        return null;
    }

    @Override
    public INDArray reshape(int rows, int columns) {
        return null;
    }

    @Override
    public IComplexNDArray transpose() {
        return null;
    }

    @Override
    public IComplexNDArray swapAxes(int dimension, int with) {
        return null;
    }

    @Override
    public IComplexNDArray permute(int[] rearrange) {
        return null;
    }

    @Override
    public IComplexNDArray getColumn(int i) {
        return null;
    }

    @Override
    public IComplexNDArray getRow(int i) {
        return null;
    }

    @Override
    public int columns() {
        return 0;
    }

    @Override
    public int rows() {
        return 0;
    }

    @Override
    public boolean isColumnVector() {
        return false;
    }

    @Override
    public boolean isRowVector() {
        return false;
    }

    @Override
    public boolean isVector() {
        return false;
    }

    @Override
    public boolean isMatrix() {
        return false;
    }

    @Override
    public boolean isScalar() {
        return false;
    }

    @Override
    public int[] shape() {
        return new int[0];
    }

    @Override
    public int[] stride() {
        return new int[0];
    }

    @Override
    public int size(int dimension) {
        return 0;
    }

    @Override
    public int length() {
        return 0;
    }

    @Override
    public IComplexNDArray broadcast(int[] shape) {
        return null;
    }

    @Override
    public IComplexNDArray broadcasti(int[] shape) {
        return null;
    }

    @Override
    public Object element() {
        return null;
    }

    @Override
    public double[] data() {
        return new double[0];
    }

    @Override
    public void setData(double[] data) {

    }

    @Override
    public float[] floatData() {
        return new float[0];
    }

    @Override
    public void setData(float[] data) {

    }
}
