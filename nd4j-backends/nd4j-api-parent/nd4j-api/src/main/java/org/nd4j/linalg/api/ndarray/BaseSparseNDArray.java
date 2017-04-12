package org.nd4j.linalg.api.ndarray;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.ShapeOffsetResolution;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.nio.IntBuffer;
import java.util.List;

/**
 * @author Audrey Loeffel
 */
@Slf4j
public abstract class BaseSparseNDArray implements ISparseNDArray {

    protected static final double THRESHOLD_MEMORY_ALLOCATION = 0.5;
    protected transient volatile long nnz = -1;
    protected int nbRows, nbColumns;
    protected Boolean isVector = null;
    protected Boolean isMatrix = null;
    protected Boolean isScalar = null;
    public static final boolean isSparse = true;

    protected DataBuffer reallocate(DataBuffer buffer) {
        int newSize = (int) buffer.length() * 2; // should be bound to max(nnz, size*2)
        DataBuffer newBuffer = Nd4j.createBuffer(newSize);

        switch(buffer.dataType()){
            case INT:
                newBuffer.setData(buffer.asInt());
                break;
            case DOUBLE:
                newBuffer.setData(buffer.asDouble());
                break;
            case FLOAT:
                newBuffer.setData(buffer.asFloat());
                break;
            case HALF:
                // ??
                break;
            case COMPRESSED:
                // ??
                break;
            default:
                break;
        }
        return newBuffer;
    }
    //@Override
    public boolean isSparse(){
        return isSparse;
    }

    //@Override
    public int length(){
        return (int) nnz;
    }
    public long lengthLong(){
        return nnz;
    }

    public abstract SparseFormat getFormat();

    // Override methods from INDArray


    @Override
    public String shapeInfoToString() {
        return null;
    }

    @Override
    public DataBuffer shapeInfoDataBuffer() {
        return null;
    }

    @Override
    public IntBuffer shapeInfo() {
        return null;
    }

    @Override
    public boolean isView() {
        return false;
    }

    @Override
    public boolean isCompressed() {
        return false;
    }

    @Override
    public void markAsCompressed(boolean reallyCompressed) {

    }

    @Override
    public void setWrapAround(boolean wrapAround) {

    }

    @Override
    public boolean isWrapAround() {
        return false;
    }

    @Override
    public int rank() {
        return 0;
    }

    @Override
    public int stride(int dimension) {
        return 0;
    }

    @Override
    public int elementStride() {
        return 0;
    }

    @Override
    public int elementWiseStride() {
        return 0;
    }

    @Override
    public boolean isCleanedUp() {
        return false;
    }

    @Override
    public void cleanup() {

    }

    @Override
    public void resetLinearView() {

    }

    @Override
    public int secondaryStride() {
        return 0;
    }

    @Override
    public double getDoubleUnsafe(int offset) {
        return 0;
    }

    @Override
    public INDArray putScalarUnsafe(int offset, double value) {
        return null;
    }

    @Override
    public int majorStride() {
        return 0;
    }

    @Override
    public int innerMostStride() {
        return 0;
    }

    @Override
    public INDArray linearView() {
        return null;
    }

    @Override
    public INDArray linearViewColumnOrder() {
        return null;
    }

    @Override
    public int vectorsAlongDimension(int dimension) {
        return 0;
    }

    @Override
    public INDArray vectorAlongDimension(int index, int dimension) {
        return null;
    }

    @Override
    public int tensorssAlongDimension(int... dimension) {
        return 0;
    }

    @Override
    public INDArray tensorAlongDimension(int index, int... dimension) {
        return null;
    }

    @Override
    public INDArray javaTensorAlongDimension(int index, int... dimension) {
        return null;
    }

    @Override
    public INDArray cumsumi(int dimension) {
        return null;
    }

    @Override
    public INDArray cumsum(int dimension) {
        return null;
    }

    @Override
    public INDArray assign(INDArray arr) {
        return null;
    }

    @Override
    public INDArray assignIf(INDArray arr, Condition condition) {
        return null;
    }

    @Override
    public INDArray replaceWhere(INDArray arr, Condition condition) {
        return null;
    }

    @Override
    public INDArray putScalar(int i, double value) {
        return null;
    }

    @Override
    public INDArray putScalar(int i, float value) {
        return null;
    }

    @Override
    public INDArray putScalar(int i, int value) {
        return null;
    }

    @Override
    public INDArray putScalar(int[] i, double value) {
        return null;
    }

    @Override
    public INDArray putScalar(int row, int col, double value) {
        return null;
    }

    @Override
    public INDArray putScalar(int dim0, int dim1, int dim2, double value) {
        return null;
    }

    @Override
    public INDArray putScalar(int dim0, int dim1, int dim2, int dim3, double value) {
        return null;
    }

    @Override
    public INDArray lt(Number other) {
        return null;
    }

    @Override
    public INDArray lti(Number other) {
        return null;
    }

    @Override
    public INDArray putScalar(int[] indexes, float value) {
        return null;
    }

    @Override
    public INDArray putScalar(int[] indexes, int value) {
        return null;
    }

    @Override
    public INDArray eps(Number other) {
        return null;
    }

    @Override
    public INDArray epsi(Number other) {
        return null;
    }

    @Override
    public INDArray eq(Number other) {
        return null;
    }

    @Override
    public INDArray eqi(Number other) {
        return null;
    }

    @Override
    public INDArray gt(Number other) {
        return null;
    }

    @Override
    public INDArray gte(Number other) {
        return null;
    }

    @Override
    public INDArray lte(Number other) {
        return null;
    }

    @Override
    public INDArray gtei(Number other) {
        return null;
    }

    @Override
    public INDArray ltei(Number other) {
        return null;
    }

    @Override
    public INDArray gti(Number other) {
        return null;
    }

    @Override
    public INDArray lt(INDArray other) {
        return null;
    }

    @Override
    public INDArray lti(INDArray other) {
        return null;
    }

    @Override
    public INDArray eps(INDArray other) {
        return null;
    }

    @Override
    public INDArray epsi(INDArray other) {
        return null;
    }

    @Override
    public INDArray neq(Number other) {
        return null;
    }

    @Override
    public INDArray neqi(Number other) {
        return null;
    }

    @Override
    public INDArray neq(INDArray other) {
        return null;
    }

    @Override
    public INDArray neqi(INDArray other) {
        return null;
    }

    @Override
    public INDArray eq(INDArray other) {
        return null;
    }

    @Override
    public INDArray eqi(INDArray other) {
        return null;
    }

    @Override
    public INDArray gt(INDArray other) {
        return null;
    }

    @Override
    public INDArray gti(INDArray other) {
        return null;
    }

    @Override
    public INDArray neg() {
        return null;
    }

    @Override
    public INDArray negi() {
        return null;
    }

    @Override
    public INDArray rdiv(Number n) {
        return null;
    }

    @Override
    public INDArray rdivi(Number n) {
        return null;
    }

    @Override
    public INDArray rsub(Number n) {
        return null;
    }

    @Override
    public INDArray rsubi(Number n) {
        return null;
    }

    @Override
    public INDArray div(Number n) {
        return null;
    }

    @Override
    public INDArray divi(Number n) {
        return null;
    }

    @Override
    public INDArray mul(Number n) {
        return null;
    }

    @Override
    public INDArray muli(Number n) {
        return null;
    }

    @Override
    public INDArray sub(Number n) {
        return null;
    }

    @Override
    public INDArray subi(Number n) {
        return null;
    }

    @Override
    public INDArray add(Number n) {
        return null;
    }

    @Override
    public INDArray addi(Number n) {
        return null;
    }

    @Override
    public INDArray rdiv(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray rdivi(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray rsub(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray rsubi(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray div(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray divi(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray mul(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray muli(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray sub(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray subi(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray add(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray addi(Number n, INDArray result) {
        return null;
    }

    @Override
    public INDArray get(INDArrayIndex... indexes) {
        return null;
    }

    @Override
    public INDArray getColumns(int... columns) {
        return null;
    }

    @Override
    public INDArray getRows(int... rows) {
        return null;
    }

    @Override
    public INDArray rdiv(INDArray other) {
        return null;
    }

    @Override
    public INDArray rdivi(INDArray other) {
        return null;
    }

    @Override
    public INDArray rdiv(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray rdivi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray rsub(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray rsub(INDArray other) {
        return null;
    }

    @Override
    public INDArray rsubi(INDArray other) {
        return null;
    }

    @Override
    public INDArray rsubi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray assign(Number value) {
        return null;
    }

    @Override
    public int linearIndex(int i) {
        return 0;
    }

    @Override
    public void checkDimensions(INDArray other) {

    }

    @Override
    public void sliceVectors(List<INDArray> list) {

    }

    @Override
    public INDArray putSlice(int slice, INDArray put) {
        return null;
    }

    @Override
    public INDArray cond(Condition condition) {
        return null;
    }

    @Override
    public INDArray condi(Condition condition) {
        return null;
    }

    @Override
    public INDArray repmat(int... shape) {
        return null;
    }

    @Override
    public INDArray repeat(int dimension, int... repeats) {
        return null;
    }

    @Override
    public INDArray putRow(int row, INDArray toPut) {
        return null;
    }

    @Override
    public INDArray putColumn(int column, INDArray toPut) {
        return null;
    }

    @Override
    public INDArray getScalar(int row, int column) {
        return null;
    }

    @Override
    public INDArray getScalar(int i) {
        return null;
    }

    @Override
    public int index(int row, int column) {
        return 0;
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
    public INDArray put(INDArrayIndex[] indices, INDArray element) {
        return null;
    }

    @Override
    public INDArray put(INDArrayIndex[] indices, Number element) {
        return null;
    }

    @Override
    public INDArray put(int[] indices, INDArray element) {
        return null;
    }

    @Override
    public INDArray put(int i, int j, INDArray element) {
        return null;
    }

    @Override
    public INDArray put(int i, int j, Number element) {
        return null;
    }

    @Override
    public INDArray put(int i, INDArray element) {
        return null;
    }

    @Override
    public INDArray diviColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray divColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray diviRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray divRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray rdiviColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray rdivColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray rdiviRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray rdivRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray muliColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray mulColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray muliRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray mulRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray rsubiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray rsubColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray rsubiRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray rsubRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray subiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray subColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray subiRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray subRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray addiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray addColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray addiRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray addRowVector(INDArray rowVector) {
        return null;
    }

    @Override
    public INDArray mmul(INDArray other) {
        return null;
    }

    @Override
    public INDArray mmul(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray div(INDArray other) {
        return null;
    }

    @Override
    public INDArray div(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray mul(INDArray other) {
        return null;
    }

    @Override
    public INDArray mul(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray sub(INDArray other) {
        return null;
    }

    @Override
    public INDArray sub(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray add(INDArray other) {
        return null;
    }

    @Override
    public INDArray add(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray mmuli(INDArray other) {
        return null;
    }

    @Override
    public INDArray mmuli(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray divi(INDArray other) {
        return null;
    }

    @Override
    public INDArray divi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray muli(INDArray other) {
        return null;
    }

    @Override
    public INDArray muli(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray subi(INDArray other) {
        return null;
    }

    @Override
    public INDArray subi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray addi(INDArray other) {
        return null;
    }

    @Override
    public INDArray addi(INDArray other, INDArray result) {
        return null;
    }

    @Override
    public INDArray normmax(int... dimension) {
        return null;
    }

    @Override
    public Number normmaxNumber() {
        return null;
    }

    @Override
    public IComplexNumber normmaxComplex() {
        return null;
    }

    @Override
    public INDArray norm2(int... dimension) {
        return null;
    }

    @Override
    public Number norm2Number() {
        return null;
    }

    @Override
    public IComplexNumber norm2Complex() {
        return null;
    }

    @Override
    public INDArray norm1(int... dimension) {
        return null;
    }

    @Override
    public Number norm1Number() {
        return null;
    }

    @Override
    public IComplexNumber norm1Complex() {
        return null;
    }

    @Override
    public INDArray std(int... dimension) {
        return null;
    }

    @Override
    public Number stdNumber() {
        return null;
    }

    @Override
    public INDArray std(boolean biasCorrected, int... dimension) {
        return null;
    }

    @Override
    public Number stdNumber(boolean biasCorrected) {
        return null;
    }

    @Override
    public IComplexNumber stdComplex() {
        return null;
    }

    @Override
    public INDArray prod(int... dimension) {
        return null;
    }

    @Override
    public Number prodNumber() {
        return null;
    }

    @Override
    public IComplexNumber prodComplex() {
        return null;
    }

    @Override
    public INDArray mean(int... dimension) {
        return null;
    }

    @Override
    public Number meanNumber() {
        return null;
    }

    @Override
    public IComplexNumber meanComplex() {
        return null;
    }

    @Override
    public INDArray var(int... dimension) {
        return null;
    }

    @Override
    public INDArray var(boolean biasCorrected, int... dimension) {
        return null;
    }

    @Override
    public Number varNumber() {
        return null;
    }

    @Override
    public IComplexNumber varComplex() {
        return null;
    }

    @Override
    public INDArray max(int... dimension) {
        return null;
    }

    @Override
    public Number maxNumber() {
        return null;
    }

    @Override
    public IComplexNumber maxComplex() {
        return null;
    }

    @Override
    public INDArray min(int... dimension) {
        return null;
    }

    @Override
    public Number minNumber() {
        return null;
    }

    @Override
    public IComplexNumber minComplex() {
        return null;
    }

    @Override
    public INDArray sum(int... dimension) {
        return null;
    }

    @Override
    public Number sumNumber() {
        return null;
    }

    @Override
    public IComplexNumber sumComplex() {
        return null;
    }

    @Override
    public void setStride(int... stride) {

    }

    @Override
    public void setShape(int... shape) {

    }

    @Override
    public void setOrder(char order) {

    }

    @Override
    public INDArray subArray(ShapeOffsetResolution resolution) {
        return null;
    }

    @Override
    public INDArray subArray(int[] offsets, int[] shape, int[] stride) {
        return null;
    }

    @Override
    public INDArray getScalar(int... indices) {
        return null;
    }

    @Override
    public int getInt(int... indices) {
        return 0;
    }

    @Override
    public double getDouble(int... indices) {
        return 0;
    }

    @Override
    public float getFloat(int[] indices) {
        return 0;
    }

    @Override
    public double getDouble(int i) {
        return 0;
    }

    @Override
    public double getDouble(int i, int j) {
        return 0;
    }

    @Override
    public float getFloat(int i) {
        return 0;
    }

    @Override
    public float getFloat(int i, int j) {
        return 0;
    }

    @Override
    public INDArray dup() {
        return null;
    }

    @Override
    public INDArray dup(char order) {
        return null;
    }

    @Override
    public INDArray ravel() {
        return null;
    }

    @Override
    public INDArray ravel(char order) {
        return null;
    }

    @Override
    public void setData(DataBuffer data) {

    }

    @Override
    public int slices() {
        return 0;
    }

    @Override
    public int getTrailingOnes() {
        return 0;
    }

    @Override
    public int getLeadingOnes() {
        return 0;
    }

    @Override
    public INDArray slice(int i, int dimension) {
        return null;
    }

    @Override
    public INDArray slice(int i) {
        return null;
    }

    @Override
    public int offset() {
        return 0;
    }

    @Override
    public int originalOffset() {
        return 0;
    }

    @Override
    public INDArray reshape(char order, int... newShape) {
        return null;
    }

    @Override
    public INDArray reshape(char order, int rows, int columns) {
        return null;
    }

    @Override
    public INDArray reshape(int... newShape) {
        return null;
    }

    @Override
    public INDArray reshape(int rows, int columns) {
        return null;
    }

    @Override
    public INDArray transpose() {
        return null;
    }

    @Override
    public INDArray transposei() {
        return null;
    }

    @Override
    public INDArray swapAxes(int dimension, int with) {
        return null;
    }

    @Override
    public INDArray permute(int... rearrange) {
        return null;
    }

    @Override
    public INDArray permutei(int... rearrange) {
        return null;
    }

    @Override
    public INDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable) {
        return null;
    }

    @Override
    public INDArray getColumn(int i) {
        return null;
    }

    @Override
    public INDArray getRow(int i) {
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
    public boolean isSquare() {
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
    public char ordering() {
        return 0;
    }

    @Override
    public int size(int dimension) {
        return 0;
    }

    @Override
    public INDArray broadcast(int... shape) {
        return null;
    }

    @Override
    public Object element() {
        return null;
    }

    @Override
    public DataBuffer data() {
        return null;
    }

    @Override
    public IComplexNDArray rdiv(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray rdivi(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray rsub(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray rsubi(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray div(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray divi(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray mul(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray muli(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray sub(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray subi(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray add(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray addi(IComplexNumber n) {
        return null;
    }

    @Override
    public IComplexNDArray rdiv(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray rdivi(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray rsub(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray rsubi(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray div(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray divi(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray mul(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray muli(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray sub(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray subi(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray add(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public IComplexNDArray addi(IComplexNumber n, IComplexNDArray result) {
        return null;
    }

    @Override
    public boolean equalsWithEps(Object o, double eps) {
        return false;
    }

    @Override
    public INDArray unsafeDuplication() {
        return null;
    }

    @Override
    public INDArray remainder(INDArray denominator) {
        return null;
    }

    @Override
    public INDArray remainder(INDArray denominator, INDArray result) {
        return null;
    }

    @Override
    public INDArray remainder(Number denominator) {
        return null;
    }

    @Override
    public INDArray remainder(Number denominator, INDArray result) {
        return null;
    }

    @Override
    public INDArray remainderi(INDArray denominator) {
        return null;
    }

    @Override
    public INDArray remainderi(Number denominator) {
        return null;
    }

    @Override
    public INDArray fmod(INDArray denominator) {
        return null;
    }

    @Override
    public INDArray fmod(INDArray denominator, INDArray result) {
        return null;
    }

    @Override
    public INDArray fmod(Number denominator) {
        return null;
    }

    @Override
    public INDArray fmod(Number denominator, INDArray result) {
        return null;
    }

    @Override
    public INDArray fmodi(INDArray denominator) {
        return null;
    }

    @Override
    public INDArray fmodi(Number denominator) {
        return null;
    }

    @Override
    public INDArray argMax(int... dimension) {
        return null;
    }

    @Override
    public boolean isAttached() {
        return false;
    }

    @Override
    public boolean isInScope() {
        return false;
    }

    @Override
    public INDArray detach() {
        return null;
    }

    @Override
    public INDArray leverage() {
        return null;
    }

    @Override
    public INDArray leverageTo(String id) {
        return null;
    }

    @Override
    public INDArray migrate() {
        return null;
    }
}
