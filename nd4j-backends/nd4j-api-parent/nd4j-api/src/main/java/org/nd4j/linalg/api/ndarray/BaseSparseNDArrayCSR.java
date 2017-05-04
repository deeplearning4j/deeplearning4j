package org.nd4j.linalg.api.ndarray;

import com.google.common.primitives.Ints;
import net.ericaro.neoitertools.Generator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.*;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * @author Audrey Loeffel
 */
public abstract class BaseSparseNDArrayCSR extends BaseSparseNDArray{
    protected static final SparseFormat format = SparseFormat.CSR;
    protected transient volatile DataBuffer values;
    protected transient volatile DataBuffer columnsPointers;
    protected transient volatile DataBuffer pointerB;
    protected transient volatile DataBuffer pointerE;



    /**
     *
     *
     * The length of the values and columns arrays is equal to the number of non-zero elements in A.
     * The length of the pointerB and pointerE arrays is equal to the number of rows in A.
     * @param data a double array that contains the non-zero element of the sparse matrix A
     * @param columnsPointers Element i of the integer array columns is the number of the column in A that contains the i-th value
     *                in the values array.
     * @param pointerB Element j of this integer array gives the index of the element in the values array that is first
     *                 non-zero element in a row j of A. Note that this index is equal to pointerB(j) - pointerB(1)+1 .
     * @param pointerE An integer array that contains row indices, such that pointerE(j)-pointerB(1) is the index of the
     *                 element in the values array that is last non-zero element in a row j of A.
     * @param shape Shape of the matrix A
     */
    public BaseSparseNDArrayCSR(double[] data, int[] columnsPointers, int[] pointerB, int[] pointerE, int[] shape) {

        checkArgument(data.length == columnsPointers.length);
        checkArgument(pointerB.length == pointerE.length);
        // TODO
        this.shapeInformation = Nd4j.getShapeInfoProvider().createShapeInformation(shape);
        init(shape);
        int valuesSpace = (int) (data.length * THRESHOLD_MEMORY_ALLOCATION) + data.length;
        this.values = Nd4j.getDataBufferFactory().createDouble(valuesSpace);
        this.values.setData(data);
        this.columnsPointers = Nd4j.getDataBufferFactory().createInt(valuesSpace);
        this.columnsPointers.setData(columnsPointers);
        nnz = columnsPointers.length;
        System.out.println("nnz set for data[]");
        // The size of these pointers are constant
        int pointersSpace = rows;
        this.pointerB = Nd4j.getDataBufferFactory().createInt(pointersSpace);
        this.pointerB.setData(pointerB);
        this.pointerE = Nd4j.getDataBufferFactory().createInt(pointersSpace);
        this.pointerE.setData(pointerE);


    }

    public BaseSparseNDArrayCSR(float[] data, int[] columnsPointers, int[] pointerB, int[] pointerE, int[] shape) {
        this(Nd4j.createBuffer(data), columnsPointers, pointerB, pointerE, shape);
    }

    public BaseSparseNDArrayCSR(DataBuffer data, int[] columnsPointers, int[] pointerB, int[] pointerE, int[] shape){
        checkArgument(pointerB.length == pointerE.length);
        this.shapeInformation = Nd4j.getShapeInfoProvider().createShapeInformation(shape);
        init(shape);
        this.values = data;
        this.columnsPointers = Nd4j.getDataBufferFactory().createInt(data.length());
        this.columnsPointers.setData(columnsPointers);
        this.nnz = columnsPointers.length;
        // The size of these pointers are constant
        int pointersSpace = rows;
        this.pointerB = Nd4j.getDataBufferFactory().createInt(pointersSpace);
        this.pointerB.setData(pointerB);
        this.pointerE = Nd4j.getDataBufferFactory().createInt(pointersSpace);
        this.pointerE.setData(pointerE);

        }

    protected void init(int[] shape) {

        if (shape.length == 1) {
            rows = 1;
            columns = shape[0];
        } else if (this.shape().length == 2) {
            rows = shape[0];
            columns = shape[1];
        }
        this.length = ArrayUtil.prodLong(shape);
        rank = shape.length;
    }

    public INDArray putScalar(int row, int col, double value){

        checkArgument(row < rows && 0 <= rows);
        checkArgument(col < columns && 0 <= columns);

        int idx = pointerB.getInt(row);
        int idxNextRow = pointerE.getInt(row);

        while(columnsPointers.getInt(idx) < col && columnsPointers.getInt(idx) < idxNextRow) {
            idx ++;
        }
        if (columnsPointers.getInt(idx) == col) {
            values.put(idx, value);
        } else {
            //Add a new entry in both buffers at a given position
            values = addAtPosition(values, nnz, idx, value);
            columnsPointers = addAtPosition(columnsPointers, nnz, idx, col);
            nnz ++;

            // shift the indices of the next rows
            pointerE.put(row, pointerE.getInt(row) + 1);
            for(int i = row + 1; i < rows; i ++){
                pointerB.put(i, pointerB.getInt(i) + 1);
                pointerE.put(i, pointerE.getInt(i) + 1);
            }
        }
        return this;
    }


    /**
     * Returns a subset of this array based on the specified
     * indexes
     *
     * @param indexes the indexes in to the array
     * @return a view of the array with the specified indices
     */
    @Override
    public INDArray get(INDArrayIndex... indexes) {
        //check for row/column vector and point index being 0
        if (indexes.length == 1 && indexes[0] instanceof NDArrayIndexAll
                || (indexes.length == 2 && (isRowVector()
                && indexes[0] instanceof PointIndex && indexes[0].offset() == 0
                && indexes[1] instanceof NDArrayIndexAll
                || isColumnVector()
                && indexes[1] instanceof PointIndex && indexes[0].offset() == 0
                && indexes[0] instanceof NDArrayIndexAll)))
            return this;

        indexes = NDArrayIndex.resolve(shapeInfoDataBuffer(), indexes);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(this);
        resolution.exec(indexes);

        if (indexes.length < 1)
            throw new IllegalStateException("Invalid index found of zero length");

        int[] shape = resolution.getShapes();
        int numSpecifiedIndex = 0;

        for (int i = 0; i < indexes.length; i++)
            if (indexes[i] instanceof SpecifiedIndex)
                numSpecifiedIndex++;


        if (shape != null && numSpecifiedIndex > 0) {
            // TODO create a new ndarray with the specified indexes
            return null;

        }

        INDArray ret = subArray(resolution);
        return ret;

    }

    /**
     * Return the minor pointers. (columns for CSR, rows for CSC,...)
     * */
    public DataBuffer getMinorPointer(){
       return Nd4j.getDataBufferFactory().create(columnsPointers, 0, length());
    }

    public double[] getDoubleValues(){
        return values.getDoublesAt(0, (int) nnz);
    }

    public double[] getColumns(){
        return columnsPointers.getDoublesAt(0, (int) nnz);
    }

    public int[] getPointerBArray(){
        return pointerB.asInt();
    }

    public int[] getPointerEArray(){
        return pointerE.asInt();
    }

    public SparseFormat getFormat(){
        return format;
    }

    private void add(DataBuffer buffer, int value){
        // TODO add value and the end of the array
    }

    public DataBuffer getPointerB() {
        return pointerB;
    }

    public DataBuffer getPointerE() {
        return pointerE;
    }

    private DataBuffer addAtPosition(DataBuffer buf, long dataSize, int pos, double value){

        DataBuffer buffer = (buf.length() == dataSize) ? reallocate(buf) : buf;
        double[] tail = buffer.getDoublesAt(pos, (int) dataSize - pos);

        buffer.put(pos, value);
        for(int i = 0; i < tail.length; i++) {
            buffer.put(i + pos + 1, tail[i]);
        }
        return buffer;
    }

    @Override
    public DataBuffer data(){return values;}

    @Override
    public int columns() {
        return columns;
    }

    @Override
    public int rows() {
        return rows;
    }

    /**
     * Checks whether the matrix is a vector.
     */
    @Override
    public boolean isVector() {
        return isRowVector() || isColumnVector();
    }

    @Override
    public boolean isSquare() {

        return isMatrix() && rows() == columns();
    }

    /**
     * Checks whether the matrix is a row vector.
     */
    @Override
    public boolean isRowVector() {
        return rank == 2 && rows == 1;
    }

    /**
     * Checks whether the matrix is a column vector.
     */
    @Override
    public boolean isColumnVector() {
        return rank == 2 && columns == 1;
    }


    @Override
    public boolean isMatrix() {
        if (isMatrix != null)
            return isMatrix;
        isMatrix = (rank == 2 && (size(0) != 1 && size(1) != 1));
        return isMatrix;
    }

    @Override
    public boolean isScalar() {
        return false;
        //todo
    }

    @Override
    public INDArray toDense() {
        // Dummy way - going to use the conversion routines in level2 (?)
        INDArray result = Nd4j.zeros(shape());

        int[] pointersB = pointerB.asInt();
        int[] pointersE = pointerE.asInt();

        for(int row = 0; row < rows(); row++){
            for(int idx = pointersB[row]; idx < pointersE[row]; idx++){
                result.put(row, columnsPointers.getInt(idx), values.getNumber(idx));
            }
        }
        return result;
    }

    @Override
    public DataBuffer shapeInfoDataBuffer() {
        return shapeInformation;
    }

    @Override
    public INDArray subArray(ShapeOffsetResolution resolution) {

        int[] offsets = resolution.getOffsets();
        int[] shape = resolution.getShapes();

        List<Integer> accuColumns = new ArrayList<>();
        List<Integer> accuPointerB = new ArrayList<>();
        List<Integer> accuPointerE = new ArrayList<>();

        if(shape.length == 2) {

            int firstRow = 0;
            int lastRow = 0;
            int firstElement = 0;
            int lastElement = 0;

            if(resolution.getOffset() != 0) {
                //System.out.println("resolution offset " + (int)resolution.getOffset() + " length row "+ shape[1]);
                firstRow = (int)resolution.getOffset() / shape()[1];
                lastRow = firstRow + shape[0];
                firstElement = (int)resolution.getOffset() % shape()[1];
                lastElement = firstElement + shape[1];
            } else {
                firstRow = offsets [0];
                lastRow = firstRow + shape[0];
                firstElement = offsets [1];
                lastElement = firstElement + shape[1];
            }

            System.out.println(firstRow + " to " + lastRow);

            for(int rowIdx = firstRow; rowIdx < lastRow; rowIdx++){
                //System.out.println("Row : " + rowIdx);
                boolean isFirstInRow = true;
                for(int idx = pointerB.getInt(rowIdx); idx < pointerE.getInt(rowIdx); idx++){
                    //System.out.println("Idx: " + idx);
                    int colIdx = columnsPointers.getInt(idx);

                    // add the element in the subarray it it belongs to the view
                    if(colIdx >= firstElement && colIdx < lastElement){

                        // add the new column pointer for this element
                        //System.out.println("row " + rowIdx + " idx " + idx + "colidx " + colIdx);
                        //System.out.println("value " + values.getNumber(idx) +" - add " + colIdx + " " + offsets[1]);
                        accuColumns.add(colIdx - offsets[1]);

                        if(isFirstInRow){
                            // Add the index of the first element of the row in the pointer array
                            accuPointerB.add(idx);
                            accuPointerE.add(idx+1);
                            isFirstInRow = false;
                        } else {
                            // update the last element pointer array
                            accuPointerE.set(rowIdx - firstRow,idx + 1);
                        }
                    }
                    if(colIdx > lastElement){
                        break;
                    }
                }

                // If the row doesn't contain any element
                if(isFirstInRow){
                    int lastIdx = rowIdx == 0 ? 0 : accuPointerE.get(rowIdx-1);
                    accuPointerB.add(lastIdx);
                    accuPointerE.add(lastIdx);
                }
                isFirstInRow = true;
            }

            int[] newColumns = Ints.toArray(accuColumns);
            int[] newPointerB = Ints.toArray(accuPointerB);
            int[] newPointerE = Ints.toArray(accuPointerE);

            INDArray subarray = Nd4j.createSparseCSR(values, newColumns, newPointerB, newPointerE, shape);

            return subarray;

        } else {
            throw new UnsupportedOperationException();
        }
    }

    @Override
    public INDArray subArray(int[] offsets, int[] shape, int[] stride) {
        return null;
    }
}
