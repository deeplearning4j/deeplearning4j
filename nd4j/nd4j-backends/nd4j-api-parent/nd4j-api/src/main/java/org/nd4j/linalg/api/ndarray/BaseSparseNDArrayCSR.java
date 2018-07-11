package org.nd4j.linalg.api.ndarray;

import com.google.common.primitives.Ints;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.*;
import org.nd4j.linalg.util.LongUtils;

import java.util.ArrayList;
import java.util.List;

import static org.nd4j.base.Preconditions.checkArgument;


/**
 * @author Audrey Loeffel
 */
public abstract class BaseSparseNDArrayCSR extends BaseSparseNDArray {
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
    public BaseSparseNDArrayCSR(double[] data, int[] columnsPointers, int[] pointerB, int[] pointerE, long[] shape) {
        checkArgument(data.length == columnsPointers.length);
        checkArgument(pointerB.length == pointerE.length);
        // TODO
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape));
        init(shape);
        int valuesSpace = (int) (data.length * THRESHOLD_MEMORY_ALLOCATION);
        this.values = Nd4j.getDataBufferFactory().createDouble(valuesSpace);
        this.values.setData(data);
        this.columnsPointers = Nd4j.getDataBufferFactory().createInt(valuesSpace);
        this.columnsPointers.setData(columnsPointers);
        this.length = columnsPointers.length;
        long pointersSpace = rows;
        this.pointerB = Nd4j.getDataBufferFactory().createInt(pointersSpace);
        this.pointerB.setData(pointerB);
        this.pointerE = Nd4j.getDataBufferFactory().createInt(pointersSpace);
        this.pointerE.setData(pointerE);


    }

    public BaseSparseNDArrayCSR(float[] data, int[] columnsPointers, int[] pointerB, int[] pointerE, long[] shape) {
        this(Nd4j.createBuffer(data), columnsPointers, pointerB, pointerE, shape);
    }

    public BaseSparseNDArrayCSR(DataBuffer data, int[] columnsPointers, int[] pointerB, int[] pointerE, long[] shape) {
        checkArgument(pointerB.length == pointerE.length);
        setShapeInformation(Nd4j.getShapeInfoProvider().createShapeInformation(shape));
        init(shape);
        this.values = data;
        this.columnsPointers = Nd4j.getDataBufferFactory().createInt(data.length());
        this.columnsPointers.setData(columnsPointers);
        this.length = columnsPointers.length;
        // The size of these pointers are constant
        long pointersSpace = rows;
        this.pointerB = Nd4j.getDataBufferFactory().createInt(pointersSpace);
        this.pointerB.setData(pointerB);
        this.pointerE = Nd4j.getDataBufferFactory().createInt(pointersSpace);
        this.pointerE.setData(pointerE);

    }

    public INDArray putScalar(int row, int col, double value) {

        checkArgument(row < rows && 0 <= rows);
        checkArgument(col < columns && 0 <= columns);

        int idx = pointerB.getInt(row);
        int idxNextRow = pointerE.getInt(row);

        while (columnsPointers.getInt(idx) < col && columnsPointers.getInt(idx) < idxNextRow) {
            idx++;
        }
        if (columnsPointers.getInt(idx) == col) {
            values.put(idx, value);
        } else {
            //Add a new entry in both buffers at a given position
            values = addAtPosition(values, length, idx, value);
            columnsPointers = addAtPosition(columnsPointers, length, idx, col);
            length++;

            // shift the indices of the next rows
            pointerE.put(row, pointerE.getInt(row) + 1);
            for (int i = row + 1; i < rows; i++) {
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
        if (indexes.length == 1 && indexes[0] instanceof NDArrayIndexAll || (indexes.length == 2 && (isRowVector()
                        && indexes[0] instanceof PointIndex && indexes[0].offset() == 0
                        && indexes[1] instanceof NDArrayIndexAll
                        || isColumnVector() && indexes[1] instanceof PointIndex && indexes[0].offset() == 0
                                        && indexes[0] instanceof NDArrayIndexAll)))
            return this;

        indexes = NDArrayIndex.resolve(shapeInfoDataBuffer(), indexes);
        ShapeOffsetResolution resolution = new ShapeOffsetResolution(this);
        resolution.exec(indexes);

        if (indexes.length < 1)
            throw new IllegalStateException("Invalid index found of zero length");

        // FIXME: LONG
        int[] shape = LongUtils.toInts(resolution.getShapes());
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
    public DataBuffer getVectorCoordinates() {
        return Nd4j.getDataBufferFactory().create(columnsPointers, 0, length());

    }

    public double[] getDoubleValues() {
        return values.getDoublesAt(0, (int) length);
    }

    public double[] getColumns() {
        return columnsPointers.getDoublesAt(0, (int) length);
    }

    public int[] getPointerBArray() {
        return pointerB.asInt();
    }

    public int[] getPointerEArray() {
        return pointerE.asInt();
    }

    public SparseFormat getFormat() {
        return format;
    }

    public DataBuffer getPointerB() {
        return Nd4j.getDataBufferFactory().create(pointerB, 0, rows());
    }

    public DataBuffer getPointerE() {
        return Nd4j.getDataBufferFactory().create(pointerE, 0, rows());
    }

    private DataBuffer addAtPosition(DataBuffer buf, long dataSize, int pos, double value) {

        DataBuffer buffer = (buf.length() == dataSize) ? reallocate(buf) : buf;
        double[] tail = buffer.getDoublesAt(pos, (int) dataSize - pos);

        buffer.put(pos, value);
        for (int i = 0; i < tail.length; i++) {
            buffer.put(i + pos + 1, tail[i]);
        }
        return buffer;
    }

    @Override
    public DataBuffer data() {
        return Nd4j.getDataBufferFactory().create(values, 0, length());
    }

    @Override
    public INDArray toDense() {
        // Dummy way - going to use the conversion routines in level2 (?)
        INDArray result = Nd4j.zeros(shape());

        int[] pointersB = pointerB.asInt();
        int[] pointersE = pointerE.asInt();

        for (int row = 0; row < rows(); row++) {
            for (int idx = pointersB[row]; idx < pointersE[row]; idx++) {
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

        long[] offsets = resolution.getOffsets();
        long[] shape = resolution.getShapes();


        List<Integer> accuColumns = new ArrayList<>();
        List<Integer> accuPointerB = new ArrayList<>();
        List<Integer> accuPointerE = new ArrayList<>();

        if (shape.length == 2) {

            if (resolution.getOffset() != 0) {
                offsets[0] = (int) resolution.getOffset() / shape()[1];
                offsets[1] = (int) resolution.getOffset() % shape()[1];
            }
            long firstRow = offsets[0];
            long lastRow = firstRow + shape[0];
            long firstElement = offsets[1];
            long lastElement = firstElement + shape[1];

            int count = 0;
            int i = 0;
            for (int rowIdx = 0; rowIdx < lastRow; rowIdx++) {

                boolean isFirstInRow = true;
                for (int idx = pointerB.getInt(rowIdx); idx < pointerE.getInt(rowIdx); idx++) {

                    int colIdx = columnsPointers.getInt(count);

                    // add the element in the subarray it it belongs to the view
                    if (colIdx >= firstElement && colIdx < lastElement && rowIdx >= firstRow && rowIdx < lastRow) {

                        // add the new column pointer for this element
                        accuColumns.add((int) (colIdx - firstElement));

                        if (isFirstInRow) {
                            // Add the index of the first element of the row in the pointer array
                            accuPointerB.add(idx);
                            accuPointerE.add(idx + 1);
                            isFirstInRow = false;
                        } else {
                            // update the last element pointer array
                            accuPointerE.set((int) (rowIdx - firstRow), idx + 1);
                        }
                    }
                    count++;
                }

                // If the row doesn't contain any element and is included in the selected rows
                if (isFirstInRow && rowIdx >= firstRow && rowIdx < lastRow) {
                    int lastIdx = i == 0 ? 0 : accuPointerE.get(i - 1);
                    accuPointerB.add(lastIdx);
                    accuPointerE.add(lastIdx);
                }
                if (rowIdx >= firstRow && rowIdx <= lastRow) {
                    i++;
                }
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
    public INDArray subArray(long[] offsets, int[] shape, int[] stride) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean equals(Object o) {
        //TODO use op
        // fixme
        if (o == null || !(o instanceof INDArray)) {
            return false;
        }
        INDArray n = (INDArray) o;
        if (n.isSparse()) {
            BaseSparseNDArray s = (BaseSparseNDArray) n;
            switch (s.getFormat()) {
                case CSR:
                    BaseSparseNDArrayCSR csrArray = (BaseSparseNDArrayCSR) s;
                    if (csrArray.rows() == rows() && csrArray.columns() == columns()
                                    && csrArray.getVectorCoordinates().equals(getVectorCoordinates())
                                    && csrArray.data().equals(data()) && csrArray.getPointerB().equals(getPointerB())
                                    && csrArray.getPointerE().equals(getPointerE())) {
                        return true;
                    }
                    break;
                default:
                    INDArray dense = toDense();
                    INDArray oDense = s.toDense();
                    return dense.equals(oDense);
            }
        } else {
            INDArray dense = toDense();
            return dense.equals(o);
        }
        return false;
    }

    @Override
    public boolean isView() {
        return false; //todo
    }

    @Override
    public int underlyingRank() {
        return rank;
    }

    @Override
    public INDArray putiColumnVector(INDArray columnVector) {
        return null;
    }

    @Override
    public INDArray putiRowVector(INDArray rowVector) {
        return null;
    }
}
