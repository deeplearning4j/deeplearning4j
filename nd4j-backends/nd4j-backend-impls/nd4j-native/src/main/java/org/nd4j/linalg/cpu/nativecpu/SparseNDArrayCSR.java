package org.nd4j.linalg.cpu.nativecpu;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.ISparseNDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * @author Audrey Loeffel
 */
@Slf4j
public class SparseNDArrayCSR extends BaseSparseNDArray {
    protected static final SparseFormat format = SparseFormat.CSR;
    protected transient volatile DoubleBuffer values;
    protected transient volatile DoubleBuffer columns;
    protected transient volatile IntBuffer pointerB;
    protected transient volatile IntBuffer pointerE;


/**
 *
 *
 * The length of the values and columns arrays is equal to the number of non-zero elements in A.
 * The length of the pointerB and pointerE arrays is equal to the number of rows in A.
 * @param data a double array that contains the non-zero element of the sparse matrix A
 * @param columns Element i of the integer array columns is the number of the column in A that contains the i-th value
 *                in the values array.
 * @param pointerB Element j of this integer array gives the index of the element in the values array that is first
 *                 non-zero element in a row j of A. Note that this index is equal to pointerB(j) - pointerB(1)+1 .
 * @param pointerE An integer array that contains row indices, such that pointerE(j)-pointerB(1) is the index of the
 *                 element in the values array that is last non-zero element in a row j of A.
 * @param shape Shape of the matrix A
*/
    public SparseNDArrayCSR(double[] data, int[] columns, int[] pointerB, int[] pointerE, int[] shape) {

        checkArgument(data.length == columns.length);
        checkArgument(pointerB.length == pointerE.length);

        if (shape.length == 2) {
            nbRows = shape[0];
            nbColumns = shape[1];
        } else if (shape.length == 1) {
            nbRows = 1;
            nbColumns = shape[0];
        } else {
            // ???
        }

        int valuesSpace = (int) (data.length * THRESHOLD_MEMORY_ALLOCATION) + data.length;
        this.values = new DoubleBuffer(valuesSpace);
        this.values.setData(data);
        this.columns = new DoubleBuffer(valuesSpace);
        this.columns.setData(columns);
        //bufferValuesLength = valuesSpace;
        nnz = data.length;

        // The size of these pointers are constant
        int pointersSpace = nbRows;
        this.pointerB = new IntBuffer(pointersSpace);
        this.pointerB.setData(pointerB);
        this.pointerE = new IntBuffer(pointersSpace);
        this.pointerE.setData(pointerE);
    }

    public ISparseNDArray putScalar(int row, int col, double value){
        // TODO use shape information to get the corresponding index ?

        checkArgument(row < nbRows);
        checkArgument(col < nbColumns);

        int idx = pointerB.getInt(row);
        int idxNextRow = pointerB.getInt(row + 1);

        while(columns.getInt(idx) < col && columns.getInt(idx) < idxNextRow) {
            idx ++;
        }
        if (columns.getInt(idx) == col) {
            values.put(idx, value);
        } else {
            //Add a new entry in both buffers at a given position
            values = addAtPosition(values, nnz, idx, value);
            columns = addAtPosition(columns, nnz, idx, col);
            nnz ++;

            // shift the indices of the next rows
            pointerE.put(row, pointerE.getInt(row) + 1);
            for(int i = row + 1; i < nbRows; i ++){
                pointerB.put(i, pointerB.getInt(i) + 1);
                pointerE.put(i, pointerE.getInt(i) + 1);
            }
        }
        return this;
    }

    /*
    * TODO Should take an index in parameter and should return a view of the current matrix
    * */
    public INDArray get(int r, int c) {
//        DataBuffer ret = new FloatBuffer()
//        for(int i = 0; i < length; i++){
//            if(cooColIndA.get(i) == c && cooRowIndA.get(i) == r) {
//                return cooValA.get(i);
//cooColIndA.getFloat(i);
//            }
//        }
//        return 0;
        return null;
    }

    public double[] getDoubleValues(){
        return values.getDoublesAt(0, (int) nnz);
    }

    public double[] getColumns(){
        return columns.getDoublesAt(0, (int) nnz);
    }

    public int[] getPointersB(){
        return pointerB.asInt();
    }
    public int[] getPointersE(){
        return pointerE.asInt();
    }
    public SparseFormat getFormat(){
        return format;
    }

    private void add(DataBuffer buffer, int value){
        // TODO add value and the end of the array
    }

    private DoubleBuffer addAtPosition(DoubleBuffer buf, long dataSize, int pos, double value){
        // TODO add at the given position and shift the tail
        DataBuffer buffer = (buf.length() == dataSize) ? reallocate(buf) : buf;
        double[] tail = buffer.getDoublesAt(pos, (int) dataSize - pos);

        buffer.put(pos, value);
        for(int i = 0; i < tail.length; i++) {
            buffer.put(i + pos + 1, tail[i]);
        }
        return (DoubleBuffer) buffer;
    }
}
