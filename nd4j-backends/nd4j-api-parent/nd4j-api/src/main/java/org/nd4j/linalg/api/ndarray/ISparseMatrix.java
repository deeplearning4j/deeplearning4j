package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Audrey Loeffel
 */
public abstract class ISparseMatrix /*implements INDArray*/ {

    private static final double THRESHOLD_MEMORY_ALLOCATION = 0.5;
    protected static final Logger log = LoggerFactory.getLogger(BaseNDArray.class);

    protected transient volatile DoubleBuffer values;
    protected transient volatile IntBuffer columns;
    protected transient volatile IntBuffer pointerB;
    protected transient volatile IntBuffer pointerE;
    protected transient volatile long nnz = -1;
    protected transient volatile long bufferValuesLength = -1;
    protected int nbRows, nbColumns;
    protected Boolean isVector = null;
    protected Boolean isMatrix = null;
    protected Boolean isScalar = null;

/**
 *
 *
 * The length of the values and columns arrays is equal to the number of non-zero elements in A.
 * The length of the pointerB and pointerE arrays is equal to the number of rows in A.
 * @param values a double array that contains the non-zero element of the sparse matrix A
 * @param columns Element i of the integer array columns is the number of the column in A that contains the i-th value
 *                in the values array.
 * @param pointerB Element j of this integer array gives the index of the element in the values array that is first
 *                 non-zero element in a row j of A. Note that this index is equal to pointerB(j) - pointerB(1)+1 .
 * @param pointerE An integer array that contains row indices, such that pointerE(j)-pointerB(1) is the index of the
 *                 element in the values array that is last non-zero element in a row j of A.
 * @param shape Shape of the matrix A
 * */
    public ISparseMatrix(double[] values, int[] columns, int[] pointerB, int[] pointerE, int[] shape) {

        assert(values.length == columns.length);
        assert(pointerB.length == pointerE.length);

        if (shape.length == 2) {
            nbRows = shape[0];
            nbColumns = shape[1];
        } else if (shape.length == 1) {
            nbRows = 1;
            nbColumns = shape[0];
        } else {
            // ???
        }

        int valuesSpace = (int) (values.length * THRESHOLD_MEMORY_ALLOCATION) + values.length;
        this.values = new DoubleBuffer(valuesSpace);
        this.values.setData(values);
        this.columns = new IntBuffer(valuesSpace);
        this.columns.setData(columns);
        bufferValuesLength = valuesSpace;
        nnz = values.length;

        // The size of these pointers are constant
        int pointersSpace = nbRows;
        this.pointerB = new IntBuffer(pointersSpace);
        this.pointerB.setData(pointerB);
        this.pointerE = new IntBuffer(pointersSpace);
        this.pointerE.setData(pointerE);
    }

    public ISparseMatrix(){}

    public ISparseMatrix putScalar(int row, int col, double value){
        // TODO use shape information to get the corresponding index ?

        return this;
    }



/*
* TODO Should return a view of the current matrix
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

    private void addInBuffer(DataBuffer buffer, int value){

    }
}
