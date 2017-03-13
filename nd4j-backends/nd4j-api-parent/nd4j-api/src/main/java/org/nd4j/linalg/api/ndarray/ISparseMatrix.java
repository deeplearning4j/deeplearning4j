package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by audrey on 3/2/17.
 */
public abstract class ISparseMatrix /*implements INDArray*/ {
    protected static final Logger log = LoggerFactory.getLogger(BaseNDArray.class);

    protected transient volatile DataBuffer cooValA;
    protected transient volatile IntBuffer cooRowIndA;
    protected transient volatile IntBuffer cooColIndA;
    protected transient volatile long nnz;
    protected int rows, columns;
    protected long length = -1;

    public ISparseMatrix(){}

    public ISparseMatrix(DataBuffer cooValA, IntBuffer cooRowIndA, IntBuffer cooColIndA, int[] shape) {
        this.cooValA = cooValA;
        this.cooColIndA = cooColIndA;
        this.cooRowIndA = cooRowIndA;
        this.nnz = cooValA.length();
        if(shape.length == 2) {
            rows = shape[0];
            columns = shape[1];
        } else if(shape.length == 1) {
            rows = 1;
            columns = shape[0];
        } else {
            // ???
        }
    }

    public ISparseMatrix /*INDArray*/ putScalar(int row, int col, double value) {
        // add row, col , value to the databuffers
        return this;
    }

/*
* Should return a view of the current matrix
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
}
