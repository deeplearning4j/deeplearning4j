package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by audrey on 3/2/17.
 */
public abstract class ISparseMatrix /*implements INDArray*/ {
    private static final double ALLOCATION_RATION = 0.3;
    protected static final Logger log = LoggerFactory.getLogger(BaseNDArray.class);

    protected transient volatile DataBuffer values;
    protected transient volatile IntBuffer columns;
    protected transient volatile IntBuffer pointerB;
    protected transient volatile IntBuffer pointerE;

    protected transient volatile long nnz;
    protected int nbRows, nbColumns;
    protected long length = -1;
    protected long actualDataLength;

    public ISparseMatrix(double[] data, int[] columns, int[] pointerB, int[] pointerE, int nnz, int[] shape){
        this.nnz = nnz;
        if(shape.length == 2) {
            nbRows = shape[0];
            nbColumns = shape[1];
        } else if(shape.length == 1) {
            nbRows = 1;
            nbColumns = shape[0];
        } else {
            // ???
        }
        actualDataLength = data.length;

        int freeSpace =(int)(data.length * ALLOCATION_RATION);
        values = new DoubleBuffer(data.length + freeSpace);
        values.setData(data);

    }

    public ISparseMatrix(){}




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
}
