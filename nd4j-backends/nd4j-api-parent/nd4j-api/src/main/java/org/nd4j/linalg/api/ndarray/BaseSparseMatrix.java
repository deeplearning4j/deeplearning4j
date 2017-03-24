package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Audrey Loeffel
 */
public class BaseSparseMatrix implements ISparseMatrix {

    protected static final double THRESHOLD_MEMORY_ALLOCATION = 0.5;
    protected static final Logger log = LoggerFactory.getLogger(ISparseMatrix.class);


    protected transient volatile long nnz = -1;
    protected int nbRows, nbColumns;
    protected Boolean isVector = null;
    protected Boolean isMatrix = null;
    protected Boolean isScalar = null;

    protected DataBuffer reallocate(DataBuffer buffer) {
        int newSize = (int) buffer.length() * 2; // should be bound to max(nnz, size*2)
        DataBuffer newBuffer = null;
        if (buffer instanceof DoubleBuffer){
            newBuffer = new DoubleBuffer(newSize);
            newBuffer.setData(buffer.asDouble());
        } else if (buffer instanceof IntBuffer) {
            newBuffer = new IntBuffer(newSize);
            newBuffer.setData(buffer.asInt());
        }
        return newBuffer;
    }

}
