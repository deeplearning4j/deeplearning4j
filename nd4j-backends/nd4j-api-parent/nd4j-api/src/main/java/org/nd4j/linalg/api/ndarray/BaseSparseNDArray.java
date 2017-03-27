package org.nd4j.linalg.api.ndarray;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;

/**
 * @author Audrey Loeffel
 */
@Slf4j
public class BaseSparseNDArray implements ISparseNDArray {

    protected static final double THRESHOLD_MEMORY_ALLOCATION = 0.5;

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
        }else if (buffer instanceof FloatBuffer) {
            newBuffer = new FloatBuffer(newSize);
            newBuffer.setData(buffer.asFloat());
        }
        return newBuffer;
    }

}
