package org.nd4j.linalg.api.ndarray;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

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
}
