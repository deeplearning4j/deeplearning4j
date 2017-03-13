package org.nd4j.linalg.api.blas.impl;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Audrey Loeffel
 */
public abstract class SparseBaseLevel {
    //TODO implementations
    public float[] getFloatData(INDArray buf){
        return null;
    }

    public double[] getDoubleData(INDArray buf){
        return null;
    }

    public float[] getFloatData(DataBuffer buf){
        return null;
    }

    public double[] getDoubleData(DataBuffer buf){
        return null;
    }


}
