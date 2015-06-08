package org.nd4j.linalg.api.blas.impl;

import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Provides auxillary methods for
 * blas to databuffer interactions
 * @author Adam Gibson
 */
public abstract class BaseLevel {


    public float[] getFloatData(INDArray buf) {
        return BlasBufferUtil.getFloatData(buf);
    }

    public double[] getDoubleData(INDArray buf) {
        return BlasBufferUtil.getDoubleData(buf);
    }

    public float[] getFloatData(DataBuffer buf) {
        return BlasBufferUtil.getFloatData(buf);
    }

    public double[] getDoubleData(DataBuffer buf) {
        return BlasBufferUtil.getDoubleData(buf);
    }

}

