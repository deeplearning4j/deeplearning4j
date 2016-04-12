package org.nd4j.linalg.api.shape.tensor;

public interface TensorCalculator {

    int getNumTensors();

    int getOffsetForTensor(int tensorIdx);

    int[] getShape();

    int[] getStride();

    int getBaseOffset();

    int getTensorLength();

    int getElementWiseStrideForTensor();

}
