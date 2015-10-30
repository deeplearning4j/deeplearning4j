package org.nd4j.linalg.api.shape.tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

public class TensorCalculatorFactory {

    /** Tensor calculator: 1d */
    public static TensorCalculator getTensorCalculator(INDArray array, int tensorDim){
        return getTensorCalculator(array.offset(), array.shape(), array.stride(), tensorDim);
    }

    public static TensorCalculator getTensorCalculator(int baseOffset, int[] shape, int[] stride, int tensorDim ){
        return new TensorCalculator1d(baseOffset, shape, stride, tensorDim);
    }

}
