package org.nd4j.linalg.lossfunctions.impl;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.lossfunctions.ILossFunction;

/**
 * Created by Alex on 08/08/2016.
 */
public class MeanSquaredError implements ILossFunction {



    @Override
    public double computeScore(INDArray labels, INDArray output, INDArray mask, boolean average) {
        return 0;
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray output, INDArray mask) {
        return null;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray output, INDArray mask, boolean average) {
        return null;
    }
}
