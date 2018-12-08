package org.deeplearning4j.nn.weights;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Gaussian distribution with mean 0, variance 1.0/(fanOut)
 *
 * @author Adam Gibson
 */
public class WeightInitVarScalingNormalFanOut implements IWeightInit {

    @Override
    public void init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        Nd4j.randn(paramView).divi(FastMath.sqrt(fanOut));
    }
}
