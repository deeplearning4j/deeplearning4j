package org.deeplearning4j.nn.weights;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Uniform U[-a,a] with a=3.0/(fanOut)
 *
 * @author Adam Gibson
 */
public class WeightInitVarScalingUniformFanOut implements IWeightInit {

    @Override
    public void init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        double scalingFanOut = 3.0 / Math.sqrt(fanOut);
        Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-scalingFanOut, scalingFanOut));
    }
}
