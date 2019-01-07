package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A version of {@link WeightInitXavierUniform} for sigmoid activation functions. U(-r,r) with r=4sqrt(6/(fanIn + fanOut))
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class WeightInitSigmoidUniform implements IWeightInit {



    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        double r = 4.0 * Math.sqrt(6.0 / (fanIn + fanOut));
        Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-r, r));
        return paramView.reshape(order, shape);
    }
}
