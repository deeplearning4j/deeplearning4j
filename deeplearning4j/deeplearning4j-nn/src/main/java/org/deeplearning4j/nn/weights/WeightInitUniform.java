package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Uniform U[-a,a] with a=1/sqrt(fanIn). "Commonly used heuristic" as per Glorot and Bengio 2010
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class WeightInitUniform implements IWeightInit {


    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        double a = 1.0 / Math.sqrt(fanIn);
        Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-a, a));
        return paramView.reshape(order, shape);
    }
}
