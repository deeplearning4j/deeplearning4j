package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Uniform U[-a,a] with a=3.0/((fanIn + fanOut)/2)
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class WeightInitVarScalingUniformFanAvg implements IWeightInit {

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        double scalingFanAvg = 3.0 / Math.sqrt((fanIn + fanOut) / 2);
        Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-scalingFanAvg, scalingFanAvg));
        return paramView.reshape(order, shape);
    }
}
