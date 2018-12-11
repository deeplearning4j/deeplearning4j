package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * He et al. (2015), "Delving Deep into Rectifiers". Uniform distribution U(-s,s) with s = sqrt(6/fanIn)
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class WeightInitReluUniform implements IWeightInit {



    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        double u = Math.sqrt(6.0 / fanIn);
        Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-u, u)); //U(-sqrt(6/fanIn), sqrt(6/fanIn)
        return paramView.reshape(order, shape);
    }
}
