package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * : He et al. (2015), "Delving Deep into Rectifiers". Normal distribution with variance 2.0/nIn
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class WeightInitRelu implements IWeightInit {

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        Nd4j.randn(paramView).muli(FastMath.sqrt(2.0 / fanIn)); //N(0, 2/nIn)
        return paramView.reshape(order, shape);
    }
}
