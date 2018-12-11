package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Xavier weight init in DL4J up to 0.6.0. XAVIER should be preferred.
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class WeightInitXavierLegacy implements IWeightInit {

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        Nd4j.randn(paramView).divi(FastMath.sqrt(shape[0] + shape[1]));
        return paramView.reshape(order, shape);
    }
}
