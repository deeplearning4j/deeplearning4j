package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Initialize to zeroes
 *
 * @author Christian Skarby
 */
@EqualsAndHashCode
public class WeightInitConstant implements IWeightInit {

    private final double value;

    public WeightInitConstant() {
        this(0.0);
    }

    public WeightInitConstant(double value) {
        this.value = value;
    }


    @Override
    public INDArray init(long fanIn, long fanOut, long[] shape, char order, INDArray paramView) {
        paramView.assign(value);
        return paramView.reshape(order, shape);
    }
}
