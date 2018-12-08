package org.deeplearning4j.nn.weights;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * : Sample weights from a provided distribution<br>
 *
 * @author Adam Gibson
 */
public class WeightInitDistribution implements IWeightInit {



    @Override
    public void init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        paramView.assign(value);
    }
}
