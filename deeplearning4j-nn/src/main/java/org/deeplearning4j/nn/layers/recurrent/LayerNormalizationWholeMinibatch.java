package org.deeplearning4j.nn.layers.recurrent;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by fac2003 on 4/6/17.
 */
public class LayerNormalizationWholeMinibatch implements LayerNormalization {
    @Override
    public void normalize(INDArray activations) {
        // normalize across entires mini-batch:
        Number stdev = activations.stdNumber();
        activations.subi(Nd4j.mean(activations));
        final double stdNumber = stdev.doubleValue();
        if (!Double.isNaN(stdNumber)) {
            // don't divide by NaN. i.e., all zero activations.
            activations.divi(stdev);
        }
    }
}
