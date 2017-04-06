package org.deeplearning4j.nn.layers.recurrent;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Layer normalization, but normalizing each record independently, following  suggestions
 * from https://github.com/deeplearning4j/deeplearning4j/pull/3195
 * This does not seem to work as well as whole minibatch layer normalization on a test (only tested
 * on one use case though, your experience may differ).
 * Created by fac2003 on 4/6/17.
 */
public class LayerNormalizationPerRecord implements LayerNormalization {
    @Override
    public void normalize(INDArray activations) {
        INDArray stdevVector = activations.std(1);
        activations.subiColumnVector(Nd4j.mean(activations, 1));
        if (!Double.isNaN(stdevVector.sum().getDouble(0))) {
            // don't divide by NaN. i.e., all zero activations.
            activations.diviColumnVector(stdevVector);
        }
    }
}
