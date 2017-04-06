package org.deeplearning4j.nn.layers.recurrent;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
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
