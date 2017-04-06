package org.deeplearning4j.nn.layers.recurrent;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by fac2003 on 4/6/17.
 */
public class NoNormalization implements LayerNormalization {
    @Override
    public void normalize(INDArray activations) {
        // do nothing.
    }
}
