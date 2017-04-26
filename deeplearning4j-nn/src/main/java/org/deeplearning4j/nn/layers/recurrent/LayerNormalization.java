package org.deeplearning4j.nn.layers.recurrent;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by fac2003 on 4/6/17.
 */
public interface LayerNormalization {
    void normalize(INDArray activations);
}
