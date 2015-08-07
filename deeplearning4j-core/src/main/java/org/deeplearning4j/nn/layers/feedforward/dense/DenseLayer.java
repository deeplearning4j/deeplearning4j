package org.deeplearning4j.nn.layers.feedforward.dense;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class DenseLayer extends BaseLayer {
    public DenseLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public DenseLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }
}
