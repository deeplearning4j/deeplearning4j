package org.deeplearning4j.nn.layers.feedforward.dense;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class DenseLayer extends AutoEncoder {
    public DenseLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public DenseLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


}
