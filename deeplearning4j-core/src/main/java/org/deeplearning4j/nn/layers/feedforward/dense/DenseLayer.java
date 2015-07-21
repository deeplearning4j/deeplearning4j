package org.deeplearning4j.nn.layers.feedforward.dense;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder;
import org.nd4j.linalg.api.ndarray.INDArray;

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

    @Override
    public void update(INDArray gradient, String paramType) {
        if (paramType.contains("b"))
            setParam(paramType, getParam(paramType).subi(gradient.sum(0)));
        else
            setParam(paramType, getParam(paramType).subi(gradient));
    }

}
