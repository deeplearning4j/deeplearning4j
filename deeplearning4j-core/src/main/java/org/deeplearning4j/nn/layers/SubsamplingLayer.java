package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 9/7/14.
 */
public class SubsamplingLayer extends BaseLayer {
    public SubsamplingLayer(NeuralNetConfiguration conf, INDArray W, INDArray b, INDArray input) {
        super(conf, W, b, input);
    }

    @Override
    protected INDArray createWeightMatrix() {
        return super.createWeightMatrix();
    }

    @Override
    public INDArray activate(INDArray input) {
        return super.activate(input);
    }
}
