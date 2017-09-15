package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by davekale on 12/7/16.
 */
public class DropoutLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.DropoutLayer> {

    public DropoutLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public DropoutLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public Gradients backpropGradient(Gradients epsilon) {
        INDArray delta = epsilon.get(0).dup();

        if (maskArray != null) {
            delta.muliColumnVector(maskArray);
        }

        Gradient ret = new DefaultGradient();
        return GradientsFactory.getInstance().create(delta, ret);
    }

    @Override
    public INDArray preOutput(boolean training) {
        if (input == null) {
            throw new IllegalArgumentException("Cannot perform forward pass with null input " + layerId());
        }
        applyDropOutIfNecessary(training);

        if (maskArray != null) {
            input.muliColumnVector(maskArray);
        }

        return input;
    }

    @Override
    public Activations activate(boolean training) {
        INDArray z = preOutput(training);
        return ActivationsFactory.getInstance().create(z);
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public INDArray params() {
        return null;
    }
}
