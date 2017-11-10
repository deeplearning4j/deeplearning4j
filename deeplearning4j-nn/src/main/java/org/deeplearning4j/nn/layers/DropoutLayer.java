package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by davekale on 12/7/16.
 */
public class DropoutLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.DropoutLayer> {

    public DropoutLayer(org.deeplearning4j.nn.conf.layers.DropoutLayer conf) {
        super(conf);
    }

    @Override
    public Gradients backpropGradient(Gradients epsilon) {
        INDArray delta = epsilon.get(0).dup();

        if (this.input.getMask(0) != null) {
            delta.muliColumnVector(this.input.getMask(0));
        }

        Gradient ret = new DefaultGradient();
        Gradients g = GradientsFactory.getInstance().create(delta, ret);
        return backpropPreprocessor(g);
    }

    @Override
    public INDArray preOutput(boolean training) {
        if (input == null || input.get(0) == null) {
            throw new IllegalArgumentException("Cannot perform forward pass with null input " + layerId());
        }
        applyPreprocessorIfNecessary(training);
        applyDropOutIfNecessary(training);

        if (this.input.getMask(0) != null) {
            input.get(0).muliColumnVector(this.input.getMask(0));
        }

        return input.get(0);
    }

    @Override
    public Activations activate(boolean training) {
        INDArray z = preOutput(training);
        return ActivationsFactory.getInstance().create(z, input.getMask(0), input.getMaskState(0));
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
