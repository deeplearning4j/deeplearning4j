package org.deeplearning4j.nn.layers.convolution.subsampling;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * 1D (temporal) subsampling layer. Currently, we just subclass off the
 * SubsamplingLayer and override the preOutput and backpropGradient methods.
 * Specifically, since this layer accepts RNN (not CNN) InputTypes, we
 * need to add a singleton fourth dimension before calling the respective
 * superclass method, then remove it from the result.
 *
 * This approach treats a multivariate time series with L timesteps and
 * P variables as an L x 1 x P image (L rows high, 1 column wide, P
 * channels deep). The kernel should be H<L pixels high and W=1 pixels
 * wide.
 *
 * TODO: We will eventually want to add a 1D-specific im2col method.
 *
 * @author dave@skymind.io
 */
public class Subsampling1DLayer extends SubsamplingLayer {
    public Subsampling1DLayer(org.deeplearning4j.nn.conf.layers.Subsampling1DLayer conf) {
        super(conf);
    }

    @Override
    public Gradients backpropGradient(Gradients gradients) {
        INDArray epsilon = gradients.get(0);
        if (epsilon.rank() != 3)
            throw new DL4JInvalidInputException("Got rank " + epsilon.rank()
                            + " array as epsilon for Subsampling1DLayer backprop with shape "
                            + Arrays.toString(epsilon.shape())
                            + ". Expected rank 3 array with shape [minibatchSize, features, length]. " + layerId());

        // add singleton fourth dimension to input and next layer's epsilon
        INDArray input = this.input.get(0);
        INDArray origInput = input;
        this.input.set(0, input.reshape(input.size(0), input.size(1), input.size(2), 1));
        epsilon = epsilon.reshape(epsilon.size(0), epsilon.size(1), epsilon.size(2), 1);

        // call 2D SubsamplingLayer's backpropGradient method
        gradients.set(0, epsilon);
        Gradients gradientEpsNext = super.backpropGradient(gradients);
        INDArray epsNext = gradientEpsNext.get(0);

        // remove singleton fourth dimension from input and current epsilon
        this.input.set(0, origInput);
        epsNext = epsNext.reshape(epsNext.size(0), epsNext.size(1), epsNext.size(2));
        gradientEpsNext.set(0, epsNext);

        return gradientEpsNext;
    }

    @Override
    public Activations activate(boolean training) {
        INDArray input = this.input.get(0);
        if (input.rank() != 3)
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                            + " array as input to Subsampling1DLayer with shape " + Arrays.toString(input.shape())
                            + ". Expected rank 3 array with shape [minibatchSize, features, length]. " + layerId());

        // add singleton fourth dimension to input
        INDArray origInput = input;
        this.input.set(0, input.reshape(input.size(0), input.size(1), input.size(2), 1));

        // call 2D SubsamplingLayer's activate method
        Activations acts = super.activate(training);

        // remove singleton fourth dimension from input and output activations
        this.input.set(0, origInput);
        INDArray a = acts.get(0);
        acts.set(0, acts.get(0).reshape(a.size(0), a.size(1), a.size(2)));

        return acts;
    }
}
