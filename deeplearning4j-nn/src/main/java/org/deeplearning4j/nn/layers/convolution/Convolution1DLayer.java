package org.deeplearning4j.nn.layers.convolution;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * 1D (temporal) convolutional layer. Currently, we just subclass off the
 * ConvolutionLayer and override the preOutput and backpropGradient methods.
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
public class Convolution1DLayer extends ConvolutionLayer {
    public Convolution1DLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public Convolution1DLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        if (epsilon.rank() != 3)
            throw new DL4JInvalidInputException("Got rank " + epsilon.rank()
                            + " array as epsilon for Convolution1DLayer backprop with shape "
                            + Arrays.toString(epsilon.shape())
                            + ". Expected rank 3 array with shape [minibatchSize, features, length]. " + layerId());

        // add singleton fourth dimension to input and next layer's epsilon
        epsilon = epsilon.reshape(epsilon.size(0), epsilon.size(1), epsilon.size(2), 1);
        INDArray origInput = input;
        input = input.reshape(input.size(0), input.size(1), input.size(2), 1);

        // call 2D ConvolutionLayer's backpropGradient method
        Pair<Gradient, INDArray> gradientEpsNext = super.backpropGradient(epsilon);
        INDArray epsNext = gradientEpsNext.getSecond();

        // remove singleton fourth dimension from input and current epsilon
        epsNext = epsNext.reshape(epsNext.size(0), epsNext.size(1), epsNext.size(2));
        input = origInput;

        return new Pair<>(gradientEpsNext.getFirst(), epsNext);
    }

    @Override
    protected Pair<INDArray, INDArray> preOutput4d(boolean training, boolean forBackprop) {
        return super.preOutput(true, forBackprop);
    }

    @Override
    public INDArray preOutput(boolean training) {
        INDArray origInput = input;
        input = input.reshape(input.size(0), input.size(1), input.size(2), 1);

        // call 2D ConvolutionLayer's activate method
        INDArray preOutput = super.preOutput(training);

        // remove singleton fourth dimension from output activations
        input = origInput;
        preOutput = preOutput.reshape(preOutput.size(0), preOutput.size(1), preOutput.size(2));

        return preOutput;
    }
}
