package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

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
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
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
        Pair<Gradient, INDArray> gradientEpsNext = super.backpropGradient(epsilon, workspaceMgr);
        INDArray epsNext = gradientEpsNext.getSecond();

        // remove singleton fourth dimension from input and current epsilon
        epsNext = epsNext.reshape(epsNext.size(0), epsNext.size(1), epsNext.size(2));
        input = origInput;

        return new Pair<>(gradientEpsNext.getFirst(), epsNext);
    }

    @Override
    protected Pair<INDArray, INDArray> preOutput4d(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        Pair<INDArray,INDArray> preOutput = super.preOutput(true, forBackprop, workspaceMgr);
        INDArray p3d = preOutput.getFirst();
        INDArray p = preOutput.getFirst().reshape(p3d.size(0), p3d.size(1), p3d.size(2), 1);
        preOutput.setFirst(p);
        return preOutput;
    }

    @Override
    protected Pair<INDArray,INDArray> preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        INDArray origInput = input;
        input = input.reshape(input.size(0), input.size(1), input.size(2), 1);

        // call 2D ConvolutionLayer's activate method
        Pair<INDArray,INDArray> preOutput = super.preOutput(training, forBackprop, workspaceMgr);

        // remove singleton fourth dimension from output activations
        input = origInput;
        INDArray p4d = preOutput.getFirst();
        INDArray p = preOutput.getFirst().reshape(p4d.size(0), p4d.size(1), p4d.size(2));
        preOutput.setFirst(p);

        return preOutput;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr){
        INDArray act4d = super.activate(training, workspaceMgr);
        INDArray act3d = act4d.reshape(act4d.size(0), act4d.size(1), act4d.size(2));
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, act3d);   //Should be zero copy most of the time
    }
}
