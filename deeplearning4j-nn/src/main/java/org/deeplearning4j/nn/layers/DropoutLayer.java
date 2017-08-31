package org.deeplearning4j.nn.layers;

import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.api.Layer;
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
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public Type type() {
        return Type.FEED_FORWARD;
    }

    @Override
    public void fit(INDArray input) {}

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        INDArray delta = epsilon.dup();

        if (maskArray != null) {
            delta.muliColumnVector(maskArray);
        }

        Gradient ret = new DefaultGradient();
        return new Pair<>(ret, delta);
    }

    @Override
    public INDArray preOutput(boolean training) {
        if (input == null) {
            throw new IllegalArgumentException("Cannot perform forward pass with null input " + layerId());
        }
        applyDropOutIfNecessary(training, -1, -1);      //Dups input if necessary

        if (maskArray != null) {
            input.muliColumnVector(maskArray);
        }

        return input;
    }

    @Override
    public INDArray activate(boolean training) {
        INDArray z = preOutput(training);
        return z;
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException("Not supported - " + layerId());
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }


    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException("Not supported " + layerId());
    }

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException("Not supported - " + layerId());
    }

    @Override
    public INDArray params() {
        return null;
    }
}
