package org.deeplearning4j.nn.layers.convolution;

import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Zero padding 1D layer for convolutional neural networks.
 * Allows padding to be done separately for left and right boundaries.
 *
 * @author Max Pumperla
 */
public class ZeroPadding1DLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer> {

    private int[] padding; // [padLeft, padRight]

    public ZeroPadding1DLayer(NeuralNetConfiguration conf) {
        super(conf);
        this.padding = ((org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer) conf.getLayer()).getPadding();
    }

    @Override
    public INDArray preOutput(boolean training) {
        return activate(training);
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public Type type() {
        return Type.RECURRENT;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        int[] inShape = input.shape();

        INDArray epsNext = epsilon.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(padding[0], padding[0] + inShape[2]));

        return new Pair<>((Gradient) new DefaultGradient(), epsNext);
    }

    @Override
    public INDArray activationMean() {
        throw new UnsupportedOperationException();
    }


    @Override
    public INDArray activate(boolean training) {
        int[] inShape = input.shape();
        int paddedOut = inShape[2] + padding[0] + padding[1];
        int[] outShape = new int[] {inShape[0], inShape[1], paddedOut};

        INDArray out = Nd4j.create(outShape);
        out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(padding[0], padding[0] + inShape[2])}, input);

        return out;
    }

    @Override
    public Layer clone() {
        return new ZeroPadding1DLayer(conf.clone());
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }
}
