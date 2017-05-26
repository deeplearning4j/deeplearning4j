package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Zero padding layer for convolutional neural networks.
 * Allows padding to be done separately for top/bottom/left/right
 *
 * @author Alex Black
 */
public class ZeroPaddingLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer> {

    private int[] padding; //[padTop, padBottom, padLeft, padRight]

    public ZeroPaddingLayer(NeuralNetConfiguration conf) {
        super(conf);
        this.padding = ((org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer) conf.getLayer()).getPadding();
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        int[] inShape = input.shape();

        INDArray epsNext = epsilon.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(padding[0], padding[0] + inShape[2]),
                        NDArrayIndex.interval(padding[2], padding[2] + inShape[3]));

        return new Pair<>((Gradient) new DefaultGradient(), epsNext);
    }


    @Override
    public INDArray activate(boolean training) {
        int[] inShape = input.shape();
        int outH = inShape[2] + padding[0] + padding[1];
        int outW = inShape[3] + padding[2] + padding[3];
        int[] outShape = new int[] {inShape[0], inShape[1], outH, outW};

        INDArray out = Nd4j.create(outShape);

        out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(padding[0], padding[0] + inShape[2]),
                        NDArrayIndex.interval(padding[2], padding[2] + inShape[3])}, input);

        return out;
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
