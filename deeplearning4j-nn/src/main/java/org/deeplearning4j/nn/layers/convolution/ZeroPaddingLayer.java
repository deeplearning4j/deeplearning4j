package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
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
public class ZeroPaddingLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer> {

    private static final Gradient EMPTY_GRADIENT = new DefaultGradient();
    private int[] padding; //[padTop, padBottom, padLeft, padRight]

    public ZeroPaddingLayer(org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer conf) {
        super(conf);
        this.padding = conf.getPadding();
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //No op
    }

    @Override
    public Gradients backpropGradient(Gradients gradients) {
        INDArray input = this.input.get(0);
        INDArray epsilon = gradients.get(0);
        int[] inShape = input.shape();

        INDArray epsNext = epsilon.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(padding[0], padding[0] + inShape[2]),
                        NDArrayIndex.interval(padding[2], padding[2] + inShape[3]));

        Gradients g = GradientsFactory.getInstance().create(epsNext, EMPTY_GRADIENT);
        return backpropPreprocessor(g);
    }


    @Override
    public Activations activate(boolean training) {
        INDArray input = this.input.get(0);
        int[] inShape = input.shape();
        int outH = inShape[2] + padding[0] + padding[1];
        int outW = inShape[3] + padding[2] + padding[3];
        int[] outShape = new int[] {inShape[0], inShape[1], outH, outW};

        INDArray out = Nd4j.create(outShape);

        out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(padding[0], padding[0] + inShape[2]),
                        NDArrayIndex.interval(padding[2], padding[2] + inShape[3])}, input);

        return ActivationsFactory.getInstance().create(out);
    }
}
