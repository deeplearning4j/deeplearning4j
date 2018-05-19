package org.deeplearning4j.nn.layers.convolution;

import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

/**
 * Zero padding layer for convolutional neural networks.
 * Allows padding to be done separately for top/bottom/left/right
 *
 * @author Alex Black
 */
public class ZeroPaddingLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer> {

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
    public void clearNoiseWeightParams() {
        //No op
    }

    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        val inShape = input.shape();

        INDArray epsNext = epsilon.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(padding[0], padding[0] + inShape[2]),
                        NDArrayIndex.interval(padding[2], padding[2] + inShape[3]));

        epsNext = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsNext);
        return new Pair<>((Gradient) new DefaultGradient(), epsNext);
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        val inShape = input.shape();
        val outH = inShape[2] + padding[0] + padding[1];
        val outW = inShape[3] + padding[2] + padding[3];
        val outShape = new long[] {inShape[0], inShape[1], outH, outW};

        INDArray out = workspaceMgr.create(ArrayType.ACTIVATIONS, outShape, 'c');

        out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(padding[0], padding[0] + inShape[2]),
                        NDArrayIndex.interval(padding[2], padding[2] + inShape[3])}, input);

        return out;
    }

    @Override
    public Layer clone() {
        return new ZeroPaddingLayer(conf.clone());
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
