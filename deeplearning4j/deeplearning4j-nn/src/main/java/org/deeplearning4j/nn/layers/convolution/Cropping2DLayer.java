package org.deeplearning4j.nn.layers.convolution;

import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * Zero cropping layer for convolutional neural networks.
 * Allows cropping to be done separately for top/bottom/left/right
 *
 * @author Alex Black
 */
public class Cropping2DLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D> {

    private int[] cropping; //[padTop, padBottom, padLeft, padRight]

    public Cropping2DLayer(NeuralNetConfiguration conf) {
        super(conf);
        this.cropping = ((org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D) conf.getLayer()).getCropping();
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
        val inShape = input.shape();
        INDArray epsNext = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, inShape, 'c');
        INDArray epsNextSubset = inputSubset(epsNext);
        epsNextSubset.assign(epsilon);
        return new Pair<>((Gradient) new DefaultGradient(), epsNext);
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        INDArray ret = inputSubset(input);
        ret = workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, ret);
        workspaceMgr.validateArrayLocation(ArrayType.ACTIVATIONS, ret, false, false);
        return ret;
    }

    @Override
    public Layer clone() {
        return new Cropping2DLayer(conf.clone());
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    private INDArray inputSubset(INDArray from){
        //NCHW format
        return from.get(all(), all(),
                interval(cropping[0], from.size(2)-cropping[1]),
                interval(cropping[2], from.size(3)-cropping[3]));
    }
}
