package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.Shape;

/**
 * Convolution layer
 *
 * @author Adam Gibson
 */
public class ConvolutionDownSampleLayer extends BaseLayer {


    public ConvolutionDownSampleLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public ConvolutionDownSampleLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public INDArray activate(INDArray input) {
        ActivationFunction f = conf.getActivationFunction();
        INDArray W = getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS);
        INDArray b = getParam(ConvolutionParamInitializer.CONVOLUTION_BIAS);

        INDArray convolution = Convolution.conv2d(input,W, Convolution.Type.VALID);
        INDArray pooled = Transforms.maxPool(convolution, conf.getFilterSize(),true);
        INDArray bias = b.dimShuffle(new Object[]{'x',0,'x','x'},new int[]{0},new boolean[]{false});
        INDArray broadCastedBias = bias.broadcast(pooled.shape());
        pooled.addi(broadCastedBias);
        return f.apply(pooled);
    }

    @Override
    public void update(Gradient gradient) {

    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public INDArray transform(INDArray data) {
        return null;
    }

    @Override
    public void setParams(INDArray params) {

    }

    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient getGradient() {
        return null;
    }
}
