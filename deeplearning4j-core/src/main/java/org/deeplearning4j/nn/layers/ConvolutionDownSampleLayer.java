package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.ndarray.DimensionSlice;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SliceOp;
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
        final INDArray b = getParam(ConvolutionParamInitializer.CONVOLUTION_BIAS);

        INDArray convolution = Convolution.conv2d(input,W, Convolution.Type.VALID);
        INDArray pooled = Transforms.maxPool(convolution, conf.getFilterSize(),true);
        final INDArray bias = b.broadcast(pooled.shape()[pooled.shape().length - 1]);
        pooled.iterateOverAllRows(new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {

            }

            @Override
            public void operate(INDArray nd) {
               nd.addi(bias);
            }
        });

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
        return activate(data);
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
