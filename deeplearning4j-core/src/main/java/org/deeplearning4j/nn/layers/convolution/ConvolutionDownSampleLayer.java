package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
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
    public INDArray activate() {
        ActivationFunction f = conf.getActivationFunction();
        INDArray W = getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS);
        if(W.shape()[1] != input.shape()[1])
            throw new IllegalStateException("Input size at dimension 1 must be same as the filter size");
        final INDArray b = getParam(ConvolutionParamInitializer.CONVOLUTION_BIAS);

        INDArray convolution = Convolution.conv2d(input,W, Convolution.Type.VALID);
        if(convolution.shape().length < 4) {
            int[] newShape = new int[4];
            for(int i = 0; i < newShape.length; i++)
                newShape[i] = 1;
            int lengthDiff = 4 - convolution.shape().length;
            for(int i = lengthDiff; i < 4; i++)
                newShape[i] = convolution.shape()[i - lengthDiff];
            convolution = convolution.reshape(newShape);

        }

        final INDArray pooled = Transforms.maxPool(convolution, conf.getStride(),true);
        final INDArray bias = b.dimShuffle(new Object[]{'x',0,'x','x'},new int[4],new boolean[]{true});
        final INDArray broadCasted = bias.broadcast(pooled.shape());
        broadCasted.iterateOverAllRows(new SliceOp() {
            @Override
            public void operate(DimensionSlice nd) {

            }

            @Override
            public void operate(final INDArray nd1) {
                pooled.iterateOverAllRows(new SliceOp() {
                    @Override
                    public void operate(DimensionSlice nd) {

                    }

                    @Override
                    public void operate(INDArray nd2) {
                        nd1.addi(nd2);
                    }
                });
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

    @Override
    public void fit() {
        //no-op
    }

    @Override
    public void fit(INDArray input) {
        //no-op
    }
}
