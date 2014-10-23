package org.deeplearning4j.nn.layers;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Convolution layer
 *
 * @author Adam Gibson
 */
public class ConvolutionDownSampleLayer extends BaseLayer {

       public ConvolutionDownSampleLayer(NeuralNetConfiguration conf, INDArray W, INDArray b, INDArray input) {
        super(conf, W, b, input);
    }


    public ConvolutionDownSampleLayer(NeuralNetConfiguration conf) {
        super(conf,null,null,null);


    }

    @Override
    protected INDArray createBias() {
        return Nd4j.zeros(conf.getFilterSize()[0]);
    }

    @Override
    protected INDArray createWeightMatrix() {
        float prod = ArrayUtil.prod(ArrayUtil.removeIndex(conf.getWeightShape(), 0));
        float min = -1 / prod;
        float max = 1 / prod;
        RealDistribution dist = new UniformRealDistribution(conf.getRng(),min,max);
        return Nd4j.rand(conf.getWeightShape(),dist);
    }

    @Override
    public INDArray activate(INDArray input) {
        ActivationFunction f = conf.getActivationFunction();
        INDArray convolution = Convolution.conv2d(input,getW(), Convolution.Type.VALID);
        INDArray pooled = Transforms.maxPool(convolution, conf.getFilterSize(),true);
        INDArray bias = b.dimShuffle(new Object[]{'x',0,'x','x'},new int[]{0},new boolean[]{false});
        pooled.addi(bias.broadcast(pooled.shape()));
        return f.apply(pooled);
    }
}
