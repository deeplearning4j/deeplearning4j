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
public class ConvolutionLayer extends BaseLayer {
    private int[] shape;

    //number of feature mapConvolution
    protected int[] numFilters = {4,4};

    public ConvolutionLayer(NeuralNetConfiguration conf, INDArray W, INDArray b, INDArray input) {
        super(conf, W, b, input);
    }


    public ConvolutionLayer(int[] shape,NeuralNetConfiguration conf) {
        super(conf,null,null,null);
        this.shape = shape;

    }

    @Override
    protected INDArray createBias() {
        return Nd4j.zeros(numFilters[0]);
    }

    @Override
    protected INDArray createWeightMatrix() {
        float prod = ArrayUtil.prod(ArrayUtil.removeIndex(shape, 0));
        float min = -1 / prod;
        float max = 1 / prod;
        RealDistribution dist = new UniformRealDistribution(conf.getRng(),min,max);
        return Nd4j.rand(shape,dist);
    }

    @Override
    public INDArray activate(INDArray input) {
        ActivationFunction f = conf.getActivationFunction();
        INDArray convolution = Convolution.conv2d(input,getW(), Convolution.Type.VALID);
        INDArray pooled = Transforms.maxPool(convolution,numFilters,true);
        INDArray bias = b.broadcast(new int[]{1,numFilters[0],1,1});
        pooled.addi(bias);
        return f.apply(pooled);
    }
}
