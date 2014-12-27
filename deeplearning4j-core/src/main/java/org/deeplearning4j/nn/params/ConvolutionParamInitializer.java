package org.deeplearning4j.nn.params;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Map;

/**
 * Initialize convolution params.
 * @author Adam Gibson
 */
public class ConvolutionParamInitializer implements ParamInitializer {

    public final static String CONVOLUTION_BIAS = "convbias";
    public final static String CONVOLUTION_WEIGHTS = "convweights";
    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        params.put(CONVOLUTION_BIAS,createBias(conf));
        params.put(CONVOLUTION_WEIGHTS,createWeightMatrix(conf));
    }



    protected INDArray createBias(NeuralNetConfiguration conf) {
        return Nd4j.zeros(conf.getFilterSize()[0]);
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf) {
        float prod = ArrayUtil.prod(ArrayUtil.removeIndex(conf.getWeightShape(), 0));
        float min = -1 / prod;
        float max = 1 / prod;
        RealDistribution dist = new UniformRealDistribution(conf.getRng(),min,max);
        return Nd4j.rand(conf.getWeightShape(),dist);
    }

}
