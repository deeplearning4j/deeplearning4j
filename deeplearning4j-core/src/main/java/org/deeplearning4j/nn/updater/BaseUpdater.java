package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Adam Gibson
 */
public abstract class BaseUpdater implements Updater {
    protected Map<String,GradientUpdater> updaterForVariable = new HashMap<>();


    @Override
    public void update(Layer layer, Gradient gradient,int iteration) {
        for(Map.Entry<String,INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
            GradientUpdater updater = init(gradientPair.getKey(),gradientPair.getValue(),layer);
            INDArray gradient2 = updater.getGradient(gradientPair.getValue(), iteration);
            postApply(layer,gradient2,gradientPair.getKey());
            gradient.setGradientFor(gradientPair.getKey(),gradient2);
        }
    }

    /**
     * Apply the regularization
     * @param layer
     * @param gradient
     * @param param
     */
    public void postApply(Layer layer,INDArray gradient,String param) {
        NeuralNetConfiguration conf = layer.conf();
        INDArray params = layer.getParam(param);
        if(conf.isUseRegularization() && conf.getLayer().getL2() > 0 && !(param.equals(DefaultParamInitializer.BIAS_KEY)))
        	gradient.addi(params.mul(conf.getLayer().getL2()));	//dC/dw = dC0/dw + lambda/n * w where C0 is pre-l2 cost function
        if(conf.isUseRegularization() && conf.getLayer().getL1() > 0 && !(param.equals(DefaultParamInitializer.BIAS_KEY)))
        	gradient.addi(Transforms.sign(params).muli(conf.getLayer().getL1()));
        if(conf.isMiniBatch())
            gradient.divi(layer.getInputMiniBatchSize());
        if(conf.isConstrainGradientToUnitNorm())
            gradient.divi(gradient.norm2(Integer.MAX_VALUE));

    }

    public abstract void init();

    public abstract GradientUpdater init(String variable, INDArray gradient, Layer layer);

}
