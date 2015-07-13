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
    public void update(Layer layer, Gradient gradient) {
        for(Map.Entry<String,INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
            GradientUpdater updater = init(gradientPair.getKey(),gradientPair.getValue(),layer);
            postApply(layer,updater.getGradient(gradientPair.getValue()),gradientPair.getKey());
        }
    }

    public void postApply(Layer layer,INDArray gradient,String param) {
        NeuralNetConfiguration conf = layer.conf();
        INDArray params = layer.getParam(param);
        if(conf.isUseRegularization() && conf.getL2() > 0 && !(gradient.equals(DefaultParamInitializer.BIAS_KEY)))
            gradient.subi(params.mul(conf.getL2()));
        else if(conf.isUseRegularization() && conf.getL1() < 0 && !(gradient.equals(DefaultParamInitializer.BIAS_KEY)))
            gradient.subi(Transforms.sign(params).muli(conf.getL1()));


        if(conf.isConstrainGradientToUnitNorm())
            gradient.divi(gradient.norm2(Integer.MAX_VALUE));

        gradient.divi(layer.input().size(0));
    }

    public abstract void init();

    public abstract GradientUpdater init(String variable, INDArray gradient, Layer layer);

}
