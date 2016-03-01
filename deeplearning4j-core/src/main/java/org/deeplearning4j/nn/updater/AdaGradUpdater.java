package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGrad;
import org.nd4j.linalg.learning.GradientUpdater;


/**
 * Ada grad updater
 *
 * @author Adam Gibson
 */
public class AdaGradUpdater extends BaseUpdater {



    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, INDArray gradient, Layer layer) {
        AdaGrad adaGrad = (AdaGrad) updaterForVariable.get(variable);
        if(adaGrad == null) {
            adaGrad = new AdaGrad(layer.conf().getLearningRateByParam(variable));
            updaterForVariable.put(variable, adaGrad);
        }

        return adaGrad;
    }

    @Override
    public UpdaterAggregator getAggregator(boolean addThis){
        AdaGradAggregator ag = new AdaGradAggregator();
        if(addThis) ag.aggregate(this);
        return ag;
    }

    protected static class AdaGradAggregator extends BaseUpdater.UpdaterAggregatorImpl {
        @Override
        public Updater getUpdater() {
            return setUpdaterState(new AdaGradUpdater());
        }
    }
}
