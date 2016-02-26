package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * @author Adam Gibson
 */
public class RmsPropUpdater extends BaseUpdater {


    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, INDArray gradient, Layer layer) {
        org.nd4j.linalg.learning.RmsProp rmsprop = (org.nd4j.linalg.learning.RmsProp) updaterForVariable.get(variable);
        if(rmsprop == null) {
            rmsprop = new org.nd4j.linalg.learning.RmsProp(layer.conf().getLearningRateByParam(variable), layer.conf().getLayer().getRmsDecay());
            updaterForVariable.put(variable,rmsprop);
        }

        return rmsprop;
    }

    @Override
    public UpdaterAggregator getAggregator(boolean addThis){
        RmsPropAggregator ag = new RmsPropAggregator();
        if(addThis) ag.aggregate(this);
        return ag;
    }

    protected static class RmsPropAggregator extends BaseUpdater.UpdaterAggregatorImpl {
        @Override
        public Updater getUpdater() {
            return setUpdaterState(new RmsPropUpdater());
        }
    }
}
