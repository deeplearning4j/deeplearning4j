package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;

import java.util.LinkedHashMap;

/**
 * @author Adam Gibson
 */
public class SgdUpdater extends BaseUpdater {


    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, INDArray gradient, Layer layer) {
        org.nd4j.linalg.learning.Sgd updater = (org.nd4j.linalg.learning.Sgd) updaterForVariable.get(variable);
        if(updater == null) {
            updater = new org.nd4j.linalg.learning.Sgd(layer.conf().getLearningRateByParam(variable));
            updaterForVariable.put(variable,updater);
        }

        return updater;
    }

    @Override
    public UpdaterAggregator getAggregator(boolean addThis){
        SgdAggregator ag = new SgdAggregator();
        if(addThis) ag.aggregate(this);
        return ag;
    }

    protected static class SgdAggregator extends BaseUpdater.UpdaterAggregatorImpl {
        @Override
        public Updater getUpdater() {
            return setUpdaterState(new SgdUpdater());
        }
    }
}
