package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaDelta;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * @author Adam Gibson
 */
public class AdaDeltaUpdater extends BaseUpdater {



    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, Layer layer) {
        GradientUpdater updater = updaterForVariable.get(variable);
        if (updater == null) {
            updater = new AdaDelta(layer.conf().getLayer().getRho());
            updaterForVariable.put(variable,updater);
        }

        return updater;
    }

    protected static class AdaDeltaAggregator extends BaseUpdater.UpdaterAggregatorImpl {
        @Override
        public Updater getUpdater() {
            return setUpdaterState(new AdaDeltaUpdater());
        }
    }
}
