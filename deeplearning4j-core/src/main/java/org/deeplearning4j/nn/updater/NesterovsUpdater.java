package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.Nesterovs;

/**
 * @author Adam Gibson
 */
public class NesterovsUpdater extends BaseUpdater {



    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, INDArray gradient, Layer layer) {
        Nesterovs nesterovs = (Nesterovs) updaterForVariable.get(variable);
        if(nesterovs == null) {
            nesterovs = new Nesterovs(layer.conf().getLayer().getMomentum(), layer.conf().getLearningRateByParam(variable));
            updaterForVariable.put(variable,nesterovs);
        }

        return nesterovs;
    }

    @Override
    public UpdaterAggregator getAggregator(boolean addThis){
        NesterovsAggregator ag = new NesterovsAggregator();
        if(addThis) ag.aggregate(this);
        return ag;
    }

    protected static class NesterovsAggregator extends BaseUpdater.UpdaterAggregatorImpl {
        @Override
        public Updater getUpdater() {
            return setUpdaterState(new NesterovsUpdater());
        }
    }
}
