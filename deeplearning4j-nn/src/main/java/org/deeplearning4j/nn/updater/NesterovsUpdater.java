package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.Nesterovs;

/**
 * @author Adam Gibson
 * @deprecated As of 0.6.0. Use {@link LayerUpdater instead}
 */
@Deprecated
public class NesterovsUpdater extends BaseUpdater {



    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, Layer layer) {
        Nesterovs nesterovs = (Nesterovs) updaterForVariable.get(variable);
        if(nesterovs == null) {
            nesterovs = new Nesterovs(layer.conf().getLayer().getMomentum(), layer.conf().getLearningRateByParam(variable));
            updaterForVariable.put(variable,nesterovs);
        }

        return nesterovs;
    }
}
