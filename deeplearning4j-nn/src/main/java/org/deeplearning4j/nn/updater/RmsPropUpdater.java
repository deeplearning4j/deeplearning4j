package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * @author Adam Gibson
 * @deprecated As of 0.6.0. Use {@link LayerUpdater instead}
 */
@Deprecated
public class RmsPropUpdater extends BaseUpdater {


    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, Layer layer) {
        org.nd4j.linalg.learning.RmsProp rmsprop = (org.nd4j.linalg.learning.RmsProp) updaterForVariable.get(variable);
        if(rmsprop == null) {
            rmsprop = new org.nd4j.linalg.learning.RmsProp(layer.conf().getLearningRateByParam(variable), layer.conf().getLayer().getRmsDecay());
            updaterForVariable.put(variable,rmsprop);
        }

        return rmsprop;
    }

}
