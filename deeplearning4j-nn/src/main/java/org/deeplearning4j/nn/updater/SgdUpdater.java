package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * @author Adam Gibson
 * @deprecated As of 0.6.0. Use {@link LayerUpdater instead}
 */
@Deprecated
public class SgdUpdater extends BaseUpdater {


    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, Layer layer) {
        org.nd4j.linalg.learning.Sgd updater = (org.nd4j.linalg.learning.Sgd) updaterForVariable.get(variable);
        if(updater == null) {
            updater = new org.nd4j.linalg.learning.Sgd(layer.conf().getLearningRateByParam(variable));
            updaterForVariable.put(variable,updater);
        }

        return updater;
    }
}
