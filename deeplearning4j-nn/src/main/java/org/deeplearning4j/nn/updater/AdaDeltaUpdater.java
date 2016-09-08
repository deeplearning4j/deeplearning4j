package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.learning.AdaDelta;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * @author Adam Gibson
 * @deprecated As of 0.6.0. Use {@link LayerUpdater instead}
 */
@Deprecated
public class AdaDeltaUpdater extends BaseUpdater {



    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, Layer layer) {
        GradientUpdater updater = updaterForVariable.get(variable);
        if (updater == null) {
            updater = new AdaDelta(layer.conf().getLayer().getRho(), layer.conf().getLayer().getEpsilon());
            updaterForVariable.put(variable,updater);
        }

        return updater;
    }
}
