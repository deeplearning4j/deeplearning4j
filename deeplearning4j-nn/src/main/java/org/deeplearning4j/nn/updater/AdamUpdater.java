package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.learning.Adam;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * @author Adam Gibson
 * @deprecated As of 0.6.0. Use {@link LayerUpdater instead}
 */
@Deprecated
public class AdamUpdater extends BaseUpdater {



    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, Layer layer) {
        Adam adam = (Adam) updaterForVariable.get(variable);
        if(adam == null) {
            adam = new Adam(layer.conf().getLearningRateByParam(variable),
                    layer.conf().getLayer().getAdamMeanDecay(),
                    layer.conf().getLayer().getAdamVarDecay());
            updaterForVariable.put(variable,adam);
        }

        return adam;
    }
}
