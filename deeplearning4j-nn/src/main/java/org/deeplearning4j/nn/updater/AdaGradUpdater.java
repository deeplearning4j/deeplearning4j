package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.learning.AdaGrad;
import org.nd4j.linalg.learning.GradientUpdater;


/**
 * Ada grad updater
 *
 * @author Adam Gibson
 * @deprecated As of 0.6.0. Use {@link LayerUpdater instead}
 */
@Deprecated
public class AdaGradUpdater extends BaseUpdater {



    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, Layer layer) {
        AdaGrad adaGrad = (AdaGrad) updaterForVariable.get(variable);
        if(adaGrad == null) {
            adaGrad = new AdaGrad(layer.conf().getLearningRateByParam(variable), layer.conf().getLayer().getEpsilon());
            updaterForVariable.put(variable, adaGrad);
        }

        return adaGrad;
    }
}
