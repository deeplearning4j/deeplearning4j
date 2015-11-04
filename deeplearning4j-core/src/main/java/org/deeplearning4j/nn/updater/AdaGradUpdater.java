package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGrad;
import org.nd4j.linalg.learning.GradientUpdater;


/**
 * Ada grad updater
 *
 * @author Adam Gibson
 */
public class AdaGradUpdater extends BaseUpdater {



    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, INDArray gradient, Layer layer) {
        AdaGrad adaGrad = (AdaGrad) updaterForVariable.get(variable);
        if(adaGrad == null) {
            adaGrad = new AdaGrad(layer.conf().getLayer().getLearningRate());
            updaterForVariable.put(variable, adaGrad);
        }

        return adaGrad;
    }
}
