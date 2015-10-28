package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;

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
            updater = new org.nd4j.linalg.learning.Sgd(layer.conf().getLayer().getLearningRate());
            updaterForVariable.put(variable,updater);
        }

        return updater;
    }
}
