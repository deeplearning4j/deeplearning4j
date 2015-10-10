package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.Adam;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * @author Adam Gibson
 */
public class AdamUpdater extends BaseUpdater {



    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, INDArray gradient, Layer layer) {
        Adam adam = (Adam) updaterForVariable.get(variable);
        if(adam == null) {
            adam = new Adam(layer.conf().getLr(), 0.9, 0.999, 1e-8);
            updaterForVariable.put(variable,adam);

        }

        return adam;
    }
}
