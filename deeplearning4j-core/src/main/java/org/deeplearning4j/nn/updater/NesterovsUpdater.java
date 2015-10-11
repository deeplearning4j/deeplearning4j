package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
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
            nesterovs = new Nesterovs(layer.conf().getMomentum(), layer.conf().getMomentumAfter(), layer.conf().getLr());
            updaterForVariable.put(variable,nesterovs);
        }

        return nesterovs;
    }
}
