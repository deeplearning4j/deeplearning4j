package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * @author Adam Gibson
 */
public class AdaDeltaUpdater extends BaseUpdater {



    @Override
    public void init() {

    }

    @Override
    public GradientUpdater init(String variable, INDArray gradient, Layer layer) {

    }
}
