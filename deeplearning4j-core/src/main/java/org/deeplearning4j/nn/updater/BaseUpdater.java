package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Adam Gibson
 */
public abstract class BaseUpdater implements Updater {
     protected Map<String,GradientUpdater> updaterForVariable = new HashMap<>();


    @Override
    public void update(Layer layer, Gradient gradient) {
        for(Map.Entry<String,INDArray> gradientPair : gradient.gradientForVariable().entrySet()) {
            GradientUpdater updater = init(gradientPair.getKey(),gradientPair.getValue(),layer);
            layer.getParam(gradientPair.getKey()).subi(updater.getGradient(gradientPair.getValue()));
        }
    }
    public abstract void init();

    public abstract GradientUpdater init(String variable, INDArray gradient, Layer layer);

}
