package org.deeplearning4j.nn.updater.custom;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * Created by Alex on 09/05/2017.
 */
@AllArgsConstructor
public class CustomGradientUpdater implements GradientUpdater<CustomIUpdater> {

    private CustomIUpdater config;

    @Override
    public CustomIUpdater getConfig() {
        return config;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, long[] gradientShape, char gradientOrder, boolean initialize) {
        //No op
    }

    @Override
    public void applyUpdater(INDArray gradient, int iteration, int epoch) {
        gradient.muli(config.getLearningRate());
    }
}
