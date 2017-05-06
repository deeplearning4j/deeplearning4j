package org.nd4j.linalg.learning.config;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * Created by Alex on 06/05/2017.
 */
public interface IUpdater {

    long stateSize(long numParams);

    void applySchedules(int iteration, double newLearningRate);

    GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray);

    boolean equals(Object updater);

}
