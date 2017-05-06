package org.nd4j.linalg.learning.config;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.NoOpUpdater;

/**
 * Created by Alex on 06/05/2017.
 */
public class NoOp implements Updater {
    @Override
    public long stateSize(long numParams) {
        return 0;
    }

    @Override
    public void applySchedules(int iteration, double newLearningRate) {

    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        if(viewArray != null){
            throw new IllegalStateException("Cannot use view array with NoOp updater");
        }
        return new NoOpUpdater(this);
    }
}
