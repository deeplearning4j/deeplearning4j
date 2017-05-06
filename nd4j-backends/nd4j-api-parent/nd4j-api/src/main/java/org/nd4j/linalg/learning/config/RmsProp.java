package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.RmsPropUpdater;

/**
 * Created by Alex on 06/05/2017.
 */
@Data
@AllArgsConstructor
public class RmsProp implements Updater {
    public static final double DEFAULT_RMSPROP_LEARNING_RATE = 1e-1;
    public static final double DEFAULT_RMSPROP_EPSILON = 1e-8;
    public static final double DEFAULT_RMSPROP_RMSDECAY = 0.95;

    private double learningRate = 1e-1;
    private double rmsDecay = DEFAULT_RMSPROP_RMSDECAY;
    private double epsilon = DEFAULT_RMSPROP_EPSILON;

    public RmsProp(){
        this(DEFAULT_RMSPROP_LEARNING_RATE, DEFAULT_RMSPROP_RMSDECAY, DEFAULT_RMSPROP_EPSILON);
    }

    @Override
    public long stateSize(long numParams) {
        return numParams;
    }

    @Override
    public void applySchedules(int iteration, double newLearningRate) {
        this.learningRate = newLearningRate;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        RmsPropUpdater u = new RmsPropUpdater(this);
        u.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);
        return u;
    }
}
