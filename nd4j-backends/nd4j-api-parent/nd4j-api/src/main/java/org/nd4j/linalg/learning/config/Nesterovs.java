package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.NesterovsUpdater;

/**
 * Created by Alex on 06/05/2017.
 */
@AllArgsConstructor
@Data
public class Nesterovs implements Updater {
    public static final double DEFAULT_NESTEROV_MOMENTUM = 0.9;
    public static final double DEFAULT_NESTEROV_LEARNING_RATE = 0.1;

    private double learningRate;
    private double momentum;

    public Nesterovs(){
        this(DEFAULT_NESTEROV_LEARNING_RATE, DEFAULT_NESTEROV_MOMENTUM);
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
        NesterovsUpdater u = new NesterovsUpdater(this);
        u.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);
        return u;
    }
}
