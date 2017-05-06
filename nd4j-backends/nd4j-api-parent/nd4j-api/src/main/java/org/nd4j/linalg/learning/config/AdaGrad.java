package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGradUpdater;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * Created by Alex on 06/05/2017.
 */
@Builder
@AllArgsConstructor
@Data
public class AdaGrad implements IUpdater {

    public static final double DEFAULT_ADAGRAD_LEARNING_RATE = 1e-1;
    public static final double DEFAULT_ADAGRAD_EPSILON = 1e-6;

    private double learningRate;
    private double epsilon;

    public AdaGrad(){
        this(DEFAULT_ADAGRAD_LEARNING_RATE, DEFAULT_ADAGRAD_EPSILON);
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
        AdaGradUpdater u = new AdaGradUpdater(this);
        u.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public AdaGrad clone() {
        return new AdaGrad(learningRate, epsilon);
    }
}
