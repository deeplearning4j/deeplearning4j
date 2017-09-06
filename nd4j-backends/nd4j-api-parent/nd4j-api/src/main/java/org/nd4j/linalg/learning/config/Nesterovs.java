package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.NesterovsUpdater;

import java.util.HashMap;
import java.util.Map;

/**
 * Nesterov's momentum.
 * Keep track of the previous layer's gradient
 * and use it as a way of updating the gradient.
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
@Data
@Builder(builderClassName = "Builder")
public class Nesterovs implements IUpdater {
    public static final double DEFAULT_NESTEROV_MOMENTUM = 0.9;
    public static final double DEFAULT_NESTEROV_LEARNING_RATE = 0.1;

    private double learningRate;
    private double momentum;
    private Map<Integer, Double> momentumSchedule;

    public Nesterovs() {
        this(DEFAULT_NESTEROV_LEARNING_RATE, DEFAULT_NESTEROV_MOMENTUM, null);
    }

    public Nesterovs(double momentum) {
        this(DEFAULT_NESTEROV_LEARNING_RATE, momentum);
    }

    public Nesterovs(double learningRate, double momentum) {
        this(learningRate, momentum, null);
    }

    @Override
    public long stateSize(long numParams) {
        return numParams;
    }

    @Override
    public void applySchedules(int iteration, double newLearningRate) {
        this.learningRate = newLearningRate;
        if (momentumSchedule != null && momentumSchedule.containsKey(iteration)) {
            momentum = momentumSchedule.get(iteration);
        }
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        NesterovsUpdater u = new NesterovsUpdater(this);
        u.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public Nesterovs clone() {
        return new Nesterovs(learningRate, momentum, momentumSchedule == null ? null : new HashMap<>(momentumSchedule));
    }

    //Partial builder class implementation for default values & public no-arg constructor
    //https://reinhard.codes/2016/07/13/using-lomboks-builder-annotation-with-default-values/
    public static class Builder {
        private double learningRate = DEFAULT_NESTEROV_LEARNING_RATE;
        private double momentum = DEFAULT_NESTEROV_LEARNING_RATE;

        public Builder() {}
    }
}
