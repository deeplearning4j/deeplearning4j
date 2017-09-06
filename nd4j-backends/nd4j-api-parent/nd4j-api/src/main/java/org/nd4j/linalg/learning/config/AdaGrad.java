package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGradUpdater;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 * Vectorized Learning Rate used per Connection Weight
 * <p/>
 * Adapted from: http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
 * See also http://cs231n.github.io/neural-networks-3/#ada
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
@Data
@Builder(builderClassName = "Builder")
public class AdaGrad implements IUpdater {

    public static final double DEFAULT_ADAGRAD_LEARNING_RATE = 1e-1;
    public static final double DEFAULT_ADAGRAD_EPSILON = 1e-6;

    private double learningRate;
    private double epsilon;

    public AdaGrad() {
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

    //Partial builder class implementation for default values & public no-arg constructor
    //https://reinhard.codes/2016/07/13/using-lomboks-builder-annotation-with-default-values/
    public static class Builder {
        private double learningRate = DEFAULT_ADAGRAD_LEARNING_RATE;
        private double epsilon = DEFAULT_ADAGRAD_EPSILON;

        public Builder() {}
    }
}
