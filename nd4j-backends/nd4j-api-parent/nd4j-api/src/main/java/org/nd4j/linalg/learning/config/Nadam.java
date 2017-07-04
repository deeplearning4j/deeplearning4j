package org.nd4j.linalg.learning.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.NadamUpdater;

import java.util.Arrays;

/**
 * Setup and Builder for Nadam updater.
 * https://arxiv.org/pdf/1609.04747.pdf
 *
 * @author Andrey Spiridonov
 */
@Data
@Builder(builderClassName = "Builder")
public class Nadam implements IUpdater {

    public static final double DEFAULT_NADAM_LEARNING_RATE = 1e-3;
    public static final double DEFAULT_NADAM_EPSILON = 1e-8;
    public static final double DEFAULT_NADAM_BETA1_MEAN_DECAY = 0.9;
    public static final double DEFAULT_NADAM_BETA2_VAR_DECAY = 0.999;

    private double learningRate = 1e-3; // learning rate
    private double beta1 = DEFAULT_NADAM_BETA1_MEAN_DECAY; // gradient moving avg decay rate
    private double beta2 = DEFAULT_NADAM_BETA2_VAR_DECAY; // gradient sqrd decay rate
    private double epsilon = DEFAULT_NADAM_EPSILON;

    public Nadam() {
        this(DEFAULT_NADAM_LEARNING_RATE, DEFAULT_NADAM_BETA1_MEAN_DECAY, DEFAULT_NADAM_BETA2_VAR_DECAY,
                        DEFAULT_NADAM_EPSILON);
    }

    public Nadam(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public long stateSize(long numParams) {
        return 2 * numParams;
    }

    @Override
    public void applySchedules(int iteration, double newLearningRate) {
        this.learningRate = newLearningRate;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        NadamUpdater u = new NadamUpdater(this);
        int[] gradientShape = viewArray.shape();
        gradientShape = Arrays.copyOf(gradientShape, gradientShape.length);
        gradientShape[1] /= 2;
        u.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public Nadam clone() {
        return new Nadam(learningRate, beta1, beta2, epsilon);
    }

    //Partial builder class implementation for default values & public no-arg constructor
    //https://reinhard.codes/2016/07/13/using-lomboks-builder-annotation-with-default-values/
    public static class Builder {
        private double learningRate = DEFAULT_NADAM_LEARNING_RATE;
        private double beta1 = DEFAULT_NADAM_BETA1_MEAN_DECAY;
        private double beta2 = DEFAULT_NADAM_BETA2_VAR_DECAY;
        private double epsilon = DEFAULT_NADAM_EPSILON;

        public Builder() {}
    }
}
