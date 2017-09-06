package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaMaxUpdater;
import org.nd4j.linalg.learning.GradientUpdater;

import java.util.Arrays;

/**
 * The AdaMax updater, a variant of Adam.
 * http://arxiv.org/abs/1412.6980
 *
 * @author Justin Long
 */
@Data
@Builder(builderClassName = "Builder")
@AllArgsConstructor
@NoArgsConstructor
public class AdaMax implements IUpdater {
    public static final double DEFAULT_ADAMAX_LEARNING_RATE = 1e-3;
    public static final double DEFAULT_ADAMAX_EPSILON = 1e-8;
    public static final double DEFAULT_ADAMAX_BETA1_MEAN_DECAY = 0.9;
    public static final double DEFAULT_ADAMAX_BETA2_VAR_DECAY = 0.999;

    private double learningRate = DEFAULT_ADAMAX_LEARNING_RATE; // learning rate
    private double beta1 = DEFAULT_ADAMAX_BETA1_MEAN_DECAY; // gradient moving avg decay rate
    private double beta2 = DEFAULT_ADAMAX_BETA2_VAR_DECAY; // gradient sqrd decay rate
    private double epsilon = DEFAULT_ADAMAX_EPSILON;

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
        AdaMaxUpdater a = new AdaMaxUpdater(this);
        int[] gradientShape = viewArray.shape();
        gradientShape = Arrays.copyOf(gradientShape, gradientShape.length);
        gradientShape[1] /= 2;
        a.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray);
        return a;
    }

    @Override
    public IUpdater clone() {
        return new AdaMax(learningRate, beta1, beta2, epsilon);
    }

    //Partial builder class implementation for default values & public no-arg constructor
    //https://reinhard.codes/2016/07/13/using-lomboks-builder-annotation-with-default-values/
    public static class Builder {
        private double learningRate = DEFAULT_ADAMAX_LEARNING_RATE;
        private double beta1 = DEFAULT_ADAMAX_BETA1_MEAN_DECAY;
        private double beta2 = DEFAULT_ADAMAX_BETA2_VAR_DECAY;
        private double epsilon = DEFAULT_ADAMAX_EPSILON;

        public Builder() {}
    }
}
