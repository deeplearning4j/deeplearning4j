package org.nd4j.linalg.learning.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdamUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * The Adam updater.
 * http://arxiv.org/abs/1412.6980
 *
 * @author Adam Gibson
 */
@Data
@Builder(builderClassName = "Builder")
public class Adam implements IUpdater {

    public static final double DEFAULT_ADAM_LEARNING_RATE = 1e-3;
    public static final double DEFAULT_ADAM_EPSILON = 1e-8;
    public static final double DEFAULT_ADAM_BETA1_MEAN_DECAY = 0.9;
    public static final double DEFAULT_ADAM_BETA2_VAR_DECAY = 0.999;

    @lombok.Builder.Default private double learningRate = DEFAULT_ADAM_LEARNING_RATE; // learning rate
    private ISchedule learningRateSchedule;
    @lombok.Builder.Default private double beta1 = DEFAULT_ADAM_BETA1_MEAN_DECAY; // gradient moving avg decay rate
    @lombok.Builder.Default private double beta2 = DEFAULT_ADAM_BETA2_VAR_DECAY; // gradient sqrt decay rate
    @lombok.Builder.Default private double epsilon = DEFAULT_ADAM_EPSILON;

    public Adam() {
        this(DEFAULT_ADAM_LEARNING_RATE, DEFAULT_ADAM_BETA1_MEAN_DECAY, DEFAULT_ADAM_BETA2_VAR_DECAY,
                        DEFAULT_ADAM_EPSILON);
    }

    public Adam(double learningRate){
        this(learningRate, null, DEFAULT_ADAM_BETA1_MEAN_DECAY, DEFAULT_ADAM_BETA2_VAR_DECAY, DEFAULT_ADAM_EPSILON);
    }

    public Adam(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_ADAM_BETA1_MEAN_DECAY, DEFAULT_ADAM_BETA2_VAR_DECAY, DEFAULT_ADAM_EPSILON);
    }

    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        this(learningRate, null, beta1, beta2, epsilon);
    }

    private Adam(@JsonProperty("learningRate") double learningRate,
                 @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule,
                 @JsonProperty("beta1") double beta1,
                 @JsonProperty("beta2") double beta2,
                 @JsonProperty("epsilon") double epsilon){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public long stateSize(long numParams) {
        return 2 * numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        AdamUpdater u = new AdamUpdater(this);
        long[] gradientShape = viewArray.shape();
        gradientShape = Arrays.copyOf(gradientShape, gradientShape.length);
        gradientShape[1] /= 2;
        u.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public Adam clone() {
        return new Adam(learningRate, learningRateSchedule, beta1, beta2, epsilon);
    }

    @Override
    public double getLearningRate(int iteration, int epoch){
        if(learningRateSchedule != null){
            return learningRateSchedule.valueAt(iteration, epoch);
        }
        return learningRate;
    }

    @Override
    public boolean hasLearningRate() {
        return true;
    }

    @Override
    public void setLrAndSchedule(double lr, ISchedule lrSchedule) {
        this.learningRate = lr;
        this.learningRateSchedule = lrSchedule;
    }

    //Partial builder implementation to give public no-arg constructor
    public static class Builder {
        public Builder(){ }
    }
}
