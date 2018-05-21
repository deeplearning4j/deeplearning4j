package org.nd4j.linalg.learning.config;

import lombok.Builder;
import lombok.Data;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AMSGradUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * The AMSGrad updater<br>
 * Reference: On the Convergence of Adam and Beyond - https://openreview.net/forum?id=ryQu7f-RZ
 *
 * @author Alex Black
 */
@Data
@Builder(builderClassName = "Builder")
public class AMSGrad implements IUpdater {

    public static final double DEFAULT_AMSGRAD_LEARNING_RATE = 1e-3;
    public static final double DEFAULT_AMSGRAD_EPSILON = 1e-8;
    public static final double DEFAULT_AMSGRAD_BETA1_MEAN_DECAY = 0.9;
    public static final double DEFAULT_AMSGRAD_BETA2_VAR_DECAY = 0.999;

    @lombok.Builder.Default private double learningRate = DEFAULT_AMSGRAD_LEARNING_RATE; // learning rate
    private ISchedule learningRateSchedule;
    @lombok.Builder.Default private double beta1 = DEFAULT_AMSGRAD_BETA1_MEAN_DECAY; // gradient moving avg decay rate
    @lombok.Builder.Default private double beta2 = DEFAULT_AMSGRAD_BETA2_VAR_DECAY; // gradient sqrt decay rate
    @lombok.Builder.Default private double epsilon = DEFAULT_AMSGRAD_EPSILON;

    public AMSGrad() {
        this(DEFAULT_AMSGRAD_LEARNING_RATE, DEFAULT_AMSGRAD_BETA1_MEAN_DECAY, DEFAULT_AMSGRAD_BETA2_VAR_DECAY,
                        DEFAULT_AMSGRAD_EPSILON);
    }

    public AMSGrad(double learningRate){
        this(learningRate, null, DEFAULT_AMSGRAD_BETA1_MEAN_DECAY, DEFAULT_AMSGRAD_BETA2_VAR_DECAY, DEFAULT_AMSGRAD_EPSILON);
    }

    public AMSGrad(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_AMSGRAD_BETA1_MEAN_DECAY, DEFAULT_AMSGRAD_BETA2_VAR_DECAY, DEFAULT_AMSGRAD_EPSILON);
    }

    public AMSGrad(double learningRate, double beta1, double beta2, double epsilon) {
        this(learningRate, null, beta1, beta2, epsilon);
    }

    private AMSGrad(@JsonProperty("learningRate") double learningRate,
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
        return 3 * numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        AMSGradUpdater u = new AMSGradUpdater(this);
        long[] gradientShape = viewArray.shape();
        gradientShape = Arrays.copyOf(gradientShape, gradientShape.length);
        gradientShape[1] /= 3;
        u.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public AMSGrad clone() {
        return new AMSGrad(learningRate, learningRateSchedule, beta1, beta2, epsilon);
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
