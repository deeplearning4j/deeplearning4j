package org.nd4j.linalg.learning.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGradUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Vectorized Learning Rate used per Connection Weight
 * <p/>
 * Adapted from: http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
 * See also http://cs231n.github.io/neural-networks-3/#ada
 *
 * @author Adam Gibson
 */
@Data
@Builder(builderClassName = "Builder")
public class AdaGrad implements IUpdater {

    public static final double DEFAULT_ADAGRAD_LEARNING_RATE = 1e-1;
    public static final double DEFAULT_ADAGRAD_EPSILON = 1e-6;

    @lombok.Builder.Default private double learningRate = DEFAULT_ADAGRAD_LEARNING_RATE;
    private ISchedule learningRateSchedule;
    @lombok.Builder.Default private double epsilon = DEFAULT_ADAGRAD_EPSILON;

    public AdaGrad(){
        this(DEFAULT_ADAGRAD_LEARNING_RATE, null, DEFAULT_ADAGRAD_EPSILON);
    }

    public AdaGrad(double learningRate){
        this(learningRate, null, DEFAULT_ADAGRAD_EPSILON);
    }

    public AdaGrad(double learningRate, double epsilon){
        this(learningRate, null, epsilon);
    }

    public AdaGrad(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_ADAGRAD_EPSILON);
    }

    public AdaGrad(ISchedule learningRateSchedule, double epsilon){
        this(Double.NaN, learningRateSchedule, epsilon);
    }

    private AdaGrad(@JsonProperty("learningRate") double learningRate,
                    @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule,
                    @JsonProperty("epsilon") double epsilon){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.epsilon = epsilon;
    }

    @Override
    public long stateSize(long numParams) {
        return numParams;
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
