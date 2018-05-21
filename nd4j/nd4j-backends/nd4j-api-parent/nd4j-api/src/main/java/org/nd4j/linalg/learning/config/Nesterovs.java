package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.NesterovsUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

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

    @lombok.Builder.Default private double learningRate = DEFAULT_NESTEROV_LEARNING_RATE;
    private ISchedule learningRateSchedule;
    @lombok.Builder.Default private double momentum = DEFAULT_NESTEROV_MOMENTUM;
    private ISchedule momentumISchedule;
    @Deprecated
    private Map<Integer,Double> momentumSchedule;

    public Nesterovs(){
        this(DEFAULT_NESTEROV_LEARNING_RATE, null, DEFAULT_NESTEROV_MOMENTUM, null);
    }

    public Nesterovs(double momentum) {
        this(DEFAULT_NESTEROV_LEARNING_RATE, momentum);
    }

    public Nesterovs(double learningRate, double momentum){
        this(learningRate, null, momentum, null);
    }

    public Nesterovs(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_NESTEROV_MOMENTUM, null);
    }

    public Nesterovs(ISchedule learningRateSchedule, double momentum){
        this(Double.NaN, learningRateSchedule, momentum, null);
    }

    public Nesterovs(ISchedule learningRateSchedule, ISchedule momentumSchedule){
        this(Double.NaN, learningRateSchedule, Double.NaN, momentumSchedule);
    }

    public Nesterovs(double learningRate, ISchedule momentumSchedule){
        this(learningRate, null, Double.NaN, momentumSchedule);
    }

    private Nesterovs(@JsonProperty("learningRate") double learningRate,
                      @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule,
                      @JsonProperty("momentum") double momentum,
                      @JsonProperty("momentumSchedule") ISchedule momentumISchedule){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.momentum = momentum;
        this.momentumISchedule = momentumISchedule;
    }

    @Override
    public long stateSize(long numParams) {
        return numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        NesterovsUpdater u = new NesterovsUpdater(this);
        u.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public Nesterovs clone() {
        return new Nesterovs(learningRate, learningRateSchedule, momentum, momentumISchedule);
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

    public double currentMomentum(int iteration, int epoch){
        if(momentumISchedule != null){
            return momentumISchedule.valueAt(iteration, epoch);
        }
        return momentum;
    }

    //Partial builder implementation to give public no-arg constructor
    public static class Builder {
        public Builder(){ }
    }
}
