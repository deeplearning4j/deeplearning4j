package org.deeplearning4j.arbiter.conf.updater;

import lombok.Data;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
public class NesterovsSpace extends BaseUpdaterSpace {

    protected ParameterSpace<Double> learningRate;
    protected ParameterSpace<ISchedule> learningRateSchedule;
    protected ParameterSpace<Double> momentum;
    protected ParameterSpace<ISchedule> momentumSchedule;

    public NesterovsSpace(ParameterSpace<Double> learningRate) {
        this(learningRate, null);
    }

    public NesterovsSpace(ParameterSpace<Double> learningRate, ParameterSpace<Double> momentum) {
        this(learningRate, null, momentum, null);
    }

    public NesterovsSpace(@JsonProperty("learningRate") ParameterSpace<Double> learningRate,
                          @JsonProperty("learningRateSchedule") ParameterSpace<ISchedule> learningRateSchedule,
                          @JsonProperty("momentum") ParameterSpace<Double> momentum,
                          @JsonProperty("learningRateSchedule") ParameterSpace<ISchedule> momentumSchedule){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.momentum = momentum;
        this.momentumSchedule = momentumSchedule;
    }

    @Override
    public IUpdater getValue(double[] parameterValues) {
        double lr = learningRate == null ? Nesterovs.DEFAULT_NESTEROV_LEARNING_RATE : learningRate.getValue(parameterValues);
        ISchedule lrS = learningRateSchedule == null ? null : learningRateSchedule.getValue(parameterValues);
        double m = momentum == null ? Nesterovs.DEFAULT_NESTEROV_MOMENTUM : momentum.getValue(parameterValues);
        ISchedule mS = momentumSchedule == null ? null : momentumSchedule.getValue(parameterValues);
        if(lrS == null){
            if(momentumSchedule == null){
                return new Nesterovs(lr, m);
            } else {
                return new Nesterovs(lr, mS);
            }
        } else {
            if(momentumSchedule == null){
                return new Nesterovs(lrS, m);
            } else {
                return new Nesterovs(lrS, mS);
            }
        }
    }
}
