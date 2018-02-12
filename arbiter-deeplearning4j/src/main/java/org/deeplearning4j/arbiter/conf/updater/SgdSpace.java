package org.deeplearning4j.arbiter.conf.updater;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
@EqualsAndHashCode(callSuper = false)
public class SgdSpace extends BaseUpdaterSpace {

    protected ParameterSpace<Double> learningRate;
    protected ParameterSpace<ISchedule> learningRateSchedule;

    public SgdSpace(ParameterSpace<Double> learningRate) {
        this(learningRate, null);
    }

    public SgdSpace(@JsonProperty("learningRate") ParameterSpace<Double> learningRate,
                    @JsonProperty("learningRateSchedule") ParameterSpace<ISchedule> learningRateSchedule){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
    }

    @Override
    public IUpdater getValue(double[] parameterValues) {
        double lr = learningRate == null ? Sgd.DEFAULT_SGD_LR : learningRate.getValue(parameterValues);
        ISchedule lrS = learningRateSchedule == null ? null : learningRateSchedule.getValue(parameterValues);
        if(lrS == null){
            return new Sgd(lr);
        } else {
            return new Sgd(lrS);
        }
    }
}
