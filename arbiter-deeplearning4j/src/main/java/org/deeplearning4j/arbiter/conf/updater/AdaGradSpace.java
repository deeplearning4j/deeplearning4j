package org.deeplearning4j.arbiter.conf.updater;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@Data
@EqualsAndHashCode(callSuper = false)
public class AdaGradSpace extends BaseUpdaterSpace {

    private ParameterSpace<Double> learningRate;
    private ParameterSpace<ISchedule> lrSchedule;

    @Getter @Setter
    private int[] indices;

    public AdaGradSpace(ParameterSpace<Double> learningRate) {
        this(learningRate, null);
    }

    public static AdaGradSpace withLR(ParameterSpace<Double> lr){
        return new AdaGradSpace(lr, null);
    }

    public static AdaGradSpace withLRSchedule(ParameterSpace<ISchedule> lrSchedule){
        return new AdaGradSpace(null, lrSchedule);
    }

    protected AdaGradSpace(@JsonProperty("learningRate") ParameterSpace<Double> learningRate,
                           @JsonProperty("lrSchedule") ParameterSpace<ISchedule> lrSchedule){
        this.learningRate = learningRate;
        this.lrSchedule = lrSchedule;
    }

    @Override
    public IUpdater getValue(double[] parameterValues) {
        if(lrSchedule != null){
            return new AdaGrad(lrSchedule.getValue(parameterValues));
        } else {
            return new AdaGrad(learningRate.getValue(parameterValues));
        }
    }
}
