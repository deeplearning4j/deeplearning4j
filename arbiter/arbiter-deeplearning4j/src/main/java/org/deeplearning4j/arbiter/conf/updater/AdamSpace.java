package org.deeplearning4j.arbiter.conf.updater;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
@EqualsAndHashCode(callSuper = false)
public class AdamSpace extends BaseUpdaterSpace {

    private ParameterSpace<Double> learningRate;
    private ParameterSpace<ISchedule> learningRateSchedule;
    private ParameterSpace<Double> beta1;
    private ParameterSpace<Double> beta2;
    private ParameterSpace<Double> epsilon;

    public AdamSpace(ParameterSpace<Double> learningRate) {
        this(learningRate, null, null, null);
    }

    public AdamSpace(ParameterSpace<Double> learningRate, ParameterSpace<Double> beta1,
                     ParameterSpace<Double> beta2, ParameterSpace<Double> epsilon) {
        this(learningRate, null, beta1, beta2, epsilon);
    }

    public static AdamSpace withLR(ParameterSpace<Double> lr){
        return new AdamSpace(lr, null, null, null, null);
    }

    public static AdamSpace withLRSchedule(ParameterSpace<ISchedule> lrSchedule){
        return new AdamSpace(null, lrSchedule, null, null, null);
    }

    protected AdamSpace(@JsonProperty("learningRate") ParameterSpace<Double> learningRate,
                     @JsonProperty("learningRateSchedule") ParameterSpace<ISchedule> learningRateSchedule,
                     @JsonProperty("beta1") ParameterSpace<Double> beta1,
                     @JsonProperty("beta2") ParameterSpace<Double> beta2,
                     @JsonProperty("epsilon") ParameterSpace<Double> epsilon){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public IUpdater getValue(double[] parameterValues) {
        double lr = learningRate == null ? Adam.DEFAULT_ADAM_LEARNING_RATE : learningRate.getValue(parameterValues);
        ISchedule lrS = learningRateSchedule == null ? null : learningRateSchedule.getValue(parameterValues);
        double b1 = beta1 == null ? Adam.DEFAULT_ADAM_LEARNING_RATE : beta1.getValue(parameterValues);
        double b2 = beta2 == null ? Adam.DEFAULT_ADAM_LEARNING_RATE : beta2.getValue(parameterValues);
        double eps = epsilon == null ? Adam.DEFAULT_ADAM_LEARNING_RATE : epsilon.getValue(parameterValues);
        if(lrS == null){
            return new Adam(lr, b1, b2, eps);
        } else {
            Adam a = new Adam(lrS);
            a.setBeta1(b1);
            a.setBeta2(b2);
            a.setEpsilon(eps);
            return a;
        }
    }
}
