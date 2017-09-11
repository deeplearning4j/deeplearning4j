package org.deeplearning4j.arbiter.conf.updater;

import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.nd4j.linalg.learning.config.AdaMax;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.schedule.ISchedule;

public class AdaMaxSpace extends BaseUpdaterSpace {

    private ParameterSpace<Double> learningRate;
    private ParameterSpace<ISchedule> learningRateSchedule;
    private ParameterSpace<Double> beta1;
    private ParameterSpace<Double> beta2;
    private ParameterSpace<Double> epsilon;

    public AdaMaxSpace(ParameterSpace<Double> learningRate) {
        this(learningRate, null, null, null);
    }

    public AdaMaxSpace(ParameterSpace<Double> learningRate, ParameterSpace<Double> beta1,
                       ParameterSpace<Double> beta2, ParameterSpace<Double> epsilon) {
        this(learningRate, null, beta1, beta2, epsilon);
    }

    public AdaMaxSpace(ParameterSpace<Double> learningRate, ParameterSpace<ISchedule> learningRateSchedule,
                       ParameterSpace<Double> beta1, ParameterSpace<Double> beta2, ParameterSpace<Double> epsilon){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public IUpdater getValue(double[] parameterValues) {
        double lr = learningRate == null ? AdaMax.DEFAULT_ADAMAX_LEARNING_RATE : learningRate.getValue(parameterValues);
        ISchedule lrS = learningRateSchedule == null ? null : learningRateSchedule.getValue(parameterValues);
        double b1 = beta1 == null ? AdaMax.DEFAULT_ADAMAX_LEARNING_RATE : beta1.getValue(parameterValues);
        double b2 = beta2 == null ? AdaMax.DEFAULT_ADAMAX_LEARNING_RATE : beta2.getValue(parameterValues);
        double eps = epsilon == null ? AdaMax.DEFAULT_ADAMAX_LEARNING_RATE : epsilon.getValue(parameterValues);
        if(lrS == null){
            return new AdaMax(lr, b1, b2, eps);
        } else {
            AdaMax a = new AdaMax(lrS);
            a.setBeta1(b1);
            a.setBeta2(b2);
            a.setEpsilon(eps);
            return a;
        }
    }
}
