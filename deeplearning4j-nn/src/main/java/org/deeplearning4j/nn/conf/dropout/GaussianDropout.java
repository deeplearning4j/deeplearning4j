package org.deeplearning4j.nn.conf.dropout;

import org.deeplearning4j.nn.conf.schedule.ISchedule;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonProperty;

public class GaussianDropout implements IDropout {

    private final double rate;
    private final ISchedule rateSchedule;

    private double lastRate;

    public GaussianDropout(double rate){
        this(rate, null);
    }

    public GaussianDropout(ISchedule rateSchedule){
        this(Double.NaN, rateSchedule);
    }

    protected GaussianDropout(@JsonProperty("rate") double rate, @JsonProperty("rateSchedule") ISchedule rateSchedule){
        this.rate = rate;
        this.rateSchedule = rateSchedule;
    }

    @Override
    public INDArray applyDropout(INDArray inputActivations, int iteration, int epoch, boolean inPlace) {
        if(rateSchedule != null){
            lastRate = rateSchedule.valueAt(iteration, epoch);
        } else {
            lastRate = rate;
        }

        double stdev = Math.sqrt(lastRate / (1.0 - lastRate));

        INDArray noise = Nd4j.createUninitialized(inputActivations.shape(), inputActivations.ordering());
        Nd4j.getExecutioner().execAndReturn(new GaussianDistribution(noise, 1.0, stdev));

        if(inPlace){
            return inputActivations.muli(noise);
        } else {
            INDArray result = Nd4j.createUninitialized(inputActivations.shape(), inputActivations.ordering());
            return Nd4j.getExecutioner().execAndReturn(new MulOp(inputActivations, noise, result));
        }
    }

    @Override
    public GaussianDropout clone() {
        return new GaussianDropout(rate, rateSchedule == null ? null : rateSchedule.clone());
    }
}
