package org.deeplearning4j.nn.conf.dropout;

import org.deeplearning4j.nn.conf.schedule.ISchedule;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonProperty;

public class GaussianNoise implements IDropout {

    private double stddev;
    private ISchedule stddevSchedule;

    public GaussianNoise(double stddev){
        this(stddev, null);
    }

    public GaussianNoise(ISchedule stddevSchedule){
        this(Double.NaN, stddevSchedule);
    }

    protected GaussianNoise(@JsonProperty("stddev") double stddev, @JsonProperty("stddevSchedule") ISchedule stddevSchedule){
        this.stddev = stddev;
        this.stddevSchedule = stddevSchedule;
    }

    @Override
    public INDArray applyDropout(INDArray inputActivations, int iteration, int epoch, boolean inPlace) {
        double currS;
        if(stddevSchedule != null){
            currS = stddevSchedule.valueAt(iteration, epoch);
        } else {
            currS = stddev;
        }

        INDArray result = inPlace ? inputActivations : inputActivations.dup(inputActivations.ordering());
        INDArray noise = Nd4j.createUninitialized(inputActivations.shape(), inputActivations.ordering());
        Nd4j.getExecutioner().exec(new GaussianDistribution(noise, 0, currS));

        result.muli(noise);

        return result;
    }

    @Override
    public IDropout clone() {
        return new GaussianNoise(stddev, stddevSchedule);
    }
}
