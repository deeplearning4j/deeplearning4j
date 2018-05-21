package org.deeplearning4j.nn.conf.dropout;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Applies additive, mean-zero Gaussian noise to the input - i.e., x = x + N(0,stddev).<br>
 * Note that this differs from {@link GaussianDropout}, which applies <it>multiplicative</it> mean-1 N(1,s) noise.<br>
 * Note also that schedules for the standard deviation value can also be used.
 *
 * @author Alex Black
 */
@Data
public class GaussianNoise implements IDropout {

    private double stddev;
    private ISchedule stddevSchedule;

    /**
     * @param stddev Standard deviation for the mean 0 Gaussian noise
     */
    public GaussianNoise(double stddev){
        this(stddev, null);
    }

    /**
     * @param stddevSchedule Schedule for standard deviation for the mean 0 Gaussian noise
     */
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

        result.addi(noise);

        return result;
    }

    @Override
    public IDropout clone() {
        return new GaussianNoise(stddev, stddevSchedule);
    }
}
