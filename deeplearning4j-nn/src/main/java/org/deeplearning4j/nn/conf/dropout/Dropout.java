package org.deeplearning4j.nn.conf.dropout;

import org.deeplearning4j.nn.conf.schedule.ISchedule;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.DropOutInverted;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonProperty;

public class Dropout implements IDropout {

    private double p;
    private ISchedule pSchedule;

    private double lastPValue;

    public Dropout(double activationRetainProbability) {
        this(activationRetainProbability, null);
    }

    public Dropout(ISchedule activationRetainProbabilitySchedule){
        this(Double.NaN, activationRetainProbabilitySchedule);
    }

    protected Dropout(@JsonProperty("p") double activationRetainProbability, @JsonProperty("pSchedule") ISchedule activationRetainProbabilitySchedule) {
        this.p = activationRetainProbability;
        this.pSchedule = activationRetainProbabilitySchedule;
    }


    @Override
    public INDArray applyDropout(INDArray inputActivations, int iteration, int epoch, boolean inPlace) {
        INDArray result = inPlace ? inputActivations : inputActivations.dup(inputActivations.ordering());

        if(pSchedule != null){
            lastPValue = pSchedule.valueAt(lastPValue, iteration, epoch);
        } else {
            lastPValue = p;
        }

        Nd4j.getExecutioner().exec(new DropOutInverted(result, lastPValue));

        return result;
    }
}
