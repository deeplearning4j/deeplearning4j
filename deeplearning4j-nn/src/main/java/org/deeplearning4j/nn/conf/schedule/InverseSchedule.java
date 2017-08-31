package org.deeplearning4j.nn.conf.schedule;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
public class InverseSchedule implements ISchedule {

    private final ScheduleType scheduleType;
    private final double initialValue;
    private final double decayRate;
    private final double power;

    public InverseSchedule(@JsonProperty("scheduleType") ScheduleType scheduleType,
                           @JsonProperty("initialValue") double initialValue,
                           @JsonProperty("decayRate") double decayRate,
                           @JsonProperty("power") double power){
        this.scheduleType = scheduleType;
        this.initialValue = initialValue;
        this.decayRate = decayRate;
        this.power = power;
    }

    @Override
    public double valueAt(double currentValue, int iteration, int epoch) {
        int i = (scheduleType == ScheduleType.ITERATION ? iteration : epoch);
        return initialValue / Math.pow(1 + decayRate * i, power);
    }

    @Override
    public ISchedule clone() {
        return new InverseSchedule(scheduleType, initialValue, decayRate, power);
    }
}
