package org.deeplearning4j.nn.conf.schedule;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
public class ExponentialSchedule implements ISchedule {

    private final ScheduleType scheduleType;
    private final double initialValue;
    private final double decayRate;

    public ExponentialSchedule(@JsonProperty("scheduleType") ScheduleType scheduleType,
                               @JsonProperty("initialValue") double initialValue,
                               @JsonProperty("decayRate") double decayRate){
        this.scheduleType = scheduleType;
        this.initialValue = initialValue;
        this.decayRate = decayRate;
    }


    @Override
    public double valueAt(double currentValue, int iteration, int epoch) {
        int i = (scheduleType == ScheduleType.ITERATION ? iteration : epoch);
        return initialValue * Math.pow(decayRate, i);
    }

    @Override
    public ISchedule clone() {
        return new ExponentialSchedule(scheduleType, initialValue, decayRate);
    }
}
