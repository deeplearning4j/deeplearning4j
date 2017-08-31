package org.deeplearning4j.nn.conf.schedule;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
public class StepSchedule implements ISchedule {

    private final ScheduleType scheduleType;
    private final double initialValue;
    private final double decayRate;
    private final double steps;

    public StepSchedule(@JsonProperty("scheduleType") ScheduleType scheduleType,
                           @JsonProperty("initialValue") double initialValue,
                           @JsonProperty("decayRate") double decayRate,
                           @JsonProperty("steps") double steps){
        this.scheduleType = scheduleType;
        this.initialValue = initialValue;
        this.decayRate = decayRate;
        this.steps = steps;
    }

    @Override
    public double valueAt(double currentValue, int iteration, int epoch) {
        int i = (scheduleType == ScheduleType.ITERATION ? iteration : epoch);
        return initialValue * Math.pow(decayRate, Math.floor(i / steps));
    }

    @Override
    public ISchedule clone() {
        return new StepSchedule(scheduleType, initialValue, decayRate, steps);
    }

}
