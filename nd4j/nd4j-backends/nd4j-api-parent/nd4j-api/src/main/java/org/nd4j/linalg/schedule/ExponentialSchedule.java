package org.nd4j.linalg.schedule;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * An exponential schedule, with 2 parameters: initial value, and gamma.<br>
 * value(i) = initialValue * gamma^i
 * where i is the iteration or epoch (depending on the setting)
 *
 * @author Alex Black
 */
@Data
public class ExponentialSchedule implements ISchedule {

    private final ScheduleType scheduleType;
    private final double initialValue;
    private final double gamma;

    public ExponentialSchedule(@JsonProperty("scheduleType") ScheduleType scheduleType,
                               @JsonProperty("initialValue") double initialValue,
                               @JsonProperty("gamma") double gamma){
        this.scheduleType = scheduleType;
        this.initialValue = initialValue;
        this.gamma = gamma;
    }


    @Override
    public double valueAt(int iteration, int epoch) {
        int i = (scheduleType == ScheduleType.ITERATION ? iteration : epoch);
        return initialValue * Math.pow(gamma, i);
    }

    @Override
    public ISchedule clone() {
        return new ExponentialSchedule(scheduleType, initialValue, gamma);
    }
}
