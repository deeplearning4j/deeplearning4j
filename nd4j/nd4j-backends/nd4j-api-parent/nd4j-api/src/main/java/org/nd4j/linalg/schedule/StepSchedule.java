package org.nd4j.linalg.schedule;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Step decay schedule, with 3 parameters: initial value, gamma and step.<br>
 * value(i) = initialValue * gamma^( floor(iter/step) )
 * where i is the iteration or epoch (depending on the setting)
 *
 * @author Alex Black
 */
@Data
public class StepSchedule implements ISchedule {

    private final ScheduleType scheduleType;
    private final double initialValue;
    private final double decayRate;
    private final double step;

    public StepSchedule(@JsonProperty("scheduleType") ScheduleType scheduleType,
                           @JsonProperty("initialValue") double initialValue,
                           @JsonProperty("decayRate") double decayRate,
                           @JsonProperty("step") double step){
        this.scheduleType = scheduleType;
        this.initialValue = initialValue;
        this.decayRate = decayRate;
        this.step = step;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        int i = (scheduleType == ScheduleType.ITERATION ? iteration : epoch);
        return initialValue * Math.pow(decayRate, Math.floor(i / step));
    }

    @Override
    public ISchedule clone() {
        return new StepSchedule(scheduleType, initialValue, decayRate, step);
    }

}
