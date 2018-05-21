package org.nd4j.linalg.schedule;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Sigmoid decay schedule, with 3 parameters: initial value, gamma and stepSize.<br>
 * value(i) = initialValue * 1.0 / (1 + exp(-gamma * (iter - stepSize)))
 * where i is the iteration or epoch (depending on the setting)
 *
 * @author Alex Black
 */
@Data
public class SigmoidSchedule implements ISchedule {

    private final ScheduleType scheduleType;
    private final double initialValue;
    private final double gamma;
    private final int stepSize;

    public SigmoidSchedule(@JsonProperty("scheduleType") ScheduleType scheduleType,
                           @JsonProperty("initialValue") double initialValue,
                           @JsonProperty("gamma") double gamma,
                           @JsonProperty("stepSize") int stepSize){
        this.scheduleType = scheduleType;
        this.initialValue = initialValue;
        this.gamma = gamma;
        this.stepSize = stepSize;
    }


    @Override
    public double valueAt(int iteration, int epoch) {
        int i = (scheduleType == ScheduleType.ITERATION ? iteration : epoch);
        return initialValue / (1.0 + Math.exp(-gamma * (i - stepSize)));
    }

    @Override
    public ISchedule clone() {
        return new SigmoidSchedule(scheduleType, initialValue, gamma, stepSize);
    }
}
