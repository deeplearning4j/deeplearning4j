package org.nd4j.linalg.schedule;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 *
 * Polynomial decay schedule, with 3 parameters: initial value, maxIter, power.<br>
 * Note that the the value will be 0 after maxIter, otherwise is given by:
 * value(i) = initialValue * (1 + i/maxIter)^(-power)
 * where i is the iteration or epoch (depending on the setting)
 *
 * @author Alex Black
 */
@Data
public class PolySchedule implements ISchedule {

    private final ScheduleType scheduleType;
    private final double initialValue;
    private final double power;
    private final int maxIter;

    public PolySchedule(@JsonProperty("scheduleType") ScheduleType scheduleType,
                        @JsonProperty("initialValue") double initialValue,
                        @JsonProperty("power") double power,
                        @JsonProperty("maxIter") int maxIter){
        this.scheduleType = scheduleType;
        this.initialValue = initialValue;
        this.power = power;
        this.maxIter = maxIter;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        int i = (scheduleType == ScheduleType.ITERATION ? iteration : epoch);

        if( i >= maxIter ){
            return 0;
        }

        return initialValue * Math.pow(1 + i / (double)maxIter, power);
    }

    @Override
    public PolySchedule clone() {
        return new PolySchedule(scheduleType, initialValue, power, maxIter);
    }

}
