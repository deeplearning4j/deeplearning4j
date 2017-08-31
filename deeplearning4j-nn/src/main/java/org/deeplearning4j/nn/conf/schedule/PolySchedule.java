package org.deeplearning4j.nn.conf.schedule;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
public class PolySchedule implements ISchedule {

    private final ScheduleType scheduleType;
    private final double initialValue;
    private final double power;

    public PolySchedule(@JsonProperty("scheduleType") ScheduleType scheduleType,
                        @JsonProperty("initialValue") double initialValue,
                        @JsonProperty("power") double power){
        this.scheduleType = scheduleType;
        this.initialValue = initialValue;
        this.power = power;
    }

    @Override
    public double valueAt(double currentValue, int iteration, int epoch) {
        int i = (scheduleType == ScheduleType.ITERATION ? iteration : epoch);
//        return initialValue * Math.pow((1 - i), power);
        throw new UnsupportedOperationException("Needs to be fixed");
    }

    @Override
    public PolySchedule clone() {
        return new PolySchedule(scheduleType, initialValue, power);
    }

}
