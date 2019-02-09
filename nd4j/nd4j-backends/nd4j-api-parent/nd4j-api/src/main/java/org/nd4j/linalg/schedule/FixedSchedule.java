package org.nd4j.linalg.schedule;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
public class FixedSchedule implements ISchedule{

    private final double value;

    public FixedSchedule(@JsonProperty("value") double value){
        this.value = value;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        return value;
    }

    @Override
    public ISchedule clone() {
        return new FixedSchedule(value);
    }
}
