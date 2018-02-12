package org.deeplearning4j.arbiter.conf.updater.schedule;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.InverseSchedule;
import org.nd4j.linalg.schedule.PolySchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@NoArgsConstructor //JSON
@Data
public class PolyScheduleSpace implements ParameterSpace<ISchedule> {

    private ScheduleType scheduleType;
    private ParameterSpace<Double> initialValue;
    private ParameterSpace<Double> power;
    private ParameterSpace<Integer> maxIter;

    public PolyScheduleSpace(@NonNull ScheduleType scheduleType, @NonNull ParameterSpace<Double> initialValue,
                             double power, int maxIter){
        this(scheduleType, initialValue, new FixedValue<>(power), new FixedValue<>(maxIter));
    }

    public PolyScheduleSpace(@NonNull @JsonProperty("scheduleType") ScheduleType scheduleType,
                             @NonNull @JsonProperty("initialValue") ParameterSpace<Double> initialValue,
                             @NonNull @JsonProperty("power") ParameterSpace<Double> power,
                             @NonNull @JsonProperty("maxIter") ParameterSpace<Integer> maxIter){
        this.scheduleType = scheduleType;
        this.initialValue = initialValue;
        this.power = power;
        this.maxIter = maxIter;
    }

    @Override
    public ISchedule getValue(double[] parameterValues) {
        return new PolySchedule(scheduleType, initialValue.getValue(parameterValues),
                power.getValue(parameterValues), maxIter.getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return initialValue.numParameters() + power.numParameters() + maxIter.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Arrays.<ParameterSpace>asList(initialValue, power, maxIter);
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        Map<String,ParameterSpace> out = new LinkedHashMap<>();
        out.put("initialValue", initialValue);
        out.put("power", power);
        out.put("maxIter", maxIter);
        return out;
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        if(initialValue.numParameters() > 0){
            int[] sub = Arrays.copyOfRange(indices, 0, initialValue.numParameters());
            initialValue.setIndices(sub);
        }
        if(power.numParameters() > 0){
            int np = initialValue.numParameters();
            int[] sub = Arrays.copyOfRange(indices, np, np + power.numParameters());
            power.setIndices(sub);
        }
        if(maxIter.numParameters() > 0){
            int np = initialValue.numParameters() + power.numParameters();
            int[] sub = Arrays.copyOfRange(indices, np, np + maxIter.numParameters());
            maxIter.setIndices(sub);
        }
    }
}
