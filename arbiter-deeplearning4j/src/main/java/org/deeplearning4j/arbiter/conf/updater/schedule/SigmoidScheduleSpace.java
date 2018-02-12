package org.deeplearning4j.arbiter.conf.updater.schedule;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.PolySchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.SigmoidSchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@NoArgsConstructor //JSON
@Data
public class SigmoidScheduleSpace implements ParameterSpace<ISchedule> {

    private ScheduleType scheduleType;
    private ParameterSpace<Double> initialValue;
    private ParameterSpace<Double> gamma;
    private ParameterSpace<Integer> stepSize;

    public SigmoidScheduleSpace(@NonNull ScheduleType scheduleType, @NonNull ParameterSpace<Double> initialValue,
                                double gamma, int stepSize){
        this(scheduleType, initialValue, new FixedValue<>(gamma), new FixedValue<>(stepSize));
    }

    public SigmoidScheduleSpace(@NonNull @JsonProperty("scheduleType") ScheduleType scheduleType,
                                @NonNull @JsonProperty("initialValue") ParameterSpace<Double> initialValue,
                                @NonNull @JsonProperty("gamma") ParameterSpace<Double> gamma,
                                @NonNull @JsonProperty("stepSize") ParameterSpace<Integer> stepSize){
        this.scheduleType = scheduleType;
        this.initialValue = initialValue;
        this.gamma = gamma;
        this.stepSize = stepSize;
    }

    @Override
    public ISchedule getValue(double[] parameterValues) {
        return new SigmoidSchedule(scheduleType, initialValue.getValue(parameterValues),
                gamma.getValue(parameterValues), stepSize.getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return initialValue.numParameters() + gamma.numParameters() + stepSize.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Arrays.<ParameterSpace>asList(initialValue, gamma, stepSize);
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        Map<String,ParameterSpace> out = new LinkedHashMap<>();
        out.put("initialValue", initialValue);
        out.put("gamma", gamma);
        out.put("stepSize", stepSize);
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
        if(gamma.numParameters() > 0){
            int np = initialValue.numParameters();
            int[] sub = Arrays.copyOfRange(indices, np, np + gamma.numParameters());
            gamma.setIndices(sub);
        }
        if(stepSize.numParameters() > 0){
            int np = initialValue.numParameters() + gamma.numParameters();
            int[] sub = Arrays.copyOfRange(indices, np, np + stepSize.numParameters());
            stepSize.setIndices(sub);
        }
    }
}
