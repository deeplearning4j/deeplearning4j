/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.conf.updater.schedule;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.InverseSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@NoArgsConstructor //JSON
@Data
public class StepScheduleSpace implements ParameterSpace<ISchedule> {

    private ScheduleType scheduleType;
    private ParameterSpace<Double> initialValue;
    private ParameterSpace<Double> decayRate;
    private ParameterSpace<Double> step;

    public StepScheduleSpace(@NonNull ScheduleType scheduleType, @NonNull ParameterSpace<Double> initialValue,
                             double decayRate, double step){
        this(scheduleType, initialValue, new FixedValue<>(decayRate), new FixedValue<>(step));
    }

    public StepScheduleSpace(@NonNull @JsonProperty("scheduleType") ScheduleType scheduleType,
                             @NonNull @JsonProperty("initialValue") ParameterSpace<Double> initialValue,
                             @NonNull @JsonProperty("decayRate") ParameterSpace<Double> decayRate,
                             @NonNull @JsonProperty("step") ParameterSpace<Double> step){
        this.scheduleType = scheduleType;
        this.initialValue = initialValue;
        this.decayRate = decayRate;
        this.step = step;
    }

    @Override
    public ISchedule getValue(double[] parameterValues) {
        return new StepSchedule(scheduleType, initialValue.getValue(parameterValues),
                decayRate.getValue(parameterValues), step.getValue(parameterValues));
    }

    @Override
    public int numParameters() {
        return initialValue.numParameters() + decayRate.numParameters() + step.numParameters();
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Arrays.<ParameterSpace>asList(initialValue, decayRate, step);
    }

    @Override
    public Map<String, ParameterSpace> getNestedSpaces() {
        Map<String,ParameterSpace> out = new LinkedHashMap<>();
        out.put("initialValue", initialValue);
        out.put("decayRate", decayRate);
        out.put("step", step);
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
        if(decayRate.numParameters() > 0){
            int inp = initialValue.numParameters();
            int[] sub = Arrays.copyOfRange(indices, inp, inp + decayRate.numParameters());
            decayRate.setIndices(sub);
        }
        if(step.numParameters() > 0){
            int np = initialValue.numParameters() + decayRate.numParameters();
            int[] sub = Arrays.copyOfRange(indices, np, np + step.numParameters());
            step.setIndices(sub);
        }
    }
}
