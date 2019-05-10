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
