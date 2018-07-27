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
