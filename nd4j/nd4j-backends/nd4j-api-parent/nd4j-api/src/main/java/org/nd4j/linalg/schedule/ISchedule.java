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

import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * ISchedule: a general purpose interface for getting values according to some schedule.
 * Used for implementing learning rate, dropout and momentum schedules - and in principle, any univariate (double)
 * value that deponds on the current iteration and epochs numbers.<br>
 * <br>
 * Note: ISchedule objects should not have mutable state - i.e., they should be safe to share between multiple
 * locations/layers.
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface ISchedule extends Serializable, Cloneable {

    /**
     * @param iteration Current iteration number. Starts at 0
     * @param epoch     Current epoch number. Starts at 0
     * @return Value at the current iteration/epoch for this schedule
     */
    double valueAt(int iteration, int epoch);

    ISchedule clone();

}
