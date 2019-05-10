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

package org.nd4j.linalg.learning.config;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * IUpdater interface: used for configuration and instantiation of updaters - both built-in and custom.<br>
 * Note that the actual implementations for updaters are in {@link GradientUpdater}
 *
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY, getterVisibility = JsonAutoDetect.Visibility.NONE,
                setterVisibility = JsonAutoDetect.Visibility.NONE)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface IUpdater extends Serializable, Cloneable {

    /**
     * Determine the updater state size for the given number of parameters. Usually a integer multiple (0,1 or 2)
     * times the number of parameters in a layer.
     *
     * @param numParams Number of parameters
     * @return Updater state size for the given number of parameters
     */
    long stateSize(long numParams);

    /**
     * Create a new gradient updater
     *
     * @param viewArray           The updater state size view away
     * @param initializeViewArray If true: initialise the updater state
     * @return
     */
    GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray);

    boolean equals(Object updater);

    /**
     * Clone the updater
     */
    IUpdater clone();

    /**
     * Get the learning rate - if any - for the updater, at the specified iteration and epoch.
     * Note that if no learning rate is applicable (AdaDelta, NoOp updaters etc) then Double.NaN should
     * be return
     *
     * @param iteration Iteration at which to get the learning rate
     * @param epoch     Epoch at which to get the learning rate
     * @return          Learning rate, or Double.NaN if no learning rate is applicable for this updater
     */
    double getLearningRate(int iteration, int epoch);

    /**
     * @return True if the updater has a learning rate hyperparameter, false otherwise
     */
    boolean hasLearningRate();

    /**
     * Set the learning rate and schedule. Note: may throw an exception if {@link #hasLearningRate()} returns false.
     * @param lr         Learning rate to set (typically not used if LR schedule is non-null)
     * @param lrSchedule Learning rate schedule to set (may be null)
     */
    void setLrAndSchedule(double lr, ISchedule lrSchedule);



}
