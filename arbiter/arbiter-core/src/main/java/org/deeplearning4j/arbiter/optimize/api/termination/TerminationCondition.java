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

package org.deeplearning4j.arbiter.optimize.api.termination;


import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

/**
 * Global termination condition for conducting hyperparameter optimization.
 * Termination conditions are used to determine if/when the optimization should stop.
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
@JsonInclude(JsonInclude.Include.NON_NULL)
public interface TerminationCondition {

    /**
     * Initialize the termination condition (such as starting timers, etc).
     */
    void initialize(IOptimizationRunner optimizationRunner);

    /**
     * Determine whether optimization should be terminated
     *
     * @param optimizationRunner Optimization runner
     * @return true if learning should be terminated, false otherwise
     */
    boolean terminate(IOptimizationRunner optimizationRunner);

}
