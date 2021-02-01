/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.earlystopping.termination;

import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**Interface for termination conditions to be evaluated once per iteration (i.e., once per minibatch).
 * Used for example to more quickly terminate training, instead of waiting for an epoch to complete before
 * checking termination conditions.
 * */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
@JsonInclude(JsonInclude.Include.NON_NULL)
public interface IterationTerminationCondition extends Serializable {

    /** Initialize the iteration termination condition (sometimes a no-op)*/
    void initialize();

    /** Should early stopping training terminate at this iteration, based on the score for the last iteration?
     * return true if training should be terminated immediately, or false otherwise
     * @param lastMiniBatchScore Score of the last minibatch
     * @return whether to terminate or not
     */
    boolean terminate(double lastMiniBatchScore);

}
