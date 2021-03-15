/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.agent.learning.algorithm;

import java.util.List;

public interface IUpdateAlgorithm<RESULT_TYPE, EXPERIENCE_TYPE> {
    /**
     * Compute the labels required to update the network from the training batch
     * @param trainingBatch The transitions from the experience replay
     * @return A DataSet where every element is the observation and the estimated Q-Values for all actions
     */
    RESULT_TYPE compute(List<EXPERIENCE_TYPE> trainingBatch);
}
