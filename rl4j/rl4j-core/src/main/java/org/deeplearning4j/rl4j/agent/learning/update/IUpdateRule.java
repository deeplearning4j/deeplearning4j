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
package org.deeplearning4j.rl4j.agent.learning.update;

import java.util.List;

/**
 * The role of IUpdateRule implementations is to use an experience batch to improve the accuracy of the policy.
 * Used by {@link org.deeplearning4j.rl4j.agent.AgentLearner AgentLearner}
 * @param <EXPERIENCE_TYPE> The type of the experience
 */
public interface IUpdateRule<EXPERIENCE_TYPE> {
    /**
     * Perform the update
     * @param trainingBatch A batch of experience
     */
    void update(List<EXPERIENCE_TYPE> trainingBatch);

    /**
     * @return The total number of times the policy has been updated. In a multi-agent learning context, this total is
     * for all the agents.
     */
    int getUpdateCount();

    /**
     * Notify the update rule that a new training batch has been started
     */
    void notifyNewBatchStarted();
}
