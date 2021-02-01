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
package org.deeplearning4j.rl4j.agent;

import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.policy.IPolicy;

/**
 * The interface of {@link Agent}
 * @param <ACTION>
 */
public interface IAgent<ACTION> {
    /**
     * Will play a single episode
     */
    void run();

    /**
     * @return A user-supplied id to identify the IAgent instance.
     */
    String getId();

    /**
     * @return The {@link Environment} instance being used by the agent.
     */
    Environment<ACTION> getEnvironment();

    /**
     * @return The {@link IPolicy} instance being used by the agent.
     */
    IPolicy<ACTION> getPolicy();

    /**
     * @return The step count taken in the current episode.
     */
    int getEpisodeStepCount();

    /**
     * @return The cumulative reward received in the current episode.
     */
    double getReward();
}
