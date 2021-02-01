/*
 *  ******************************************************************************
 *  *
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
package org.deeplearning4j.rl4j.environment;

import java.util.Map;

/**
 * An interface for environments used by the {@link org.deeplearning4j.rl4j.agent.Agent Agents}.
 * @param <ACTION> The type of actions
 */
public interface Environment<ACTION> {

    /**
     * @return The {@link Schema} of the environment
     */
    Schema<ACTION> getSchema();

    /**
     * Reset the environment's state to start a new episode.
     * @return
     */
    Map<String, Object> reset();

    /**
     * Perform a single step.
     *
     * @param action The action taken
     * @return A {@link StepResult} describing the result of the step.
     */
    StepResult step(ACTION action);

    /**
     * @return True if the episode is finished
     */
    boolean isEpisodeFinished();

    /**
     * Called when the agent is finished using this environment instance.
     */
    void close();
}
