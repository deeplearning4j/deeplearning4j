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
package org.deeplearning4j.rl4j.experience;

import lombok.Builder;
import lombok.Data;
import lombok.experimental.SuperBuilder;
import org.deeplearning4j.rl4j.observation.Observation;

import java.util.ArrayList;
import java.util.List;

/**
 * A simple {@link ExperienceHandler experience handler} that stores the experiences.
 * Note: Calling {@link StateActionExperienceHandler#generateTrainingBatch() generateTrainingBatch()} will clear the stored experiences
 *
 * @param <A> Action type
 *
 * @author Alexandre Boulanger
 */
public class StateActionExperienceHandler<A> implements ExperienceHandler<A, StateActionReward<A>> {
    private static final int DEFAULT_BATCH_SIZE = 8;

    private final int batchSize;

    private boolean isFinalObservationSet;

    public StateActionExperienceHandler(Configuration configuration) {
        this.batchSize = configuration.getBatchSize();
    }

    private List<StateActionReward<A>> stateActionRewards = new ArrayList<>();

    public void setFinalObservation(Observation observation) {
        isFinalObservationSet = true;
    }

    public void addExperience(Observation observation, A action, double reward, boolean isTerminal) {
        stateActionRewards.add(new StateActionReward<A>(observation, action, reward, isTerminal));
    }

    @Override
    public int getTrainingBatchSize() {
        return stateActionRewards.size();
    }

    @Override
    public boolean isTrainingBatchReady() {
        return stateActionRewards.size() >= batchSize
                || (isFinalObservationSet && stateActionRewards.size() > 0);
    }

    /**
     * The elements are returned in the historical order (i.e. in the order they happened)
     * Note: the experience store is cleared after calling this method.
     *
     * @return The list of experience elements
     */
    @Override
    public List<StateActionReward<A>> generateTrainingBatch() {
        List<StateActionReward<A>> result = stateActionRewards;
        stateActionRewards = new ArrayList<>();

        return result;
    }

    @Override
    public void reset() {
        stateActionRewards = new ArrayList<>();
        isFinalObservationSet = false;
    }

    @SuperBuilder
    @Data
    public static class Configuration {
        /**
         * The default training batch size. Default is 8.
         */
        @Builder.Default
        private int batchSize = DEFAULT_BATCH_SIZE;
    }
}
