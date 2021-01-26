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
package org.deeplearning4j.rl4j.experience;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;
import org.deeplearning4j.rl4j.learning.sync.ExpReplay;
import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.rng.Random;

import java.util.List;

/**
 * A experience handler that stores the experience in a replay memory. See https://arxiv.org/abs/1312.5602
 * The experience container is a {@link StateActionRewardState Transition} that stores the tuple observation-action-reward-nextObservation,
 * as well as whether or the not the episode ended after the Transition
 *
 * @param <A> Action type
 */
@EqualsAndHashCode
public class ReplayMemoryExperienceHandler<A> implements ExperienceHandler<A, StateActionRewardState<A>> {
    private static final int DEFAULT_MAX_REPLAY_MEMORY_SIZE = 150000;
    private static final int DEFAULT_BATCH_SIZE = 32;
    private final int batchSize;

    private IExpReplay<A> expReplay;

    private StateActionRewardState<A> pendingStateActionRewardState;

    public ReplayMemoryExperienceHandler(IExpReplay<A> expReplay) {
        this.expReplay = expReplay;
        this.batchSize = expReplay.getDesignatedBatchSize();
    }

    public ReplayMemoryExperienceHandler(Configuration configuration, Random random) {
        this(new ExpReplay<A>(configuration.maxReplayMemorySize, configuration.batchSize, random));
    }

    public void addExperience(Observation observation, A action, double reward, boolean isTerminal) {
        setNextObservationOnPending(observation);
        pendingStateActionRewardState = new StateActionRewardState<>(observation, action, reward, isTerminal);
    }

    public void setFinalObservation(Observation observation) {
        setNextObservationOnPending(observation);
        pendingStateActionRewardState = null;
    }

    @Override
    public int getTrainingBatchSize() {
        return expReplay.getBatchSize();
    }

    @Override
    public boolean isTrainingBatchReady() {
        return expReplay.getBatchSize() >= batchSize;
    }

    /**
     * @return A batch of experience selected from the replay memory. The replay memory is unchanged after the call.
     */
    @Override
    public List<StateActionRewardState<A>> generateTrainingBatch() {
        return expReplay.getBatch();
    }

    @Override
    public void reset() {
        pendingStateActionRewardState = null;
    }

    private void setNextObservationOnPending(Observation observation) {
        if(pendingStateActionRewardState != null) {
            pendingStateActionRewardState.setNextObservation(observation);
            expReplay.store(pendingStateActionRewardState);
        }
    }

    @SuperBuilder
    @Data
    public static class Configuration {
        /**
         * The maximum replay memory size. Default is 150000
         */
        @Builder.Default
        private int maxReplayMemorySize = DEFAULT_MAX_REPLAY_MEMORY_SIZE;

        /**
         * The size of training batches. Default is 32.
         */
        @Builder.Default
        private int batchSize = DEFAULT_BATCH_SIZE;
    }
}
