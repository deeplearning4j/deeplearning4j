/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.trainer;

import lombok.Getter;
import lombok.NonNull;
import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.IAgentLearner;

import java.util.function.Predicate;

// TODO: Add listeners & events once AsyncTrainer is implemented

/**
 * A {@link ITrainer} implementation that will create a single {@link IAgentLearner} and perform the training in a
 * synchronized setup, until a stopping condition is met.
 *
 * @param <ACTION> The type of the actions expected by the environment
 */
public class SyncTrainer<ACTION> implements ITrainer {

    private final Predicate<SyncTrainer<ACTION>> stoppingCondition;

    @Getter
    private int episodeCount;

    @Getter
    final IAgentLearner<ACTION> agentLearner;

    /**
     * Build a SyncTrainer that will train until a stopping condition is met.
     * @param agentLearnerBuilder the builder that will be used to create the agent-learner instance
     * @param stoppingCondition the training will stop when this condition evaluates to true
     */
    @lombok.Builder
    public SyncTrainer(@NonNull Builder<IAgentLearner<ACTION>> agentLearnerBuilder,
                       @NonNull Predicate<SyncTrainer<ACTION>> stoppingCondition) {
        this.stoppingCondition = stoppingCondition;
        agentLearner = agentLearnerBuilder.build();
    }

    /**
     * Perform the training
     */
    public void train() {
        while (!stoppingCondition.test(this)) {
            agentLearner.run();
            ++episodeCount;
        }
    }
}