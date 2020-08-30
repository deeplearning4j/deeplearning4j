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
package org.deeplearning4j.rl4j.agent.learning.behavior;

import lombok.Builder;
import lombok.NonNull;
import org.deeplearning4j.rl4j.agent.learning.update.IUpdateRule;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.observation.Observation;

/**
 * A generic {@link ILearningBehavior} that delegates the handling of experience to a {@link ExperienceHandler} and
 * the update logic to a {@link IUpdateRule}
 *
 * @param <ACTION> The type of the action
 * @param <EXPERIENCE_TYPE> The type of experience the ExperienceHandler needs
 */
@Builder
public class LearningBehavior<ACTION, EXPERIENCE_TYPE> implements ILearningBehavior<ACTION> {

    @NonNull
    private final ExperienceHandler<ACTION, EXPERIENCE_TYPE> experienceHandler;

    @NonNull
    private final IUpdateRule<EXPERIENCE_TYPE> updateRule;

    @Override
    public void handleEpisodeStart() {
        experienceHandler.reset();
    }

    @Override
    public void handleNewExperience(Observation observation, ACTION action, double reward, boolean isTerminal) {
        experienceHandler.addExperience(observation, action, reward, isTerminal);
        if(experienceHandler.isTrainingBatchReady()) {
            updateRule.update(experienceHandler.generateTrainingBatch());
        }
    }

    @Override
    public void handleEpisodeEnd(Observation finalObservation) {
        experienceHandler.setFinalObservation(finalObservation);
        if(experienceHandler.isTrainingBatchReady()) {
            updateRule.update(experienceHandler.generateTrainingBatch());
        }
    }
}
