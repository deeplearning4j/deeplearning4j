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

package org.deeplearning4j.rl4j.experience;

import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class StateActionExperienceHandlerTest {

    private StateActionExperienceHandler.Configuration buildConfiguration(int batchSize) {
        return StateActionExperienceHandler.Configuration.builder()
                .batchSize(batchSize)
                .build();
    }

    @Test
    public void when_addingExperience_expect_generateTrainingBatchReturnsIt() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler(buildConfiguration(Integer.MAX_VALUE));
        sut.reset();
        Observation observation = new Observation(Nd4j.zeros(1));
        sut.addExperience(observation, 123, 234.0, true);

        // Act
        List<StateActionReward<Integer>> result = sut.generateTrainingBatch();

        // Assert
        assertEquals(1, result.size());
        assertSame(observation, result.get(0).getObservation());
        assertEquals(123, (int)result.get(0).getAction());
        assertEquals(234.0, result.get(0).getReward(), 0.00001);
        assertTrue(result.get(0).isTerminal());
    }

    @Test
    public void when_addingMultipleExperiences_expect_generateTrainingBatchReturnsItInSameOrder() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler(buildConfiguration(Integer.MAX_VALUE));
        sut.reset();
        sut.addExperience(null, 1, 1.0, false);
        sut.addExperience(null, 2, 2.0, false);
        sut.addExperience(null, 3, 3.0, false);

        // Act
        List<StateActionReward<Integer>> result = sut.generateTrainingBatch();

        // Assert
        assertEquals(3, result.size());
        assertEquals(1, (int)result.get(0).getAction());
        assertEquals(2, (int)result.get(1).getAction());
        assertEquals(3, (int)result.get(2).getAction());
    }

    @Test
    public void when_gettingExperience_expect_experienceStoreIsCleared() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler(buildConfiguration(Integer.MAX_VALUE));
        sut.reset();
        sut.addExperience(null, 1, 1.0, false);

        // Act
        List<StateActionReward<Integer>> firstResult = sut.generateTrainingBatch();
        List<StateActionReward<Integer>> secondResult = sut.generateTrainingBatch();

        // Assert
        assertEquals(1, firstResult.size());
        assertEquals(0, secondResult.size());
    }

    @Test
    public void when_addingExperience_expect_getTrainingBatchSizeReturnSize() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler(buildConfiguration(Integer.MAX_VALUE));
        sut.reset();
        sut.addExperience(null, 1, 1.0, false);
        sut.addExperience(null, 2, 2.0, false);
        sut.addExperience(null, 3, 3.0, false);

        // Act
        int size = sut.getTrainingBatchSize();

        // Assert
        assertEquals(3, size);
    }

    @Test
    public void when_experienceIsEmpty_expect_TrainingBatchNotReady() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler(buildConfiguration(5));
        sut.reset();

        // Act
        boolean isTrainingBatchReady = sut.isTrainingBatchReady();

        // Assert
        assertFalse(isTrainingBatchReady);
    }

    @Test
    public void when_experienceSizeIsGreaterOrEqualToThanBatchSize_expect_TrainingBatchIsReady() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler(buildConfiguration(5));
        sut.reset();
        sut.addExperience(null, 1, 1.0, false);
        sut.addExperience(null, 2, 2.0, false);
        sut.addExperience(null, 3, 3.0, false);
        sut.addExperience(null, 4, 4.0, false);
        sut.addExperience(null, 5, 5.0, false);

        // Act
        boolean isTrainingBatchReady = sut.isTrainingBatchReady();

        // Assert
        assertTrue(isTrainingBatchReady);
    }

    @Test
    public void when_experienceSizeIsSmallerThanBatchSizeButFinalObservationIsSet_expect_TrainingBatchIsReady() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler(buildConfiguration(5));
        sut.reset();
        sut.addExperience(null, 1, 1.0, false);
        sut.addExperience(null, 2, 2.0, false);
        sut.setFinalObservation(null);

        // Act
        boolean isTrainingBatchReady = sut.isTrainingBatchReady();

        // Assert
        assertTrue(isTrainingBatchReady);
    }

    @Test
    public void when_experienceSizeIsZeroAndFinalObservationIsSet_expect_TrainingBatchIsNotReady() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler(buildConfiguration(5));
        sut.reset();
        sut.setFinalObservation(null);

        // Act
        boolean isTrainingBatchReady = sut.isTrainingBatchReady();

        // Assert
        assertFalse(isTrainingBatchReady);
    }

}
