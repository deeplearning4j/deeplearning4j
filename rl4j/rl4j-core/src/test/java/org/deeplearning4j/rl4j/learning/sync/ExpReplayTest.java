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

package org.deeplearning4j.rl4j.learning.sync;

import org.deeplearning4j.rl4j.experience.StateActionRewardState;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.support.MockRandom;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.Assert.assertEquals;

public class ExpReplayTest {
    @Test
    public void when_storingElementWithStorageNotFull_expect_elementStored() {
        // Arrange
        MockRandom randomMock = new MockRandom(null, new int[] { 0 });
        ExpReplay<Integer> sut = new ExpReplay<Integer>(2, 1, randomMock);

        // Act
        StateActionRewardState<Integer> stateActionRewardState = buildTransition(buildObservation(),
                123, 234, new Observation(Nd4j.create(1)));
        sut.store(stateActionRewardState);
        List<StateActionRewardState<Integer>> results = sut.getBatch(1);

        // Assert
        assertEquals(1, results.size());
        assertEquals(123, (int)results.get(0).getAction());
        assertEquals(234, (int)results.get(0).getReward());
    }

    @Test
    public void when_storingElementWithStorageFull_expect_oldestElementReplacedByStored() {
        // Arrange
        MockRandom randomMock = new MockRandom(null, new int[] { 0, 1 });
        ExpReplay<Integer> sut = new ExpReplay<Integer>(2, 1, randomMock);

        // Act
        StateActionRewardState<Integer> stateActionRewardState1 = buildTransition(buildObservation(),
                1, 2, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState2 = buildTransition(buildObservation(),
                3, 4, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState3 = buildTransition(buildObservation(),
                5, 6, new Observation(Nd4j.create(1)));
        sut.store(stateActionRewardState1);
        sut.store(stateActionRewardState2);
        sut.store(stateActionRewardState3);
        List<StateActionRewardState<Integer>> results = sut.getBatch(2);

        // Assert
        assertEquals(2, results.size());

        assertEquals(3, (int)results.get(0).getAction());
        assertEquals(4, (int)results.get(0).getReward());

        assertEquals(5, (int)results.get(1).getAction());
        assertEquals(6, (int)results.get(1).getReward());
    }


    @Test
    public void when_askBatchSizeZeroAndStorageEmpty_expect_emptyBatch() {
        // Arrange
        MockRandom randomMock = new MockRandom(null, new int[] { 0 });
        ExpReplay<Integer> sut = new ExpReplay<Integer>(5, 1, randomMock);

        // Act
        List<StateActionRewardState<Integer>> results = sut.getBatch(0);

        // Assert
        assertEquals(0, results.size());
    }

    @Test
    public void when_askBatchSizeZeroAndStorageNotEmpty_expect_emptyBatch() {
        // Arrange
        MockRandom randomMock = new MockRandom(null, new int[] { 0 });
        ExpReplay<Integer> sut = new ExpReplay<Integer>(5, 1, randomMock);

        // Act
        StateActionRewardState<Integer> stateActionRewardState1 = buildTransition(buildObservation(),
                1, 2, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState2 = buildTransition(buildObservation(),
                3, 4, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState3 = buildTransition(buildObservation(),
                5, 6, new Observation(Nd4j.create(1)));
        sut.store(stateActionRewardState1);
        sut.store(stateActionRewardState2);
        sut.store(stateActionRewardState3);
        List<StateActionRewardState<Integer>> results = sut.getBatch(0);

        // Assert
        assertEquals(0, results.size());
    }

    @Test
    public void when_askBatchSizeGreaterThanStoredCount_expect_batchWithStoredCountElements() {
        // Arrange
        MockRandom randomMock = new MockRandom(null, new int[] { 0, 1, 2 });
        ExpReplay<Integer> sut = new ExpReplay<Integer>(5, 1, randomMock);

        // Act
        StateActionRewardState<Integer> stateActionRewardState1 = buildTransition(buildObservation(),
                1, 2, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState2 = buildTransition(buildObservation(),
                3, 4, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState3 = buildTransition(buildObservation(),
                5, 6, new Observation(Nd4j.create(1)));
        sut.store(stateActionRewardState1);
        sut.store(stateActionRewardState2);
        sut.store(stateActionRewardState3);
        List<StateActionRewardState<Integer>> results = sut.getBatch(10);

        // Assert
        assertEquals(3, results.size());

        assertEquals(1, (int)results.get(0).getAction());
        assertEquals(2, (int)results.get(0).getReward());

        assertEquals(3, (int)results.get(1).getAction());
        assertEquals(4, (int)results.get(1).getReward());

        assertEquals(5, (int)results.get(2).getAction());
        assertEquals(6, (int)results.get(2).getReward());
    }

    @Test
    public void when_askBatchSizeSmallerThanStoredCount_expect_batchWithAskedElements() {
        // Arrange
        MockRandom randomMock = new MockRandom(null, new int[] { 0, 1, 2, 3, 4 });
        ExpReplay<Integer> sut = new ExpReplay<Integer>(5, 1, randomMock);

        // Act
        StateActionRewardState<Integer> stateActionRewardState1 = buildTransition(buildObservation(),
                1, 2, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState2 = buildTransition(buildObservation(),
                3, 4, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState3 = buildTransition(buildObservation(),
                5, 6, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState4 = buildTransition(buildObservation(),
                7, 8, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState5 = buildTransition(buildObservation(),
                9, 10, new Observation(Nd4j.create(1)));
        sut.store(stateActionRewardState1);
        sut.store(stateActionRewardState2);
        sut.store(stateActionRewardState3);
        sut.store(stateActionRewardState4);
        sut.store(stateActionRewardState5);
        List<StateActionRewardState<Integer>> results = sut.getBatch(3);

        // Assert
        assertEquals(3, results.size());

        assertEquals(1, (int)results.get(0).getAction());
        assertEquals(2, (int)results.get(0).getReward());

        assertEquals(3, (int)results.get(1).getAction());
        assertEquals(4, (int)results.get(1).getReward());

        assertEquals(5, (int)results.get(2).getAction());
        assertEquals(6, (int)results.get(2).getReward());
    }

    @Test
    public void when_randomGivesDuplicates_expect_noDuplicatesInBatch() {
        // Arrange
        MockRandom randomMock = new MockRandom(null, new int[] { 0, 1, 2, 1, 3, 1, 4 });
        ExpReplay<Integer> sut = new ExpReplay<Integer>(5, 1, randomMock);

        // Act
        StateActionRewardState<Integer> stateActionRewardState1 = buildTransition(buildObservation(),
                1, 2, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState2 = buildTransition(buildObservation(),
                3, 4, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState3 = buildTransition(buildObservation(),
                5, 6, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState4 = buildTransition(buildObservation(),
                7, 8, new Observation(Nd4j.create(1)));
        StateActionRewardState<Integer> stateActionRewardState5 = buildTransition(buildObservation(),
                9, 10, new Observation(Nd4j.create(1)));
        sut.store(stateActionRewardState1);
        sut.store(stateActionRewardState2);
        sut.store(stateActionRewardState3);
        sut.store(stateActionRewardState4);
        sut.store(stateActionRewardState5);
        List<StateActionRewardState<Integer>> results = sut.getBatch(3);

        // Assert
        assertEquals(3, results.size());

        assertEquals(1, (int)results.get(0).getAction());
        assertEquals(2, (int)results.get(0).getReward());

        assertEquals(3, (int)results.get(1).getAction());
        assertEquals(4, (int)results.get(1).getReward());

        assertEquals(5, (int)results.get(2).getAction());
        assertEquals(6, (int)results.get(2).getReward());
    }

    private StateActionRewardState<Integer> buildTransition(Observation observation, Integer action, double reward, Observation nextObservation) {
        StateActionRewardState<Integer> result = new StateActionRewardState<Integer>(observation, action, reward, false);
        result.setNextObservation(nextObservation);

        return result;
    }

    private Observation buildObservation() {
        return new Observation(Nd4j.create(1, 1));
    }
}
