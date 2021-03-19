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

import org.deeplearning4j.rl4j.experience.StateActionReward;
import org.deeplearning4j.rl4j.experience.StateActionRewardState;
import org.deeplearning4j.rl4j.observation.IObservationSource;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

@RunWith(MockitoJUnitRunner.class)
@Disabled("mockito")
public class FeaturesBuilderTest {

    @Test
    public void when_creatingFeaturesWithObservationSourceAndNonRecurrent_expect_correctlyShapedFeatures() {
        // Arrange
        List<IObservationSource> trainingBatch = new ArrayList<IObservationSource>();
        Observation observation1 = new Observation(new INDArray[] {
                Nd4j.create(new double[] { 1.0 }).reshape(1, 1),
                Nd4j.create(new double[] { 2.0, 3.0 }).reshape(1, 2),
        });
        trainingBatch.add(new StateActionReward<Integer>(observation1, 0, 0.0, false));
        Observation observation2 = new Observation(new INDArray[] {
                Nd4j.create(new double[] { 4.0 }).reshape(1, 1),
                Nd4j.create(new double[] { 5.0, 6.0 }).reshape(1, 2),
        });
        trainingBatch.add(new StateActionReward<Integer>(observation2, 0, 0.0, false));
        FeaturesBuilder sut = new FeaturesBuilder(false);

        // Act
        Features result = sut.build(trainingBatch);

        // Assert
        assertEquals(2, result.getBatchSize());
        assertEquals(1.0, result.get(0).getDouble(0, 0), 0.00001);
        assertEquals(4.0, result.get(0).getDouble(1, 0), 0.00001);

        assertEquals(2.0, result.get(1).getDouble(0, 0), 0.00001);
        assertEquals(3.0, result.get(1).getDouble(0, 1), 0.00001);
        assertEquals(5.0, result.get(1).getDouble(1, 0), 0.00001);
        assertEquals(6.0, result.get(1).getDouble(1, 1), 0.00001);
    }

    @Test
    public void when_creatingFeaturesWithStreamAndNonRecurrent_expect_correctlyShapedFeatures() {
        // Arrange
        List<StateActionRewardState<Integer>> trainingBatch = new ArrayList<StateActionRewardState<Integer>>();
        Observation observation1 = new Observation(new INDArray[] {
                Nd4j.create(new double[] { 1.0 }).reshape(1, 1),
                Nd4j.create(new double[] { 2.0, 3.0 }).reshape(1, 2),
        });
        StateActionRewardState<Integer> stateActionRewardState1 = new StateActionRewardState<Integer>(null, 0, 0.0, false);
        stateActionRewardState1.setNextObservation(observation1);
        trainingBatch.add(stateActionRewardState1);
        Observation observation2 = new Observation(new INDArray[] {
                Nd4j.create(new double[] { 4.0 }).reshape(1, 1),
                Nd4j.create(new double[] { 5.0, 6.0 }).reshape(1, 2),
        });
        StateActionRewardState<Integer> stateActionRewardState2 = new StateActionRewardState<Integer>(null, 0, 0.0, false);
        stateActionRewardState2.setNextObservation(observation2);
        trainingBatch.add(stateActionRewardState2);

        FeaturesBuilder sut = new FeaturesBuilder(false);

        // Act
        Features result = sut.build(trainingBatch.stream().map(e -> e.getNextObservation()), trainingBatch.size());

        // Assert
        assertEquals(2, result.getBatchSize());
        assertEquals(1.0, result.get(0).getDouble(0, 0), 0.00001);
        assertEquals(4.0, result.get(0).getDouble(1, 0), 0.00001);

        assertEquals(2.0, result.get(1).getDouble(0, 0), 0.00001);
        assertEquals(3.0, result.get(1).getDouble(0, 1), 0.00001);
        assertEquals(5.0, result.get(1).getDouble(1, 0), 0.00001);
        assertEquals(6.0, result.get(1).getDouble(1, 1), 0.00001);
    }

    @Test
    public void when_creatingFeaturesWithObservationSourceAndRecurrent_expect_correctlyShapedFeatures() {
        // Arrange
        List<IObservationSource> trainingBatch = new ArrayList<IObservationSource>();
        Observation observation1 = new Observation(new INDArray[] {
                Nd4j.create(new double[] { 1.0, 2.0 }).reshape(1, 2, 1),
                Nd4j.create(new double[] { 3.0, 4.0, 5.0, 6.0 }).reshape(1, 2, 2, 1),
        });
        trainingBatch.add(new StateActionReward<Integer>(observation1, 0, 0.0, false));
        Observation observation2 = new Observation(new INDArray[] {
                Nd4j.create(new double[] { 7.0, 8.0 }).reshape(1, 2, 1),
                Nd4j.create(new double[] { 9.0, 10.0, 11.0, 12.0 }).reshape(1, 2, 2, 1),
        });
        trainingBatch.add(new StateActionReward<Integer>(observation2, 0, 0.0, false));

        FeaturesBuilder sut = new FeaturesBuilder(true);

        // Act
        Features result = sut.build(trainingBatch);

        // Assert
        assertEquals(1, result.getBatchSize()); // With recurrent, batch size is always 1; examples are stacked on the time-serie dimension
        assertArrayEquals(new long[] { 1, 2, 2 }, result.get(0).shape());
        assertEquals(1.0, result.get(0).getDouble(0, 0, 0), 0.00001);
        assertEquals(7.0, result.get(0).getDouble(0, 0, 1), 0.00001);
        assertEquals(2.0, result.get(0).getDouble(0, 1, 0), 0.00001);
        assertEquals(8.0, result.get(0).getDouble(0, 1, 1), 0.00001);

        assertArrayEquals(new long[] { 1, 2, 2, 2 }, result.get(1).shape());
        assertEquals(3.0, result.get(1).getDouble(0, 0, 0, 0), 0.00001);
        assertEquals(9.0, result.get(1).getDouble(0, 0, 0, 1), 0.00001);
        assertEquals(4.0, result.get(1).getDouble(0, 0, 1, 0), 0.00001);
        assertEquals(10.0, result.get(1).getDouble(0, 0, 1, 1), 0.00001);

        assertEquals(5.0, result.get(1).getDouble(0, 1, 0, 0), 0.00001);
        assertEquals(11.0, result.get(1).getDouble(0, 1, 0, 1), 0.00001);
        assertEquals(6.0, result.get(1).getDouble(0, 1, 1, 0), 0.00001);
        assertEquals(12.0, result.get(1).getDouble(0, 1, 1, 1), 0.00001);
    }

    @Test
    public void when_creatingFeaturesWithStreamAndRecurrent_expect_correctlyShapedFeatures() {
        // Arrange
        List<StateActionRewardState<Integer>> trainingBatch = new ArrayList<StateActionRewardState<Integer>>();
        Observation observation1 = new Observation(new INDArray[] {
                Nd4j.create(new double[] { 1.0, 2.0 }).reshape(1, 2, 1),
                Nd4j.create(new double[] { 3.0, 4.0, 5.0, 6.0 }).reshape(1, 2, 2, 1),
        });
        StateActionRewardState<Integer> stateActionRewardState1 = new StateActionRewardState<Integer>(null, 0, 0.0, false);
        stateActionRewardState1.setNextObservation(observation1);
        trainingBatch.add(stateActionRewardState1);
        Observation observation2 = new Observation(new INDArray[] {
                Nd4j.create(new double[] { 7.0, 8.0 }).reshape(1, 2, 1),
                Nd4j.create(new double[] { 9.0, 10.0, 11.0, 12.0 }).reshape(1, 2, 2, 1),
        });
        StateActionRewardState<Integer> stateActionRewardState2 = new StateActionRewardState<Integer>(null, 0, 0.0, false);
        stateActionRewardState2.setNextObservation(observation2);
        trainingBatch.add(stateActionRewardState2);

        FeaturesBuilder sut = new FeaturesBuilder(true);

        // Act
        Features result = sut.build(trainingBatch.stream().map(e -> e.getNextObservation()), trainingBatch.size());

        // Assert
        assertEquals(1, result.getBatchSize()); // With recurrent, batch size is always 1; examples are stacked on the time-serie dimension
        assertArrayEquals(new long[] { 1, 2, 2 }, result.get(0).shape());
        assertEquals(1.0, result.get(0).getDouble(0, 0, 0), 0.00001);
        assertEquals(7.0, result.get(0).getDouble(0, 0, 1), 0.00001);
        assertEquals(2.0, result.get(0).getDouble(0, 1, 0), 0.00001);
        assertEquals(8.0, result.get(0).getDouble(0, 1, 1), 0.00001);

        assertArrayEquals(new long[] { 1, 2, 2, 2 }, result.get(1).shape());
        assertEquals(3.0, result.get(1).getDouble(0, 0, 0, 0), 0.00001);
        assertEquals(9.0, result.get(1).getDouble(0, 0, 0, 1), 0.00001);
        assertEquals(4.0, result.get(1).getDouble(0, 0, 1, 0), 0.00001);
        assertEquals(10.0, result.get(1).getDouble(0, 0, 1, 1), 0.00001);

        assertEquals(5.0, result.get(1).getDouble(0, 1, 0, 0), 0.00001);
        assertEquals(11.0, result.get(1).getDouble(0, 1, 0, 1), 0.00001);
        assertEquals(6.0, result.get(1).getDouble(0, 1, 1, 0), 0.00001);
        assertEquals(12.0, result.get(1).getDouble(0, 1, 1, 1), 0.00001);
    }
}
