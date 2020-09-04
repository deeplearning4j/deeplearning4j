package org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic;

import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class NonRecurrentActorCriticHelperTest {

    private final NonRecurrentActorCriticHelper sut = new NonRecurrentActorCriticHelper(3);

    @Test
    public void when_callingCreateFeatures_expect_INDArrayWithCorrectShape() {
        // Arrange
        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 1.1, 1.2 }).reshape(1, 2)), 0, 1.0, false));
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 2.1, 2.2 }).reshape(1, 2)), 1, 2.0, false));
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 3.1, 3.2 }).reshape(1, 2)), 2, 3.0, false));
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 4.1, 4.2 }).reshape(1, 2)), 3, 4.0, false));
            }
        };

        // Act
        INDArray result = sut.createFeatures(experience);

        // Assert
        assertArrayEquals(new long[] { 4, 2 }, result.shape());
        assertEquals(1.1, result.getDouble(0, 0), 0.00001);
        assertEquals(1.2, result.getDouble(0, 1), 0.00001);
        assertEquals(2.1, result.getDouble(1, 0), 0.00001);
        assertEquals(2.2, result.getDouble(1, 1), 0.00001);
        assertEquals(3.1, result.getDouble(2, 0), 0.00001);
        assertEquals(3.2, result.getDouble(2, 1), 0.00001);
        assertEquals(4.1, result.getDouble(3, 0), 0.00001);
        assertEquals(4.2, result.getDouble(3, 1), 0.00001);
    }

    @Test
    public void when_callingCreateValueLabels_expect_INDArrayWithCorrectShape() {
        // Arrange

        // Act
        INDArray result = sut.createValueLabels(4);

        // Assert
        assertArrayEquals(new long[] { 4, 1 }, result.shape());
    }

    @Test
    public void when_callingCreatePolicyLabels_expect_ZeroINDArrayWithCorrectShape() {
        // Arrange

        // Act
        INDArray result = sut.createPolicyLabels(4);

        // Assert
        assertArrayEquals(new long[] { 4, 3 }, result.shape());
        for(int j = 0; j < 4; ++j) {
            for(int i = 0; i < 3; ++i) {
                assertEquals(0.0, result.getDouble(j, i), 0.00001);
            }
        }
    }

    @Test
    public void when_callingSetPolicy_expect_advantageSetAtCorrectLocation() {
        // Arrange
        INDArray policyArray = Nd4j.zeros(3, 3);

        // Act
        sut.setPolicy(policyArray, 1, 2, 123.0);

        // Assert
        for(int j = 0; j < 3; ++j) {
            for(int i = 0; i < 3; ++i) {
                if(j == 1 && i == 2) {
                    assertEquals(123.0, policyArray.getDouble(j, i), 0.00001);
                } else {
                    assertEquals(0.0, policyArray.getDouble(j, i), 0.00001);
                }
            }
        }
    }

}
