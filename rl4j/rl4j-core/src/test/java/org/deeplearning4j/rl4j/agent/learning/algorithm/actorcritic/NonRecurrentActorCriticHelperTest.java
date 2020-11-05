package org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class NonRecurrentActorCriticHelperTest {

    private final NonRecurrentActorCriticHelper sut = new NonRecurrentActorCriticHelper(3);

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
