package org.deeplearning4j.rl4j.observation.transform.operation;

import org.deeplearning4j.rl4j.helper.INDArrayHelper;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class SimpleNormalizationTransformTest {
    @Test(expected = IllegalArgumentException.class)
    public void when_maxIsLessThanMin_expect_exception() {
        // Arrange
        SimpleNormalizationTransform sut = new SimpleNormalizationTransform(10.0, 1.0);
    }

    @Test
    public void when_transformIsCalled_expect_inputNormalized() {
        // Arrange
        SimpleNormalizationTransform sut = new SimpleNormalizationTransform(1.0, 11.0);
        INDArray input = Nd4j.create(new double[] { 1.0, 11.0 });

        // Act
        INDArray result = sut.transform(input);

        // Assert
        assertEquals(0.0, result.getDouble(0), 0.00001);
        assertEquals(1.0, result.getDouble(1), 0.00001);
    }

}
