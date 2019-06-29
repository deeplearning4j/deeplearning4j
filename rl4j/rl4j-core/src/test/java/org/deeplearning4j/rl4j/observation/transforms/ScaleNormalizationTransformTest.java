package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.deeplearning4j.rl4j.observation.VoidObservation;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class ScaleNormalizationTransformTest {

    @Test(expected = IllegalArgumentException.class)
    public void when_scaleIsZeroOrLess_expect_IllegalArgumentException() {
        ScaleNormalizationTransform sut = ScaleNormalizationTransform.builder().scale(0.0).build();
    }

    @Test
    public void when_scaleIsTwo_expect_InputHalved() {
        // Arrange
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 100.0 }));
        ScaleNormalizationTransform sut = ScaleNormalizationTransform.builder()
                .scale(2.0)
                .build();

        // Act
        Observation result = sut.transform(input);

        // Assert
        assertEquals(50.0, result.toNDArray().getDouble(0), 0.0);
    }

    @Test
    public void when_created_expect_isReady() {
        // Arrange
        ScaleNormalizationTransform sut = ScaleNormalizationTransform.builder()
                .scale(2.0)
                .build();

        // Act
        Boolean result = sut.isReady();

        // Assert
        assertTrue(result);
    }

}
