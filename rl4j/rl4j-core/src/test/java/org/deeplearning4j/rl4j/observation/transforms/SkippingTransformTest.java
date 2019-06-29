package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.deeplearning4j.rl4j.observation.VoidObservation;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class SkippingTransformTest {

    @Test(expected = IllegalArgumentException.class)
    public void when_ctorSkipFrameIsZero_expect_IllegalArgumentException() {
        SkippingTransform sut = new SkippingTransform(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void when_builderSkipFrameIsZero_expect_IllegalArgumentException() {
        SkippingTransform sut = SkippingTransform.builder()
                .skipFrame(0)
                .build();
    }

    @Test
    public void when_skipFrameIs3_expect_Skip2OutOf3() {
        // Arrange
        SkippingTransform sut = SkippingTransform.builder()
                .skipFrame(3)
                .build();
        Observation[] observations = new Observation[4];
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));

        // Act
        for(int i = 0; i < 4; ++i) {
            observations[i] = sut.transform(input);
        }

        // Assert
        assertFalse(observations[0] instanceof VoidObservation);
        assertTrue(observations[1] instanceof VoidObservation);
        assertTrue(observations[2] instanceof VoidObservation);
        assertFalse(observations[3] instanceof VoidObservation);
    }

    @Test
    public void when_resetIsCalled_expect_skippingIsReset() {
        // Arrange
        SkippingTransform sut = SkippingTransform.builder()
                .skipFrame(3)
                .build();
        Observation[] observations = new Observation[4];
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));

        // Act
        observations[0] = sut.transform(input);
        observations[1] = sut.transform(input);
        sut.reset();
        observations[2] = sut.transform(input);
        observations[3] = sut.transform(input);

        // Assert
        assertFalse(observations[0] instanceof VoidObservation);
        assertTrue(observations[1] instanceof VoidObservation);
        assertFalse(observations[2] instanceof VoidObservation);
        assertTrue(observations[3] instanceof VoidObservation);
    }

}
