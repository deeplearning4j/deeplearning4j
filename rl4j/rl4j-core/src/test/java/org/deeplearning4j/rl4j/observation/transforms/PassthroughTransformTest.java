package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.deeplearning4j.rl4j.observation.VoidObservation;
import org.deeplearning4j.rl4j.observation.support.TestPassthroughTransform;
import org.deeplearning4j.rl4j.observation.support.TestPreviousTransform;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class PassthroughTransformTest {

    @Test
    public void when_inputIsVoidObservation_expect_VoidObservationAsReturn() {
        // Arrange
        TestPassthroughTransform sut = new TestPassthroughTransform(false);

        // Act
        Observation result = sut.transform(VoidObservation.getInstance());

        // Assert
        assertTrue(result instanceof VoidObservation);

    }

    @Test
    public void when_isReadyCalled_expect_callToGetIsREady() {
        // Arrange
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        TestPassthroughTransform sut = new TestPassthroughTransform(false);

        // Act
        sut.reset();
        sut.isReady();

        // Assert
        assertTrue(sut.getIsReadyCalled);
    }

    @Test
    public void when_transformCalled_expect_callToHandle() {
        // Arrange
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        TestPassthroughTransform sut = new TestPassthroughTransform(false);

        // Act
        sut.reset();
        sut.transform(input);

        // Assert
        assertTrue(sut.isHandledCalled);
        assertEquals(123.0, sut.handleInput, 0.0);
    }

    @Test
    public void PassthroughTransform_WithPrevious() {
        // Arrange
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        TestPreviousTransform testPrevious = new TestPreviousTransform(true);
        TestPassthroughTransform sut = new TestPassthroughTransform(false);
        sut.setPrevious(testPrevious);

        // Act
        sut.reset();
        sut.isReady();
        Observation result = sut.transform(input);

        // Assert
        assertTrue(sut.getIsReadyCalled);
        assertTrue(sut.isHandledCalled);
        assertTrue(testPrevious.isResetCalled);
        assertTrue(testPrevious.isReadyCalled);
        assertTrue(testPrevious.getObservationCalled);
        assertEquals(123.0, testPrevious.getObservationInput, 0.0);
        assertEquals(123.0, sut.handleInput, 0.0);
        assertEquals(124, result.toNDArray().getDouble(0), 0.0);

    }

    @Test
    public void when_previousNotReady_expect_notBeReady() {
        // Arrange
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        TestPreviousTransform testPrevious = new TestPreviousTransform(false);
        TestPassthroughTransform sut = new TestPassthroughTransform(false);
        sut.setPrevious(testPrevious);


        // Act
        boolean isReady = sut.isReady();

        // Assert
        assertFalse(isReady);
        assertTrue(sut.getIsReadyCalled);
        assertFalse(sut.isHandledCalled);

        assertTrue(testPrevious.isReadyCalled);
        assertFalse(testPrevious.getObservationCalled);
    }

    @Test
    public void when_previousReadyCurrentIsNot_expect_notBeReady() {
        // Arrange
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        TestPreviousTransform testPrevious = new TestPreviousTransform(true);
        TestPassthroughTransform sut = new TestPassthroughTransform(false);
        sut.setPrevious(testPrevious);


        // Act
        boolean isReady = sut.isReady();

        // Assert
        assertFalse(isReady);
        assertTrue(sut.getIsReadyCalled);
        assertTrue(testPrevious.isReadyCalled);
    }

    @Test
    public void when_allReady_expect_ready() {
        // Arrange
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        TestPreviousTransform testPrevious = new TestPreviousTransform(true);
        TestPassthroughTransform sut = new TestPassthroughTransform(true);
        sut.setPrevious(testPrevious);


        // Act
        boolean isReady = sut.isReady();

        // Assert
        assertTrue(isReady);
        assertTrue(sut.getIsReadyCalled);
        assertTrue(testPrevious.isReadyCalled);
    }
}
