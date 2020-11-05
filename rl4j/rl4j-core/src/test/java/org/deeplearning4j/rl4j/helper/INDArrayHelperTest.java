package org.deeplearning4j.rl4j.helper;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class INDArrayHelperTest {
    @Test
    public void when_inputHasIncorrectShape_expect_outputWithCorrectShape() {
        // Arrange
        INDArray input = Nd4j.create(new double[] { 1.0, 2.0, 3.0});

        // Act
        INDArray output = INDArrayHelper.forceCorrectShape(input);

        // Assert
        assertEquals(2, output.shape().length);
        assertEquals(1, output.shape()[0]);
        assertEquals(3, output.shape()[1]);
    }

    @Test
    public void when_inputHasCorrectShape_expect_outputWithSameShape() {
        // Arrange
        INDArray input = Nd4j.create(new double[] { 1.0, 2.0, 3.0}).reshape(1, 3);

        // Act
        INDArray output = INDArrayHelper.forceCorrectShape(input);

        // Assert
        assertEquals(2, output.shape().length);
        assertEquals(1, output.shape()[0]);
        assertEquals(3, output.shape()[1]);
    }

    @Test
    public void when_inputHasOneDimension_expect_outputWithTwoDimensions() {
        // Arrange
        INDArray input = Nd4j.create(new double[] { 1.0 });

        // Act
        INDArray output = INDArrayHelper.forceCorrectShape(input);

        // Assert
        assertEquals(2, output.shape().length);
        assertEquals(1, output.shape()[0]);
        assertEquals(1, output.shape()[1]);
    }

    @Test
    public void when_callingCreateBatchForShape_expect_INDArrayWithCorrectShapeAndOriginalShapeUnchanged() {
        // Arrange
        long[] shape = new long[] { 1, 3, 4};

        // Act
        INDArray output = INDArrayHelper.createBatchForShape(2, shape);

        // Assert
        // Output shape
        assertArrayEquals(new long[] { 2, 3, 4 }, output.shape());

        // Input should remain unchanged
        assertArrayEquals(new long[] { 1, 3, 4 }, shape);

    }

    @Test
    public void when_callingCreateRnnBatchForShape_expect_INDArrayWithCorrectShapeAndOriginalShapeUnchanged() {
        // Arrange
        long[] shape = new long[] { 1, 3, 1 };

        // Act
        INDArray output = INDArrayHelper.createRnnBatchForShape(5, shape);

        // Assert
        // Output shape
        assertArrayEquals(new long[] { 1, 3, 5 }, output.shape());

        // Input should remain unchanged
        assertArrayEquals(new long[] { 1, 3, 1 }, shape);
    }

}
