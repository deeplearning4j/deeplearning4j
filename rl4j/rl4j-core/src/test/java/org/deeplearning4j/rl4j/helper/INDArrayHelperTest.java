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
        assertEquals(3, output.shape().length);
        assertEquals(2, output.shape()[0]);
        assertEquals(3, output.shape()[1]);
        assertEquals(4, output.shape()[2]);

        // Input should remain unchanged
        assertEquals(1, shape[0]);
        assertEquals(3, shape[1]);
        assertEquals(4, shape[2]);

    }
}
