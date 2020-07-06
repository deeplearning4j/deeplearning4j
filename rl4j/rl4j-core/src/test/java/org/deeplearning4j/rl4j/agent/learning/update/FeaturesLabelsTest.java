package org.deeplearning4j.rl4j.agent.learning.update;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class FeaturesLabelsTest {

    @Test
    public void when_getBatchSizeIsCalled_expect_batchSizeIsReturned() {
        // Arrange
        INDArray features = Nd4j.create(5, 10);
        FeaturesLabels sut = new FeaturesLabels(features);

        // Act
        long batchSize = sut.getBatchSize();

        // Assert
        assertEquals(5, batchSize);
    }

    @Test
    public void when_puttingLabels_expect_getLabelReturnsLabels() {
        // Arrange
        INDArray features = Nd4j.create(5, 10);
        INDArray labels = Nd4j.rand(2, 3);
        FeaturesLabels sut = new FeaturesLabels(features);
        sut.putLabels("test", labels);

        // Act
        INDArray result = sut.getLabels("test");

        // Assert
        assertEquals(result, labels);
    }
}
