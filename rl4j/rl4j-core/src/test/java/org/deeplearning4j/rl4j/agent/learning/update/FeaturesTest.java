package org.deeplearning4j.rl4j.agent.learning.update;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

public class FeaturesTest {

    @Test
    public void when_creatingFeatureWithBatchSize10_expectGetBatchSizeReturn10() {
        // Arrange
        INDArray[] featuresData = new INDArray[] {Nd4j.rand(10, 1)};

        // Act
        Features sut = new Features(featuresData);

        // Assert
        assertEquals(10, sut.getBatchSize());
    }

    @Test
    public void when_callingGetWithAChannelIndex_expectGetReturnsThatChannelData() {
        // Arrange
        INDArray channel0Data = Nd4j.rand(10, 1);
        INDArray channel1Data = Nd4j.rand(10, 1);
        INDArray[] featuresData = new INDArray[] { channel0Data, channel1Data };

        // Act
        Features sut = new Features(featuresData);

        // Assert
        assertSame(channel1Data, sut.get(1));
    }

}
