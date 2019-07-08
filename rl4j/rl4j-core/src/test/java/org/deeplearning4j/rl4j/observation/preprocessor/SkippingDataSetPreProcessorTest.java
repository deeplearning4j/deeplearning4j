package org.deeplearning4j.rl4j.observation.preprocessor;

import org.deeplearning4j.rl4j.observation.preprocessors.SkippingDataSetPreProcessor;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class SkippingDataSetPreProcessorTest {
    @Test(expected = IllegalArgumentException.class)
    public void when_ctorSkipFrameIsZero_expect_IllegalArgumentException() {
        SkippingDataSetPreProcessor sut = new SkippingDataSetPreProcessor(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void when_builderSkipFrameIsZero_expect_IllegalArgumentException() {
        SkippingDataSetPreProcessor sut = SkippingDataSetPreProcessor.builder()
                .skipFrame(0)
                .build();
    }

    @Test
    public void when_skipFrameIs3_expect_Skip2OutOf3() {
        // Arrange
        SkippingDataSetPreProcessor sut = SkippingDataSetPreProcessor.builder()
                .skipFrame(3)
                .build();
        DataSet[] results = new DataSet[4];

        // Act
        for(int i = 0; i < 4; ++i) {
            results[i] = new DataSet(Nd4j.create(new double[] { 123.0 }), null);
            sut.preProcess(results[i]);
        }

        // Assert
        assertFalse(results[0].isEmpty());
        assertTrue(results[1].isEmpty());
        assertTrue(results[2].isEmpty());
        assertFalse(results[3].isEmpty());
    }

    @Test
    public void when_resetIsCalled_expect_skippingIsReset() {
        // Arrange
        SkippingDataSetPreProcessor sut = SkippingDataSetPreProcessor.builder()
                .skipFrame(3)
                .build();
        DataSet[] results = new DataSet[4];

        // Act
        results[0] = new DataSet(Nd4j.create(new double[] { 123.0 }), null);
        results[1] = new DataSet(Nd4j.create(new double[] { 123.0 }), null);
        results[2] = new DataSet(Nd4j.create(new double[] { 123.0 }), null);
        results[3] = new DataSet(Nd4j.create(new double[] { 123.0 }), null);

        sut.preProcess(results[0]);
        sut.preProcess(results[1]);
        sut.reset();
        sut.preProcess(results[2]);
        sut.preProcess(results[3]);

        // Assert
        assertFalse(results[0].isEmpty());
        assertTrue(results[1].isEmpty());
        assertFalse(results[2].isEmpty());
        assertTrue(results[3].isEmpty());
    }
}
