package org.deeplearning4j.rl4j.agent.learning.update;

import org.deeplearning4j.nn.gradient.Gradient;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.mockito.Mockito.mock;

@RunWith(MockitoJUnitRunner.class)
public class GradientsTest {

    @Test
    public void when_getBatchSizeIsCalled_expect_batchSizeIsReturned() {
        // Arrange
        Gradients sut = new Gradients(5);

        // Act
        long batchSize = sut.getBatchSize();

        // Assert
        assertEquals(5, batchSize);
    }

    @Test
    public void when_puttingLabels_expect_getLabelReturnsLabels() {
        // Arrange
        Gradient gradient = mock(Gradient.class);
        Gradients sut = new Gradients(5);
        sut.putGradient("test", gradient);

        // Act
        Gradient result = sut.getGradient("test");

        // Assert
        assertSame(gradient, result);
    }
}
