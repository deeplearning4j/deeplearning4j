package org.deeplearning4j.rl4j.observation.prefiltering;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class UniformSkippingPreFilterTest {
    @Test(expected = IllegalArgumentException.class)
    public void when_skipFrameIs0OrLess_expect_IllegalArgumentException() {
        // Assemble
        UniformSkippingPreFilter sut = new UniformSkippingPreFilter(0);
    }

    @Test
    public void when_frameIsMultipleOfSkippedFrame_expect_passingFilter() {
        // Assemble
        UniformSkippingPreFilter sut = new UniformSkippingPreFilter(2);

        // Act
        boolean result = sut.isPassing(null, 0, false);

        // Assert
        assertTrue(result);
    }

    @Test
    public void when_frameShouldBeSkipped_expect_notPassingFilter() {
        // Assemble
        UniformSkippingPreFilter sut = new UniformSkippingPreFilter(2);

        // Act
        boolean result = sut.isPassing(null, 1, false);

        // Assert
        assertFalse(result);
    }

    @Test
    public void when_frameIsLast_expect_passingFilter() {
        // Assemble
        UniformSkippingPreFilter sut = new UniformSkippingPreFilter(2);

        // Act
        boolean result = sut.isPassing(null, 1, true);

        // Assert
        assertTrue(result);
    }

}
