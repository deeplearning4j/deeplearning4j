package org.deeplearning4j.rl4j.observation.channel;

import org.deeplearning4j.rl4j.observation.channel.legacy.LegacyChannelData;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.support.MockEncodable;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.junit.Assert.assertEquals;

public class LegacyChannelDataTest {

    @Test
    public void when_supplying1DData_expect_2DINDArray() {
        // Assemble
        Encodable data = new MockEncodable(new double[] { 1.0, 2.0, 3.0 });
        long[] shape = new long[] { 3 };
        LegacyChannelData sut = new LegacyChannelData(data, shape);

        // Act
        INDArray result = sut.toINDArray();

        // Assert
        long[] resultShape = result.shape();
        assertEquals(2, resultShape.length);
        assertEquals(1, resultShape[0]);
        assertEquals(3, resultShape[1]);

        assertEquals(1.0, result.getDouble(0, 0), 0.00001);
        assertEquals(2.0, result.getDouble(0, 1), 0.00001);
        assertEquals(3.0, result.getDouble(0, 2), 0.00001);
    }

    @Test
    public void when_supplying2DData_expect_2DINDArray() {
        // Assemble
        Encodable data = new MockEncodable(new double[] { 1.0, 2.0, 3.0, 4.0 });
        long[] shape = new long[] { 2, 2 };
        LegacyChannelData sut = new LegacyChannelData(data, shape);

        // Act
        INDArray result = sut.toINDArray();

        // Assert
        long[] resultShape = result.shape();
        assertEquals(2, resultShape.length);
        assertEquals(2, resultShape[0]);
        assertEquals(2, resultShape[1]);

        assertEquals(1.0, result.getDouble(0, 0), 0.00001);
        assertEquals(2.0, result.getDouble(0, 1), 0.00001);
        assertEquals(3.0, result.getDouble(1, 0), 0.00001);
        assertEquals(4.0, result.getDouble(1, 1), 0.00001);
    }

    private static class MockEncodable implements Encodable {

        private final double[] data;

        public MockEncodable(double[] data) {
            this.data = data;
        }

        @Override
        public double[] toArray() {
            return data;
        }
    }

}
