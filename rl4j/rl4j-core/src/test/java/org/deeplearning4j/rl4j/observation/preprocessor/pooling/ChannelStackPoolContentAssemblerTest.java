package org.deeplearning4j.rl4j.observation.preprocessor.pooling;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class ChannelStackPoolContentAssemblerTest {

    @Test
    public void when_assemble_expect_poolContentStackedOnChannel() {
        // Assemble
        ChannelStackPoolContentAssembler sut = new ChannelStackPoolContentAssembler();
        INDArray[] poolContent = new INDArray[] {
            Nd4j.rand(2, 2),
            Nd4j.rand(2, 2),
        };

        // Act
        INDArray result = sut.assemble(poolContent);

        // Assert
        assertEquals(3, result.shape().length);
        assertEquals(2, result.shape()[0]);
        assertEquals(2, result.shape()[1]);
        assertEquals(2, result.shape()[2]);

        assertEquals(poolContent[0].getDouble(new int[] { 0, 0 }), result.getDouble(new int[] { 0, 0, 0 }), 0.0001);
        assertEquals(poolContent[0].getDouble(new int[] { 0, 1 }), result.getDouble(new int[] { 0, 0, 1 }), 0.0001);
        assertEquals(poolContent[0].getDouble(new int[] { 1, 0 }), result.getDouble(new int[] { 0, 1, 0 }), 0.0001);
        assertEquals(poolContent[0].getDouble(new int[] { 1, 1 }), result.getDouble(new int[] { 0, 1, 1 }), 0.0001);

        assertEquals(poolContent[1].getDouble(new int[] { 0, 0 }), result.getDouble(new int[] { 1, 0, 0 }), 0.0001);
        assertEquals(poolContent[1].getDouble(new int[] { 0, 1 }), result.getDouble(new int[] { 1, 0, 1 }), 0.0001);
        assertEquals(poolContent[1].getDouble(new int[] { 1, 0 }), result.getDouble(new int[] { 1, 1, 0 }), 0.0001);
        assertEquals(poolContent[1].getDouble(new int[] { 1, 1 }), result.getDouble(new int[] { 1, 1, 1 }), 0.0001);

    }

}
