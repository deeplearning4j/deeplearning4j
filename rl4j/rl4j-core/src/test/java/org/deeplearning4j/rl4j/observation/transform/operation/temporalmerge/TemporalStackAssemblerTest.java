package org.deeplearning4j.rl4j.observation.transform.operation.temporalmerge;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class TemporalStackAssemblerTest {

    @Test
    public void when_assembling2INDArrays_expect_stackedAsResult() {
        // Arrange
        INDArray[] input = new INDArray[] {
                Nd4j.create(new double[] { 1.0, 2.0, 3.0 }),
                Nd4j.create(new double[] { 10.0, 20.0, 30.0 }),
        };
        TemporalStackAssembler sut = new TemporalStackAssembler();

        // Act
        INDArray result = sut.assemble(input);

        // Assert
        assertEquals(2, result.shape().length);
        assertEquals(2, result.shape()[0]);
        assertEquals(3, result.shape()[1]);

        assertEquals(1.0, result.getDouble(0, 0), 0.00001);
        assertEquals(2.0, result.getDouble(0, 1), 0.00001);
        assertEquals(3.0, result.getDouble(0, 2), 0.00001);

        assertEquals(10.0, result.getDouble(1, 0), 0.00001);
        assertEquals(20.0, result.getDouble(1, 1), 0.00001);
        assertEquals(30.0, result.getDouble(1, 2), 0.00001);

    }
}
