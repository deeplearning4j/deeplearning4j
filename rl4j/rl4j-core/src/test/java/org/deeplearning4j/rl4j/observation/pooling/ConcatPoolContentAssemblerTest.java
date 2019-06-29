package org.deeplearning4j.rl4j.observation.pooling;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class ConcatPoolContentAssemblerTest {

    @Test(expected = IllegalArgumentException.class)
    public void when_dimensionNegative_expect_IllegalArgumentException() {
        ConcatPoolContentAssembler sut = new ConcatPoolContentAssembler(-1);
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void when_poolContentIsNull_expect_ND4JIllegalStateException() {
        ConcatPoolContentAssembler sut = new ConcatPoolContentAssembler();
        INDArray result = sut.assemble(null);
    }

    @Test
    public void when_concatOnDimension0_expect_NDArraysConcatenatedOnDimension0() {
        // Assemble
        ConcatPoolContentAssembler sut = new ConcatPoolContentAssembler(0);
        INDArray[] poolContent = new INDArray[] {
            Nd4j.create(new double[][] {
                    new double[] { 1.0, 1.1 },
                    new double[] { 1.11, 1.111 }
            }),
            Nd4j.create(new double[][] {
                    new double[] { 2.0, 2.2 },
                    new double[] { 2.22, 2.222 }
            }),
        };

        // Act
        INDArray result = sut.assemble(poolContent);

        // Assert
        assertEquals(2, result.shape().length);
        assertEquals(4, result.shape()[0]);
        assertEquals(2, result.shape()[1]);

        assertEquals(1.0, result.getDouble(new int[] { 0, 0 }), 0.0);
        assertEquals(1.1, result.getDouble(new int[] { 0, 1 }), 0.0);

        assertEquals(1.11, result.getDouble(new int[] { 1, 0 }), 0.0);
        assertEquals(1.111,  result.getDouble(new int[] { 1, 1 }), 0.0);

        assertEquals(2.0, result.getDouble(new int[] { 2, 0}), 0.0);
        assertEquals(2.2, result.getDouble(new int[] { 2, 1}), 0.0);

        assertEquals(2.22, result.getDouble(new int[] { 3, 0}), 0.0);
        assertEquals(2.222, result.getDouble(new int[] { 3, 1}), 0.0);

    }

}
