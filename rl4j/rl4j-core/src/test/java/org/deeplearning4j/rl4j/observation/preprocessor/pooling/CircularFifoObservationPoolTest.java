package org.deeplearning4j.rl4j.observation.preprocessor.pooling;

import org.deeplearning4j.rl4j.observation.preprocessors.pooling.CircularFifoObservationPool;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class CircularFifoObservationPoolTest {

    @Test(expected = IllegalArgumentException.class)
    public void when_poolSizeZeroOrLess_expect_IllegalArgumentException() {
        CircularFifoObservationPool sut = new CircularFifoObservationPool(0);
    }

    @Test
    public void when_poolIsEmpty_expect_NotReady() {
        // Assemble
        CircularFifoObservationPool sut = new CircularFifoObservationPool();

        // Act
        boolean isReady = sut.isAtFullCapacity();

        // Assert
        assertFalse(isReady);
    }

    @Test
    public void when_notEnoughElementsInPool_expect_notReady() {
        // Assemble
        CircularFifoObservationPool sut = new CircularFifoObservationPool();
        sut.add(Nd4j.create(new double[] { 123.0 }));

        // Act
        boolean isReady = sut.isAtFullCapacity();

        // Assert
        assertFalse(isReady);
    }

    @Test
    public void when_enoughElementsInPool_expect_ready() {
        // Assemble
        CircularFifoObservationPool sut = CircularFifoObservationPool.builder()
                .poolSize(2)
                .build();
        sut.add(Nd4j.create(new double[] { 123.0 }));
        sut.add(Nd4j.create(new double[] { 123.0 }));

        // Act
        boolean isReady = sut.isAtFullCapacity();

        // Assert
        assertTrue(isReady);
    }

    @Test
    public void when_addMoreThanSize_expect_getReturnOnlyLastElements() {
        // Assemble
        CircularFifoObservationPool sut = CircularFifoObservationPool.builder().build();
        sut.add(Nd4j.create(new double[] { 0.0 }));
        sut.add(Nd4j.create(new double[] { 1.0 }));
        sut.add(Nd4j.create(new double[] { 2.0 }));
        sut.add(Nd4j.create(new double[] { 3.0 }));
        sut.add(Nd4j.create(new double[] { 4.0 }));
        sut.add(Nd4j.create(new double[] { 5.0 }));
        sut.add(Nd4j.create(new double[] { 6.0 }));

        // Act
        INDArray[] result = sut.get();

        // Assert
        assertEquals(3.0, result[0].getDouble(0), 0.0);
        assertEquals(4.0, result[1].getDouble(0), 0.0);
        assertEquals(5.0, result[2].getDouble(0), 0.0);
        assertEquals(6.0, result[3].getDouble(0), 0.0);
    }

    @Test
    public void when_resetIsCalled_expect_poolContentFlushed() {
        // Assemble
        CircularFifoObservationPool sut = CircularFifoObservationPool.builder().build();
        sut.add(Nd4j.create(new double[] { 0.0 }));
        sut.add(Nd4j.create(new double[] { 1.0 }));
        sut.add(Nd4j.create(new double[] { 2.0 }));
        sut.add(Nd4j.create(new double[] { 3.0 }));
        sut.add(Nd4j.create(new double[] { 4.0 }));
        sut.add(Nd4j.create(new double[] { 5.0 }));
        sut.add(Nd4j.create(new double[] { 6.0 }));
        sut.reset();

        // Act
        INDArray[] result = sut.get();

        // Assert
        assertEquals(0, result.length);
    }
}
