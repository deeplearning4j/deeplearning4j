package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.deeplearning4j.rl4j.observation.VoidObservation;
import org.deeplearning4j.rl4j.observation.pooling.ObservationPool;
import org.deeplearning4j.rl4j.observation.pooling.PoolContentAssembler;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class PoolingTransformTest {

    @Test
    public void when_builderHasNoPoolOrAssembler_expect_defaultPoolBehavior() {
        // Arrange
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        PoolingTransform sut = PoolingTransform.builder().build();
        boolean isReady[] = new boolean[5];
        Observation[] observations = new Observation[5];

        // Act
        for(int i = 0; i < 5; ++i) {
            isReady[i] = sut.isReady();
            observations[i] = sut.transform(input);
        }

        // Assert
        assertFalse(isReady[0]);
        assertFalse(isReady[1]);
        assertFalse(isReady[2]);
        assertFalse(isReady[3]);
        assertTrue(isReady[4]);

        assertTrue(observations[0] instanceof VoidObservation);
        assertTrue(observations[1] instanceof VoidObservation);
        assertTrue(observations[2] instanceof VoidObservation);

        assertEquals(123.0, observations[3].toNDArray().getDouble(0), 0.0);
        assertEquals(123.0, observations[3].toNDArray().getDouble(1), 0.0);
        assertEquals(123.0, observations[3].toNDArray().getDouble(2), 0.0);
        assertEquals(123.0, observations[3].toNDArray().getDouble(3), 0.0);
    }

    @Test
    public void when_builderHasPoolAndAssembler_expect_paramPoolAndAssemblerAreUsed() {
        // Arrange
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        TestObservationPool pool = new TestObservationPool();
        TestPoolContentAssembler assembler = new TestPoolContentAssembler();
        PoolingTransform sut = PoolingTransform.builder()
                .observablePool(pool)
                .poolContentAssembler(assembler)
                .build();

        // Act
        sut.isReady();
        sut.transform(input);

        // Assert
        assertTrue(pool.isReadyCalled);
        assertTrue(pool.isGetCalled);
        assertEquals(123.0, pool.observation.getDouble(0), 0.0);
        assertTrue(assembler.assembleIsCalled);
    }

    private static class TestObservationPool implements ObservationPool {

        public INDArray observation;
        public boolean isGetCalled;
        public boolean isReadyCalled;

        @Override
        public void add(INDArray observation) {
            this.observation = observation;
        }

        @Override
        public INDArray[] get() {
            isGetCalled = true;
            return new INDArray[0];
        }

        @Override
        public boolean isReady() {
            isReadyCalled = true;
            return true;
        }
    }

    private static class TestPoolContentAssembler implements PoolContentAssembler {

        public boolean assembleIsCalled;

        @Override
        public INDArray assemble(INDArray[] poolContent) {
            assembleIsCalled = true;
            return null;
        }
    }
}
