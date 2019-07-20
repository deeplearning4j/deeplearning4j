package org.deeplearning4j.rl4j.observation.preprocessor;

import org.deeplearning4j.rl4j.observation.preprocessor.pooling.ObservationPool;
import org.deeplearning4j.rl4j.observation.preprocessor.pooling.PoolContentAssembler;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

public class PoolingDataSetPreProcessorTest {

    @Test(expected = NullPointerException.class)
    public void when_dataSetIsNull_expect_NullPointerException() {
        // Assemble
        PoolingDataSetPreProcessor sut = PoolingDataSetPreProcessor.builder().build();

        // Act
        sut.preProcess(null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void when_dataSetHasMoreThanOneExample_expect_IllegalArgumentException() {
        // Assemble
        PoolingDataSetPreProcessor sut = PoolingDataSetPreProcessor.builder().build();

        // Act
        sut.preProcess(new DataSet(Nd4j.rand(new long[] { 2, 2, 2 }), null));
    }

    @Test
    public void when_dataSetIsEmpty_expect_EmptyDataSet() {
        // Assemble
        PoolingDataSetPreProcessor sut = PoolingDataSetPreProcessor.builder().build();
        DataSet ds = new DataSet(null, null);

        // Act
        sut.preProcess(ds);

        // Assert
        Assert.assertTrue(ds.isEmpty());
    }

    @Test
    public void when_builderHasNoPoolOrAssembler_expect_defaultPoolBehavior() {
        // Arrange
        PoolingDataSetPreProcessor sut = PoolingDataSetPreProcessor.builder().build();
        DataSet[] observations = new DataSet[5];
        INDArray[] inputs = new INDArray[5];


        // Act
        for(int i = 0; i < 5; ++i) {
            inputs[i] = Nd4j.rand(new long[] { 1, 2, 2 });
            DataSet input = new DataSet(inputs[i], null);
            sut.preProcess(input);
            observations[i] = input;
        }

        // Assert
        assertTrue(observations[0].isEmpty());
        assertTrue(observations[1].isEmpty());
        assertTrue(observations[2].isEmpty());

        for(int i = 0; i < 4; ++i) {
            assertEquals(inputs[i].getDouble(new int[] { 0, 0, 0 }), observations[3].getFeatures().getDouble(new int[] { 0, i, 0, 0 }), 0.0001);
            assertEquals(inputs[i].getDouble(new int[] { 0, 0, 1 }), observations[3].getFeatures().getDouble(new int[] { 0, i, 0, 1 }), 0.0001);
            assertEquals(inputs[i].getDouble(new int[] { 0, 1, 0 }), observations[3].getFeatures().getDouble(new int[] { 0, i, 1, 0 }), 0.0001);
            assertEquals(inputs[i].getDouble(new int[] { 0, 1, 1 }), observations[3].getFeatures().getDouble(new int[] { 0, i, 1, 1 }), 0.0001);
        }

        for(int i = 0; i < 4; ++i) {
            assertEquals(inputs[i+1].getDouble(new int[] { 0, 0, 0 }), observations[4].getFeatures().getDouble(new int[] { 0, i, 0, 0 }), 0.0001);
            assertEquals(inputs[i+1].getDouble(new int[] { 0, 0, 1 }), observations[4].getFeatures().getDouble(new int[] { 0, i, 0, 1 }), 0.0001);
            assertEquals(inputs[i+1].getDouble(new int[] { 0, 1, 0 }), observations[4].getFeatures().getDouble(new int[] { 0, i, 1, 0 }), 0.0001);
            assertEquals(inputs[i+1].getDouble(new int[] { 0, 1, 1 }), observations[4].getFeatures().getDouble(new int[] { 0, i, 1, 1 }), 0.0001);
        }

    }

    @Test
    public void when_builderHasPoolAndAssembler_expect_paramPoolAndAssemblerAreUsed() {
        // Arrange
        INDArray input = Nd4j.rand(1, 1);
        TestObservationPool pool = new TestObservationPool();
        TestPoolContentAssembler assembler = new TestPoolContentAssembler();
        PoolingDataSetPreProcessor sut = PoolingDataSetPreProcessor.builder()
                .observablePool(pool)
                .poolContentAssembler(assembler)
                .build();

        // Act
        sut.preProcess(new DataSet(input, null));

        // Assert
        assertTrue(pool.isAtFullCapacityCalled);
        assertTrue(pool.isGetCalled);
        assertEquals(input.getDouble(0), pool.observation.getDouble(0), 0.0);
        assertTrue(assembler.assembleIsCalled);
    }

    @Test
    public void when_pastInputChanges_expect_outputNotChanged() {
        // Arrange
        INDArray input = Nd4j.zeros(1, 1);
        TestObservationPool pool = new TestObservationPool();
        TestPoolContentAssembler assembler = new TestPoolContentAssembler();
        PoolingDataSetPreProcessor sut = PoolingDataSetPreProcessor.builder()
                .observablePool(pool)
                .poolContentAssembler(assembler)
                .build();

        // Act
        sut.preProcess(new DataSet(input, null));
        input.putScalar(0, 0, 1.0);

        // Assert
        assertEquals(0.0, pool.observation.getDouble(0), 0.0);
    }

    private static class TestObservationPool implements ObservationPool {

        public INDArray observation;
        public boolean isGetCalled;
        public boolean isAtFullCapacityCalled;
        private boolean isResetCalled;

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
        public boolean isAtFullCapacity() {
            isAtFullCapacityCalled = true;
            return true;
        }

        @Override
        public void reset() {
            isResetCalled = true;
        }
    }

    private static class TestPoolContentAssembler implements PoolContentAssembler {

        public boolean assembleIsCalled;

        @Override
        public INDArray assemble(INDArray[] poolContent) {
            assembleIsCalled = true;
            return Nd4j.create(1, 1);
        }
    }
}
