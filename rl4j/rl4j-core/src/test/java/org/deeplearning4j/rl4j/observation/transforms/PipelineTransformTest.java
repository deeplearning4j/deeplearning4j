package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.deeplearning4j.rl4j.observation.VoidObservation;
import org.deeplearning4j.rl4j.observation.support.TestPassthroughTransform;
import org.deeplearning4j.rl4j.observation.support.TestSourceTransform;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class PipelineTransformTest {

    @Test
    public void when_inputIsVoidObservation_expect_VoidObservationAsReturn() {
        // Arrange
        TestPassthroughTransform sut = new TestPassthroughTransform(false);

        // Act
        Observation result = sut.transform(VoidObservation.getInstance());

        // Assert
        assertTrue(result instanceof VoidObservation);

    }

    @Test
    public void when_pipelineIsEmpty_expect_ready() {
        // Assemble
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        PipelineTransform sut = PipelineTransform.builder().build();

        // Act
        sut.reset();
        boolean isReady = sut.isReady();

        // Assert
        assertTrue(isReady);
    }

    @Test
    public void when_firstElementNotReady_expect_notReady() {
        // Assemble
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        TestPassthroughTransform previous1 = new TestPassthroughTransform(false);
        TestPassthroughTransform previous2 = new TestPassthroughTransform(true);
        PipelineTransform sut = PipelineTransform.builder()
                .flowTo(previous1)
                .flowTo(previous2)
                .build();

        // Act
        sut.reset();
        boolean isReady = sut.isReady();
        Observation result = sut.transform(input);

        // Assert
        assertTrue(previous1.isResetCalled);
        assertTrue(previous2.isResetCalled);

        assertFalse(isReady);
        assertEquals(125.0, result.toNDArray().getDouble(0), 0.0);
    }

    @Test
    public void when_allElementsReady_expect_ready() {
        // Assemble
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        TestPassthroughTransform previous1 = new TestPassthroughTransform(true);
        TestPassthroughTransform previous2 = new TestPassthroughTransform(true);
        PipelineTransform sut = PipelineTransform.builder()
                .flowTo(previous1)
                .flowTo(previous2)
                .build();

        // Act
        sut.reset();
        boolean isReady = sut.isReady();
        Observation result = sut.transform(input);

        // Assert
        assertTrue(previous1.isResetCalled);
        assertTrue(previous2.isResetCalled);

        assertTrue(isReady);
        assertEquals(125.0, result.toNDArray().getDouble(0), 0.0);
    }

    @Test
    public void when_previousNotReady_expect_notReady() {
        // Assemble
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        TestPassthroughTransform previous1 = new TestPassthroughTransform(true);
        TestPassthroughTransform previous2 = new TestPassthroughTransform(true);
        TestSourceTransform source = new TestSourceTransform(false);
        PipelineTransform sut = PipelineTransform.builder(source)
                .flowTo(previous1)
                .flowTo(previous2)
                .build();

        // Act
        sut.reset();
        boolean isReady = sut.isReady();
        Observation result = sut.transform(input);

        // Assert
        assertTrue(source.isResetCalled);
        assertTrue(previous1.isResetCalled);
        assertTrue(previous2.isResetCalled);

        assertFalse(isReady);
        assertEquals(126.0, result.toNDArray().getDouble(0), 0.0);
    }

    @Test
    public void when_elementsAndPreviousReady_expect_ready() {
        // Assemble
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        TestPassthroughTransform previous1 = new TestPassthroughTransform(true);
        TestPassthroughTransform previous2 = new TestPassthroughTransform(true);
        TestSourceTransform source = new TestSourceTransform(true);
        PipelineTransform sut = PipelineTransform.builder(source)
                .flowTo(previous1)
                .flowTo(previous2)
                .build();

        // Act
        sut.reset();
        boolean isReady = sut.isReady();
        Observation result = sut.transform(input);

        // Assert
        assertTrue(source.isResetCalled);
        assertTrue(previous1.isResetCalled);
        assertTrue(previous2.isResetCalled);

        assertTrue(isReady);
        assertEquals(126.0, result.toNDArray().getDouble(0), 0.0);
    }

    @Test
    public void when_pipelineIsEmpty_expect_transformReturnsInput() {
        // Assemble
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        PipelineTransform sut = PipelineTransform.builder().build();

        // Act
        Observation result = sut.transform(input);

        // Assert
        assertEquals(123.0, result.toNDArray().getDouble(0), 0.0);
    }

    // Validates that all transforms in the pipeline are called and that they are called in order.
    @Test
    public void when_pipelineHasElements_expect_transformedInput() {
        // Assemble
        Observation input = new SimpleObservation(Nd4j.create(new double[] { 123.0 }));
        PipelineTransform sut = PipelineTransform.builder()
                .flowTo(new MulTransform(2.0))
                .flowTo(new AddTransform(1.0))
                .build();

        // Act
        Observation result = sut.transform(input);

        // Assert
        assertEquals((123.0 * 2.0) + 1.0, result.toNDArray().getDouble(0), 0.0);
    }

    private static class AddTransform extends PassthroughTransform {

        private final double value;

        public AddTransform(double value) {

            this.value = value;
        }

        @Override
        protected Observation handle(Observation input) {
            return new SimpleObservation(input.toNDArray().addi(value));
        }

        @Override
        protected boolean getIsReady() {
            return false;
        }
    }

    private static class MulTransform extends PassthroughTransform {

        private final double factor;

        public MulTransform(double factor){

            this.factor = factor;
        }

        @Override
        protected Observation handle(Observation input) {
            return new SimpleObservation(input.toNDArray().muli(factor));
        }

        @Override
        protected boolean getIsReady() {
            return false;
        }
    }

}
