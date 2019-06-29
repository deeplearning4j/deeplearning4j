package org.deeplearning4j.rl4j.observation;

import org.deeplearning4j.rl4j.observation.pooling.CircularFifoObservationPool;
import org.deeplearning4j.rl4j.observation.pooling.ObservationPool;
import org.deeplearning4j.rl4j.observation.transforms.*;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class ObservationIntegrationTest {

    private Observation createObservation(double value) {
        return new SimpleObservation(Nd4j.create(new double[] { value }));
    }

    @Test
    public void PipelineWithAllTransforms() {
        // Arrange
        TestListener listener = new TestListener();
        SignalingTransform signal = SignalingTransform.builder()
                .listener(listener)
                .build();

        SkippingTransform skip = SkippingTransform.builder()
                .skipFrame(3)
                .build();

        ScaleNormalizationTransform scale = ScaleNormalizationTransform.builder()
                .scale(10.0)
                .build();

        PoolingTransform pool = PoolingTransform.builder()
                .observablePool(CircularFifoObservationPool.builder().poolSize(2).build())
                .build();

        PipelineTransform pipeline = PipelineTransform.builder()
                .flowTo(signal)
                .flowTo(skip)
                .flowTo(scale)
                .flowTo(pool)
                .build();
        List<Observation> results = new ArrayList<Observation>();
        double value = 0.0;

        // Act
        pipeline.reset();
        int readyAt = -1;
        for(int i = 0; i < 100; ++i) {
            if(pipeline.isReady()) {
                readyAt = i;
                break;
            }
            pipeline.transform(createObservation(value++));
        }

        for (int i = 0; i < 6; ++i) {
            results.add(pipeline.transform(createObservation(value++)));
        }

        // Assert
        assertEquals(4, readyAt);

        assertTrue(results.get(0) instanceof VoidObservation);
        assertTrue(results.get(1) instanceof VoidObservation);
        assertEquals(0.3, results.get(2).toNDArray().getDouble(0), 0.001);
        assertEquals(0.6, results.get(2).toNDArray().getDouble(1), 0.001);
        assertTrue(results.get(3) instanceof VoidObservation);
        assertTrue(results.get(4) instanceof VoidObservation);
        assertEquals(0.6, results.get(5).toNDArray().getDouble(0), 0.001);
        assertEquals(0.9, results.get(5).toNDArray().getDouble(1), 0.001);

    }

    private static class TestListener implements TransformListener {

        public boolean onResetCalled;
        public List<INDArray> observations = new ArrayList<INDArray>();

        @Override
        public void onReset() {
            onResetCalled = true;
        }

        @Override
        public void onTransform(Observation observation) {
            this.observations.add(observation.toNDArray().add(0.0));
        }
    }
}
