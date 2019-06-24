package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.rl4j.learning.HistoryTransformListener;
import org.deeplearning4j.rl4j.learning.RecorderTransformListener;
import org.deeplearning4j.rl4j.observation.transforms.*;
import org.deeplearning4j.rl4j.support.*;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class PolicyRefacTest {

    TestObservationPool observablePool;
    PipelineTransform pipelineObservable;
    TestHistoryProcessor hp;

    @Before
    public void setUp() {
        observablePool = new TestObservationPool(6);

        hp = new TestHistoryProcessor(6, 2);

        pipelineObservable = PipelineTransform.builder()
            .flowTo(SignalingTransform.builder().listener(new RecorderTransformListener(hp)).build())
            .flowTo(SkippingTransform.builder().skipFrame(2).build())
            .flowTo(SignalingTransform.builder().listener(new HistoryTransformListener(hp)).build())

            .flowTo(PoolingTransform.builder().observablePool(observablePool).build())
            //.flowTo(new TestObservationTransform(hp, 6))
            .build();

        /*
        ObservationTransform previous = new TestObservable();
        TransformListener listener = new HistoryProcessorObservableListener(hp);
        SignalingTransform result = new SignalingTransform(previous);
        result.addListener(listener);
        return result;
        */

    }


    @Test
    public void Policy_play_WithHistoryProcessor() {
        // Arrange
        TestMDP mdp = new TestMDP(20);
        TestPolicy sut = new TestPolicy();

        // Act
        double reward = sut.play(mdp, hp, pipelineObservable);

        // Assert
        assertEquals(190, reward, 0.0);
        //assertEquals(21, hp.getRecordCount());
        //assertEquals(15, hp.getAddCount());
        assertEquals(15, observablePool.getAddCount());
    }
}
