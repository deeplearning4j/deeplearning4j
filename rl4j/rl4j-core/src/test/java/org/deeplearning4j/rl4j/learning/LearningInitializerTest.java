package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transforms.*;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.support.TestHistoryProcessor;
import org.deeplearning4j.rl4j.support.TestMDP;
import org.deeplearning4j.rl4j.support.TestObservationPool;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class LearningInitializerTest {
    private ObservationTransform getTransform(IHistoryProcessor hp, TestObservationPool observablePool, int skipFrame) {
        return PipelineTransform.builder()
                .flowTo(SignalingTransform.builder().listener(new RecorderTransformListener(hp)).build())
                .flowTo(SkippingTransform.builder().skipFrame(skipFrame).build())
                .flowTo(PoolingTransform.builder().observablePool(observablePool).build())
                .build();
    }

    @Test
    public void LearningInitializer_init_noTransform_ShouldBeAll0() {
        // Arrange
        ILearningInitializer<Observation, Integer, ActionSpace<Integer>> sut = new LearningInitializer<>();
        TestMDP mdp = new TestMDP(1);

        // Act
        Learning.InitMdp<Observation> result = sut.initMdp(mdp, null);

        // Assert
        assertEquals(0.0, result.getLastObs().toNDArray().getDouble(0), 0.0);
        assertEquals(0.0, result.getReward(), 0.0);
        assertEquals(0, result.getSteps());
    }

    @Test
    public void HistoryProcessorLearningInitializer_init_1HistoryLength() {
        // Arrange
        int historyLength = 1;
        int skipFrame = 1;
        TestObservationPool observablePool = new TestObservationPool(historyLength);
        TestHistoryProcessor hp = new TestHistoryProcessor(historyLength, skipFrame);
        ILearningInitializer<Observation, Integer, ActionSpace<Integer>> sut = new LearningInitializer<>();
        TestMDP mdp = new TestMDP(1000);

        // Act
        Learning.InitMdp<Observation> result = sut.initMdp(mdp, getTransform(hp, observablePool, skipFrame));

        // Assert
        assertEquals(0.0, result.getLastObs().toNDArray().getDouble(0), 0.0);
        assertEquals(0.0, result.getReward(), 0.0);
        assertEquals(0, result.getSteps());
    }

    @Test
    public void HistoryProcessorLearningInitializer_init_NoSkip5HistoryLen() {
        // Arrange
        int historyLength = 5;
        int skipFrame = 1;
        TestObservationPool observablePool = new TestObservationPool(historyLength);
        TestHistoryProcessor hp = new TestHistoryProcessor(historyLength, skipFrame);
        ILearningInitializer<Observation, Integer, ActionSpace<Integer>> sut = new LearningInitializer<>();
        TestMDP mdp = new TestMDP(1000);

        // Act
        Learning.InitMdp<Observation> result = sut.initMdp(mdp, getTransform(hp, observablePool, skipFrame));

        // Assert
        assertEquals(0.04, result.getLastObs().toNDArray().getDouble(new int[] { 9, 0 }), 0.01);
        assertEquals(10.0, result.getReward(), 0.0);
        assertEquals(4, result.getSteps());
    }

    @Test
    public void HistoryProcessorLearningInitializer_init_2Skip6HistoryLen() {
        // Arrange
        int historyLength = 6;
        int skipFrame = 2;
        TestObservationPool observablePool = new TestObservationPool(historyLength);
        TestHistoryProcessor hp = new TestHistoryProcessor(historyLength, skipFrame);
        ILearningInitializer<Observation, Integer, ActionSpace<Integer>> sut = new LearningInitializer<>();
        TestMDP mdp = new TestMDP(1000);

        // Act
        Learning.InitMdp<Observation> result = sut.initMdp(mdp, getTransform(hp, observablePool, skipFrame));

        // Assert
        assertEquals(0.1, result.getLastObs().toNDArray().getDouble(new int[] { 11, 0 }), 0.01);
        //assertEquals(55.0, result.getReward(), 0.0);
        //assertEquals(10, result.getSteps());
        assertEquals(11, hp.getRecordCount());
        assertEquals(6, observablePool.getAddCount());
    }


}
