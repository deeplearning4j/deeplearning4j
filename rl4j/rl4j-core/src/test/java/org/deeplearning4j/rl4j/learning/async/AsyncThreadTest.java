package org.deeplearning4j.rl4j.learning.async;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.support.*;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public class AsyncThreadTest {

    @Test
    public void when_newEpochStarted_expect_neuralNetworkReset() {
        // Arrange
        int numberOfEpochs = 5;
        TestContext context = new TestContext(numberOfEpochs);

        // Act
        context.sut.run();

        // Assert
        assertEquals(numberOfEpochs, context.neuralNet.resetCallCount);
    }

    @Test
    public void when_onNewEpochReturnsStop_expect_threadStopped() {
        // Arrange
        int stopAfterNumCalls = 1;
        TestContext context = new TestContext(100000);
        context.listener.setRemainingOnNewEpochCallCount(stopAfterNumCalls);

        // Act
        context.sut.run();

        // Assert
        assertEquals(stopAfterNumCalls + 1, context.listener.onNewEpochCallCount); // +1: The call that returns stop is counted
        assertEquals(stopAfterNumCalls, context.listener.onEpochTrainingResultCallCount);
    }

    @Test
    public void when_epochTrainingResultReturnsStop_expect_threadStopped() {
        // Arrange
        int stopAfterNumCalls = 1;
        TestContext context = new TestContext(100000);
        context.listener.setRemainingOnEpochTrainingResult(stopAfterNumCalls);

        // Act
        context.sut.run();

        // Assert
        assertEquals(stopAfterNumCalls + 1, context.listener.onEpochTrainingResultCallCount); // +1: The call that returns stop is counted
        assertEquals(stopAfterNumCalls + 1, context.listener.onNewEpochCallCount); // +1: onNewEpoch is called on the epoch that onEpochTrainingResult() will stop
    }

    @Test
    public void when_run_expect_preAndPostEpochCalled() {
        // Arrange
        int numberOfEpochs = 5;
        TestContext context = new TestContext(numberOfEpochs);

        // Act
        context.sut.run();

        // Assert
        assertEquals(numberOfEpochs, context.sut.preEpochCallCount);
        assertEquals(numberOfEpochs, context.sut.postEpochCallCount);
    }

    @Test
    public void when_run_expect_trainSubEpochCalledAndResultPassedToListeners() {
        // Arrange
        int numberOfEpochs = 5;
        TestContext context = new TestContext(numberOfEpochs);

        // Act
        context.sut.run();

        // Assert
        assertEquals(numberOfEpochs, context.listener.statEntries.size());
        int[] expectedStepCounter = new int[] { 2, 4, 6, 8, 10 };
        double expectedReward = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) // reward from init
            + 1.0; // Reward from trainSubEpoch()
        for(int i = 0; i < numberOfEpochs; ++i) {
            IDataManager.StatEntry statEntry = context.listener.statEntries.get(i);
            assertEquals(expectedStepCounter[i], statEntry.getStepCounter());
            assertEquals(i, statEntry.getEpochCounter());
            assertEquals(expectedReward, statEntry.getReward(), 0.0001);
        }
    }

    @Test
    public void when_run_expect_trainSubEpochCalled() {
        // Arrange
        int numberOfEpochs = 5;
        TestContext context = new TestContext(numberOfEpochs);

        // Act
        context.sut.run();

        // Assert
        assertEquals(numberOfEpochs, context.sut.trainSubEpochParams.size());
        double[] expectedObservation = new double[] { 0.0, 2.0, 4.0, 6.0, 8.0 };
        for(int i = 0; i < context.sut.trainSubEpochParams.size(); ++i) {
            MockAsyncThread.TrainSubEpochParams params = context.sut.trainSubEpochParams.get(i);
            assertEquals(2, params.nstep);
            assertEquals(expectedObservation.length, params.obs.getData().shape()[1]);
            for(int j = 0; j < expectedObservation.length; ++j){
                assertEquals(expectedObservation[j], 255.0 * params.obs.getData().getDouble(j), 0.00001);
            }
        }
    }

    private static class TestContext {
        public final MockAsyncGlobal asyncGlobal = new MockAsyncGlobal();
        public final MockNeuralNet neuralNet = new MockNeuralNet();
        public final MockObservationSpace observationSpace = new MockObservationSpace();
        public final MockMDP mdp = new MockMDP(observationSpace);
        public final MockAsyncConfiguration config = new MockAsyncConfiguration(5, 10, 0, 0, 10, 0, 0, 0, 0, 0);
        public final TrainingListenerList listeners = new TrainingListenerList();
        public final MockTrainingListener listener = new MockTrainingListener();
        public final IHistoryProcessor.Configuration hpConf = new IHistoryProcessor.Configuration(5, 4, 4, 4, 4, 0, 0, 2);
        public final MockHistoryProcessor historyProcessor = new MockHistoryProcessor(hpConf);

        public final MockAsyncThread sut = new MockAsyncThread(asyncGlobal, 0, neuralNet, mdp, config, listeners);

        public TestContext(int numEpochs) {
            asyncGlobal.setMaxLoops(numEpochs);
            listeners.add(listener);
            sut.setHistoryProcessor(historyProcessor);
        }
    }

    public static class MockAsyncThread extends AsyncThread<MockEncodable, Integer, DiscreteSpace, MockNeuralNet> {

        public int preEpochCallCount = 0;
        public int postEpochCallCount = 0;

        private final MockAsyncGlobal asyncGlobal;
        private final MockNeuralNet neuralNet;
        private final AsyncConfiguration conf;

        private final List<TrainSubEpochParams> trainSubEpochParams = new ArrayList<TrainSubEpochParams>();

        public MockAsyncThread(MockAsyncGlobal asyncGlobal, int threadNumber, MockNeuralNet neuralNet, MDP mdp, AsyncConfiguration conf, TrainingListenerList listeners) {
            super(asyncGlobal, mdp, listeners, threadNumber, 0);

            this.asyncGlobal = asyncGlobal;
            this.neuralNet = neuralNet;
            this.conf = conf;
        }

        @Override
        protected void preEpoch() {
            ++preEpochCallCount;
            super.preEpoch();
        }

        @Override
        protected void postEpoch() {
            ++postEpochCallCount;
            super.postEpoch();
        }

        @Override
        protected MockNeuralNet getCurrent() {
            return neuralNet;
        }

        @Override
        protected IAsyncGlobal getAsyncGlobal() {
            return asyncGlobal;
        }

        @Override
        protected AsyncConfiguration getConf() {
            return conf;
        }

        @Override
        protected Policy getPolicy(MockNeuralNet net) {
            return null;
        }

        @Override
        protected SubEpochReturn trainSubEpoch(Observation obs, int nstep) {
            asyncGlobal.increaseCurrentLoop();
            trainSubEpochParams.add(new TrainSubEpochParams(obs, nstep));
            setStepCounter(getStepCounter() + nstep);
            return new SubEpochReturn(nstep, null, 1.0, 1.0);
        }

        @AllArgsConstructor
        @Getter
        public static class TrainSubEpochParams {
            Observation obs;
            int nstep;
        }
    }



}
