package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.deeplearning4j.rl4j.support.MockDataManager;
import org.deeplearning4j.rl4j.support.MockHistoryProcessor;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;

public class AsyncThreadTest {

    @Test
    public void refac_withoutHistoryProcessor_checkDataManagerCallsRemainTheSame() {
        // Arrange
        MockDataManager dataManager = new MockDataManager(false);
        MockAsyncGlobal asyncGlobal = new MockAsyncGlobal(10);
        MockNeuralNet neuralNet = new MockNeuralNet();
        MockObservationSpace observationSpace = new MockObservationSpace();
        MockMDP mdp = new MockMDP(observationSpace);
        MockAsyncConfiguration config = new MockAsyncConfiguration(10, 2);
        MockAsyncThread sut = new MockAsyncThread(asyncGlobal, 0, neuralNet, mdp, config, dataManager);

        // Act
        sut.run();

        // Assert
        assertEquals(4, dataManager.statEntries.size());

        IDataManager.StatEntry entry = dataManager.statEntries.get(0);
        assertEquals(2, entry.getStepCounter());
        assertEquals(0, entry.getEpochCounter());
        assertEquals(2.0, entry.getReward(), 0.0);

        entry = dataManager.statEntries.get(1);
        assertEquals(4, entry.getStepCounter());
        assertEquals(1, entry.getEpochCounter());
        assertEquals(2.0, entry.getReward(), 0.0);

        entry = dataManager.statEntries.get(2);
        assertEquals(6, entry.getStepCounter());
        assertEquals(2, entry.getEpochCounter());
        assertEquals(2.0, entry.getReward(), 0.0);

        entry = dataManager.statEntries.get(3);
        assertEquals(8, entry.getStepCounter());
        assertEquals(3, entry.getEpochCounter());
        assertEquals(2.0, entry.getReward(), 0.0);

        assertEquals(0, dataManager.isSaveDataCallCount);
        assertEquals(0, dataManager.getVideoDirCallCount);
    }

    @Test
    public void refac_withHistoryProcessor_isSaveFalse_checkDataManagerCallsRemainTheSame() {
        // Arrange
        MockDataManager dataManager = new MockDataManager(false);
        MockAsyncGlobal asyncGlobal = new MockAsyncGlobal(10);
        MockNeuralNet neuralNet = new MockNeuralNet();
        MockObservationSpace observationSpace = new MockObservationSpace();
        MockMDP mdp = new MockMDP(observationSpace);
        MockAsyncConfiguration asyncConfig = new MockAsyncConfiguration(10, 2);

        IHistoryProcessor.Configuration hpConfig = IHistoryProcessor.Configuration.builder()
            .build();
        MockHistoryProcessor hp = new MockHistoryProcessor(hpConfig);


        MockAsyncThread sut = new MockAsyncThread(asyncGlobal, 0, neuralNet, mdp, asyncConfig, dataManager);
        sut.setHistoryProcessor(hp);

        // Act
        sut.run();

        // Assert
        assertEquals(9, dataManager.statEntries.size());

        for(int i = 0; i < 9; ++i) {
            IDataManager.StatEntry entry = dataManager.statEntries.get(i);
            assertEquals(i + 1, entry.getStepCounter());
            assertEquals(i, entry.getEpochCounter());
            assertEquals(1.0, entry.getReward(), 0.0);
        }

        assertEquals(10, dataManager.isSaveDataCallCount);
        assertEquals(0, dataManager.getVideoDirCallCount);
    }

    @Test
    public void refac_withHistoryProcessor_isSaveTrue_checkDataManagerCallsRemainTheSame() {
        // Arrange
        MockDataManager dataManager = new MockDataManager(true);
        MockAsyncGlobal asyncGlobal = new MockAsyncGlobal(10);
        MockNeuralNet neuralNet = new MockNeuralNet();
        MockObservationSpace observationSpace = new MockObservationSpace();
        MockMDP mdp = new MockMDP(observationSpace);
        MockAsyncConfiguration asyncConfig = new MockAsyncConfiguration(10, 2);

        IHistoryProcessor.Configuration hpConfig = IHistoryProcessor.Configuration.builder()
                .build();
        MockHistoryProcessor hp = new MockHistoryProcessor(hpConfig);


        MockAsyncThread sut = new MockAsyncThread(asyncGlobal, 0, neuralNet, mdp, asyncConfig, dataManager);
        sut.setHistoryProcessor(hp);

        // Act
        sut.run();

        // Assert
        assertEquals(9, dataManager.statEntries.size());

        for(int i = 0; i < 9; ++i) {
            IDataManager.StatEntry entry = dataManager.statEntries.get(i);
            assertEquals(i + 1, entry.getStepCounter());
            assertEquals(i, entry.getEpochCounter());
            assertEquals(1.0, entry.getReward(), 0.0);
        }

        assertEquals(1, dataManager.isSaveDataCallCount);
        assertEquals(1, dataManager.getVideoDirCallCount);
    }

    public static class MockAsyncGlobal implements IAsyncGlobal {

        private final int maxLoops;
        private int currentLoop = 0;

        public MockAsyncGlobal(int maxLoops) {

            this.maxLoops = maxLoops;
        }

        @Override
        public boolean isRunning() {
            return true;
        }

        @Override
        public void setRunning(boolean value) {

        }

        @Override
        public boolean isTrainingComplete() {
            return ++currentLoop >= maxLoops;
        }

        @Override
        public void start() {

        }

        @Override
        public AtomicInteger getT() {
            return null;
        }

        @Override
        public NeuralNet getCurrent() {
            return null;
        }

        @Override
        public NeuralNet getTarget() {
            return null;
        }

        @Override
        public void enqueue(Gradient[] gradient, Integer nstep) {

        }
    }

    public static class MockAsyncThread extends AsyncThread {

        IAsyncGlobal asyncGlobal;
        private final MockNeuralNet neuralNet;
        private final MDP mdp;
        private final AsyncConfiguration conf;
        private final IDataManager dataManager;

        public MockAsyncThread(IAsyncGlobal asyncGlobal, int threadNumber, MockNeuralNet neuralNet, MDP mdp, AsyncConfiguration conf, IDataManager dataManager) {
            super(asyncGlobal, threadNumber);

            this.asyncGlobal = asyncGlobal;
            this.neuralNet = neuralNet;
            this.mdp = mdp;
            this.conf = conf;
            this.dataManager = dataManager;
        }

        @Override
        protected NeuralNet getCurrent() {
            return neuralNet;
        }

        @Override
        protected int getThreadNumber() {
            return 0;
        }

        @Override
        protected IAsyncGlobal getAsyncGlobal() {
            return asyncGlobal;
        }

        @Override
        protected MDP getMdp() {
            return mdp;
        }

        @Override
        protected AsyncConfiguration getConf() {
            return conf;
        }

        @Override
        protected IDataManager getDataManager() {
            return dataManager;
        }

        @Override
        protected Policy getPolicy(NeuralNet net) {
            return null;
        }

        @Override
        protected SubEpochReturn trainSubEpoch(Encodable obs, int nstep) {
            return new SubEpochReturn(1, null, 1.0, 1.0);
        }
    }

    public static class MockNeuralNet implements NeuralNet {

        @Override
        public NeuralNetwork[] getNeuralNetworks() {
            return new NeuralNetwork[0];
        }

        @Override
        public boolean isRecurrent() {
            return false;
        }

        @Override
        public void reset() {

        }

        @Override
        public INDArray[] outputAll(INDArray batch) {
            return new INDArray[0];
        }

        @Override
        public NeuralNet clone() {
            return null;
        }

        @Override
        public void copy(NeuralNet from) {

        }

        @Override
        public Gradient[] gradient(INDArray input, INDArray[] labels) {
            return new Gradient[0];
        }

        @Override
        public void fit(INDArray input, INDArray[] labels) {

        }

        @Override
        public void applyGradient(Gradient[] gradients, int batchSize) {

        }

        @Override
        public double getLatestScore() {
            return 0;
        }

        @Override
        public void save(OutputStream os) throws IOException {

        }

        @Override
        public void save(String filename) throws IOException {

        }
    }

    public static class MockEncodable implements Encodable {

        private final int value;

        public MockEncodable(int value) {

            this.value = value;
        }

        @Override
        public double[] toArray() {
            return new double[] { value };
        }
    }

    public static class MockObservationSpace implements ObservationSpace {

        @Override
        public String getName() {
            return null;
        }

        @Override
        public int[] getShape() {
            return new int[] { 1 };
        }

        @Override
        public INDArray getLow() {
            return null;
        }

        @Override
        public INDArray getHigh() {
            return null;
        }
    }

    public static class MockMDP implements MDP<MockEncodable, Integer, DiscreteSpace> {

        private final DiscreteSpace actionSpace;
        private int currentObsValue = 0;
        private final ObservationSpace observationSpace;

        public MockMDP(ObservationSpace observationSpace) {
            actionSpace = new DiscreteSpace(5);
            this.observationSpace = observationSpace;
        }

        @Override
        public ObservationSpace getObservationSpace() {
            return observationSpace;
        }

        @Override
        public DiscreteSpace getActionSpace() {
            return actionSpace;
        }

        @Override
        public MockEncodable reset() {
            return new MockEncodable(++currentObsValue);
        }

        @Override
        public void close() {

        }

        @Override
        public StepReply<MockEncodable> step(Integer obs) {
            return new StepReply<MockEncodable>(new MockEncodable(obs), (double)obs, isDone(), null);
        }

        @Override
        public boolean isDone() {
            return false;
        }

        @Override
        public MDP newInstance() {
            return null;
        }
    }

    public static class MockAsyncConfiguration implements AsyncConfiguration {

        private final int nStep;
        private final int maxEpochStep;

        public MockAsyncConfiguration(int nStep, int maxEpochStep) {
            this.nStep = nStep;

            this.maxEpochStep = maxEpochStep;
        }

        @Override
        public int getSeed() {
            return 0;
        }

        @Override
        public int getMaxEpochStep() {
            return maxEpochStep;
        }

        @Override
        public int getMaxStep() {
            return 0;
        }

        @Override
        public int getNumThread() {
            return 0;
        }

        @Override
        public int getNstep() {
            return nStep;
        }

        @Override
        public int getTargetDqnUpdateFreq() {
            return 0;
        }

        @Override
        public int getUpdateStart() {
            return 0;
        }

        @Override
        public double getRewardFactor() {
            return 0;
        }

        @Override
        public double getGamma() {
            return 0;
        }

        @Override
        public double getErrorClamp() {
            return 0;
        }
    }

}
