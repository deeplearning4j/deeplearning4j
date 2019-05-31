package org.deeplearning4j.rl4j.learning.sync.qlearning;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.TestMDP;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscrete;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.OutputStream;

public class TestQLearningDiscrete extends QLearningDiscrete<TestMDP.TestObservation> {

    public TestQLearningDiscrete(MDP<TestMDP.TestObservation, Integer, DiscreteSpace> mdp, IHistoryProcessor hp) throws IOException {
            super(mdp,
                  new TestQLearningDiscrete.TestDQN(),
                  new QLConfiguration(123, 1, 1, 1, 1, 1, 1, 1.0, 1, 1, 1, 1, true),
                  new DataManager(),
                  1);
            setHistoryProcessor(hp);
    }

    public QLStepReturn<TestMDP.TestObservation> testTrainStep(TestMDP.TestObservation obs) {
        return trainStep(obs);
    }

    public static class TestObservation implements Encodable{

        @Override
        public double[] toArray() {
            return new double[0];
        }
    }

    public static class TestDQN implements IDQN {

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
        public void fit(INDArray input, INDArray labels) {

        }

        @Override
        public void fit(INDArray input, INDArray[] labels) {

        }

        @Override
        public INDArray output(INDArray batch) {
            return Nd4j.create(new int[] { 1, 1 });
        }

        @Override
        public INDArray[] outputAll(INDArray batch) {
            return new INDArray[0];
        }

        @Override
        public IDQN clone() {
            return null;
        }

        @Override
        public void copy(NeuralNet from) {

        }

        @Override
        public void copy(IDQN from) {

        }

        @Override
        public Gradient[] gradient(INDArray input, INDArray label) {
            return new Gradient[0];
        }

        @Override
        public Gradient[] gradient(INDArray input, INDArray[] label) {
            return new Gradient[0];
        }

        @Override
        public void applyGradient(Gradient[] gradient, int batchSize) {

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
}
