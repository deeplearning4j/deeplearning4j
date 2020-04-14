package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.support.MockAsyncGlobal;
import org.deeplearning4j.rl4j.support.MockMDP;
import org.deeplearning4j.rl4j.support.MockObservationSpace;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class A3CUpdateAlgorithmTest {

    @Test
    public void refac_calcGradient_non_terminal() {
        // Arrange
        double gamma = 0.9;
        MockObservationSpace observationSpace = new MockObservationSpace(new int[] { 5 });
        MockMDP mdpMock = new MockMDP(observationSpace);
        MockActorCritic actorCriticMock = new MockActorCritic();
        MockAsyncGlobal<IActorCritic> asyncGlobalMock = new MockAsyncGlobal<IActorCritic>(actorCriticMock);
        A3CUpdateAlgorithm sut = new A3CUpdateAlgorithm(asyncGlobalMock, observationSpace.getShape(), mdpMock.getActionSpace().getSize(), -1, gamma);


        INDArray[] originalObservations = new INDArray[] {
                Nd4j.create(new double[] { 0.0, 0.1, 0.2, 0.3, 0.4 }),
                Nd4j.create(new double[] { 1.0, 1.1, 1.2, 1.3, 1.4 }),
                Nd4j.create(new double[] { 2.0, 2.1, 2.2, 2.3, 2.4 }),
                Nd4j.create(new double[] { 3.0, 3.1, 3.2, 3.3, 3.4 }),
        };
        int[] actions = new int[] { 0, 1, 2, 1 };
        double[] rewards = new double[] { 0.1, 1.0, 10.0, 100.0 };

        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>();
        for(int i = 0; i < originalObservations.length; ++i) {
            experience.add(new StateActionPair<>(new Observation(originalObservations[i]), actions[i], rewards[i], false));
        }

        // Act
        sut.computeGradients(actorCriticMock, experience);

        // Assert
        assertEquals(1, actorCriticMock.gradientParams.size());

        // Inputs
        INDArray input = actorCriticMock.gradientParams.get(0).getLeft();
        for(int i = 0; i < 4; ++i) {
            for(int j = 0; j < 5; ++j) {
                assertEquals(i + j / 10.0, input.getDouble(i, j), 0.00001);
            }
        }

        INDArray targets = actorCriticMock.gradientParams.get(0).getRight()[0];
        INDArray logSoftmax = actorCriticMock.gradientParams.get(0).getRight()[1];

        assertEquals(4, targets.shape()[0]);
        assertEquals(1, targets.shape()[1]);

        // FIXME: check targets values once fixed

        assertEquals(4, logSoftmax.shape()[0]);
        assertEquals(5, logSoftmax.shape()[1]);

        // FIXME: check logSoftmax values once fixed

    }

    public class MockActorCritic implements IActorCritic {

        public final List<Pair<INDArray, INDArray[]>> gradientParams = new ArrayList<>();

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
        public void fit(INDArray input, INDArray[] labels) {

        }

        @Override
        public INDArray[] outputAll(INDArray batch) {
            return new INDArray[] { batch.mul(-1.0) };
        }

        @Override
        public IActorCritic clone() {
            return this;
        }

        @Override
        public void copy(NeuralNet from) {

        }

        @Override
        public void copy(IActorCritic from) {

        }

        @Override
        public Gradient[] gradient(INDArray input, INDArray[] labels) {
            gradientParams.add(new Pair<INDArray, INDArray[]>(input, labels));
            return new Gradient[0];
        }

        @Override
        public void applyGradient(Gradient[] gradient, int batchSize) {

        }

        @Override
        public void save(OutputStream streamValue, OutputStream streamPolicy) throws IOException {

        }

        @Override
        public void save(String pathValue, String pathPolicy) throws IOException {

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
