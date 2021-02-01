/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNetOutput;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.support.*;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.io.OutputStream;

import static org.junit.Assert.*;

/**
 *
 * @author saudet
 */
public class PolicyTest {

    public static class DummyAC<NN extends DummyAC> implements IActorCritic<NN> {
        NeuralNetwork nn;
        DummyAC(NeuralNetwork nn) {
            this.nn = nn;
        }

        @Override
        public NeuralNetwork[] getNeuralNetworks() {
            return new NeuralNetwork[] { nn };
        }

        @Override
        public boolean isRecurrent() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void reset() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void fit(INDArray input, INDArray[] labels) {
            throw new UnsupportedOperationException();
        }

        @Override
        public INDArray[] outputAll(INDArray batch) {
            return new INDArray[] {batch, batch};
        }

        @Override
        public NN clone() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void fit(FeaturesLabels featuresLabels) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Gradients computeGradients(FeaturesLabels featuresLabels) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void applyGradients(Gradients gradients) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void copyFrom(NN from) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Gradient[] gradient(INDArray input, INDArray[] labels) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void applyGradient(Gradient[] gradient, int batchSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void save(OutputStream streamValue, OutputStream streamPolicy) throws IOException {
            throw new UnsupportedOperationException();
        }

        @Override
        public void save(String pathValue, String pathPolicy) throws IOException {
            throw new UnsupportedOperationException();
        }

        @Override
        public double getLatestScore() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void save(OutputStream os) throws IOException {
            throw new UnsupportedOperationException();
        }

        @Override
        public void save(String filename) throws IOException {
            throw new UnsupportedOperationException();
        }

        @Override
        public NeuralNetOutput output(Observation observation) {
            throw new UnsupportedOperationException();
        }

        @Override
        public NeuralNetOutput output(INDArray batch) {
            throw new UnsupportedOperationException();
        }

        @Override
        public NeuralNetOutput output(Features features) {
            throw new UnsupportedOperationException();
        }
    }

    @Test
    public void testACPolicy() throws Exception {
        ComputationGraph cg = new ComputationGraph(new NeuralNetConfiguration.Builder().seed(444).graphBuilder().addInputs("input")
                .addLayer("output", new OutputLayer.Builder().nOut(1).lossFunction(LossFunctions.LossFunction.XENT).activation(Activation.SIGMOID).build(), "input").setOutputs("output").build());
        MultiLayerNetwork mln = new MultiLayerNetwork(new NeuralNetConfiguration.Builder().seed(555).list()
                .layer(0, new OutputLayer.Builder().nOut(1).lossFunction(LossFunctions.LossFunction.XENT).activation(Activation.SIGMOID).build()).build());

        ACPolicy policy = new ACPolicy(new DummyAC(mln), true, Nd4j.getRandom());

        INDArray input = Nd4j.create(new double[] {1.0, 0.0}, new long[]{1,2});
        for (int i = 0; i < 100; i++) {
            assertEquals(0, (int)policy.nextAction(input));
        }

        input = Nd4j.create(new double[] {0.0, 1.0}, new long[]{1,2});
        for (int i = 0; i < 100; i++) {
            assertEquals(1, (int)policy.nextAction(input));
        }

        input = Nd4j.create(new double[] {0.1, 0.2, 0.3, 0.4}, new long[]{1, 4});
        int[] count = new int[4];
        for (int i = 0; i < 100; i++) {
            count[policy.nextAction(input)]++;
        }
//        System.out.println(count[0] + " " + count[1] + " " + count[2] + " " + count[3]);
        assertTrue(count[0] < 20);
        assertTrue(count[1] < 30);
        assertTrue(count[2] < 40);
        assertTrue(count[3] < 50);
    }

    @Test
    public void refacPolicyPlay() {
        // Arrange
        MockObservationSpace observationSpace = new MockObservationSpace();
        MockDQN dqn = new MockDQN();
        MockRandom random = new MockRandom(new double[] {
                0.7309677600860596,
                0.8314409852027893,
                0.2405363917350769,
                0.6063451766967773,
                0.6374173760414124,
                0.3090505599975586,
                0.5504369735717773,
                0.11700659990310669
            },
            new int[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 });
        MockMDP mdp = new MockMDP(observationSpace, 30, random);

        QLearningConfiguration conf = QLearningConfiguration.builder()
                .seed(0L)
                .maxEpochStep(0)
                .maxStep(0)
                .expRepMaxSize(5)
                .batchSize(1)
                .targetDqnUpdateFreq(0)
                .updateStart(0)
                .rewardFactor(1.0)
                .gamma(0)
                .errorClamp(0)
                .minEpsilon(0)
                .epsilonNbStep(0)
                .doubleDQN(true)
                .build();

        MockNeuralNet nnMock = new MockNeuralNet();
        IHistoryProcessor.Configuration hpConf = new IHistoryProcessor.Configuration(5, 4, 4, 4, 4, 0, 0, 2);
        MockRefacPolicy sut = new MockRefacPolicy(nnMock, observationSpace.getShape(), hpConf.getSkipFrame(), hpConf.getHistoryLength());
        MockHistoryProcessor hp = new MockHistoryProcessor(hpConf);

        // Act
        double totalReward = sut.play(mdp, hp);

        // Assert
        assertEquals(1, nnMock.resetCallCount);
        assertEquals(465.0, totalReward, 0.0001);

        // MDP
        assertEquals(1, mdp.resetCount);
        assertEquals(30, mdp.actions.size());
        for(int i = 0; i < mdp.actions.size(); ++i) {
            assertEquals(0, (int)mdp.actions.get(i));
        }

        // DQN
        assertEquals(0, dqn.fitParams.size());
        assertEquals(0, dqn.outputParams.size());
    }

    public static class MockRefacPolicy extends Policy<Integer> {

        private NeuralNet neuralNet;
        private final int[] shape;
        private final int skipFrame;
        private final int historyLength;

        public MockRefacPolicy(NeuralNet neuralNet, int[] shape, int skipFrame, int historyLength) {
            this.neuralNet = neuralNet;
            this.shape = shape;
            this.skipFrame = skipFrame;
            this.historyLength = historyLength;
        }

        @Override
        public NeuralNet getNeuralNet() {
            return neuralNet;
        }

        @Override
        public Integer nextAction(Observation obs) {
            return nextAction(obs.getData());
        }

        @Override
        public Integer nextAction(INDArray input) {
            return (int)input.getDouble(0);
        }

        @Override
        protected <MockObservation extends Encodable, AS extends ActionSpace<Integer>> Learning.InitMdp<Observation> refacInitMdp(LegacyMDPWrapper<MockObservation, Integer, AS> mdpWrapper, IHistoryProcessor hp) {
            mdpWrapper.setTransformProcess(MockMDP.buildTransformProcess(skipFrame, historyLength));
            return super.refacInitMdp(mdpWrapper, hp);
        }
    }
}
