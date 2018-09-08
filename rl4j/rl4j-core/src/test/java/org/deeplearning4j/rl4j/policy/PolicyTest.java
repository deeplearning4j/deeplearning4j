/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.io.OutputStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

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
        public void copy(NN from) {
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
    }

    @Test
    public void testACPolicy() throws Exception {
        ComputationGraph cg = new ComputationGraph(new NeuralNetConfiguration.Builder().seed(444).graphBuilder().addInputs("input")
                .addLayer("output", new OutputLayer.Builder().nOut(1).lossFunction(LossFunctions.LossFunction.XENT).activation(Activation.SIGMOID).build(), "input").setOutputs("output").build());
        MultiLayerNetwork mln = new MultiLayerNetwork(new NeuralNetConfiguration.Builder().seed(555).list()
                .layer(0, new OutputLayer.Builder().nOut(1).lossFunction(LossFunctions.LossFunction.XENT).activation(Activation.SIGMOID).build()).build());

        ACPolicy policy = new ACPolicy(new DummyAC(cg));
        assertNotNull(policy.rd);

        policy = new ACPolicy(new DummyAC(mln));
        assertNotNull(policy.rd);

        INDArray input = Nd4j.create(new double[] {1.0, 0.0});
        for (int i = 0; i < 100; i++) {
            assertEquals(0, (int)policy.nextAction(input));
        }

        input = Nd4j.create(new double[] {0.0, 1.0});
        for (int i = 0; i < 100; i++) {
            assertEquals(1, (int)policy.nextAction(input));
        }

        input = Nd4j.create(new double[] {0.1, 0.2, 0.3, 0.4});
        int[] count = new int[4];
        for (int i = 0; i < 100; i++) {
            count[policy.nextAction(input)]++;
        }
        System.out.println(count[0] + " " + count[1] + " " + count[2] + " " + count[3]);
        assertTrue(count[0] < 20);
        assertTrue(count[1] < 30);
        assertTrue(count[2] < 40);
        assertTrue(count[3] < 50);
    }
}
