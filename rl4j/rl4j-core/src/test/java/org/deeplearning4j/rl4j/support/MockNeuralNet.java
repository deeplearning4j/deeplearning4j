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

package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNetOutput;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MockNeuralNet implements NeuralNet {

    public int resetCallCount = 0;
    public int copyCallCount = 0;
    public List<INDArray> outputAllInputs = new ArrayList<INDArray>();

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
        ++resetCallCount;
    }

    @Override
    public INDArray[] outputAll(INDArray batch) {
        outputAllInputs.add(batch);
        return new INDArray[] { Nd4j.create(new double[] { outputAllInputs.size() }) };
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
    public void copyFrom(ITrainableNeuralNet from) {
        ++copyCallCount;
    }

    @Override
    public NeuralNet clone() {
        return this;
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