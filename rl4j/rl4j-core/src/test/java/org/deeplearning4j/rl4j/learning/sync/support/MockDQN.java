/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.learning.sync.support;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNetOutput;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;

public class MockDQN implements IDQN {

    private final double mult;

    public MockDQN() {
        this(1.0);
    }

    public MockDQN(double mult) {
        this.mult = mult;
    }

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
    public NeuralNetOutput output(INDArray batch) {
        NeuralNetOutput result = new NeuralNetOutput();
        INDArray data = batch;
        if(mult != 1.0) {
            data = data.dup().muli(mult);
        }
        result.put(CommonOutputNames.QValues, data);

        return result;
    }

    @Override
    public NeuralNetOutput output(Features features) {
        throw new UnsupportedOperationException();
    }

    @Override
    public NeuralNetOutput output(Observation observation) {
        return this.output(observation.getData());
    }

    @Override
    public INDArray[] outputAll(INDArray batch) {
        return new INDArray[0];
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
        throw new UnsupportedOperationException();
    }

    @Override
    public IDQN clone() {
        return null;
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
