/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning;

import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * A base helper class for the n-step Q-Learning update algorithm. The algorithm is the same whether it's used with a RNN or
 * not but, the shape of INDArrays are different. This class, {@link NonRecurrentNStepQLearningHelper},
 * and {@link RecurrentNStepQLearningHelper} handle the differences.
 */
public abstract class NStepQLearningHelper {

    /**
     * Create a feature INDArray, filled with the observations from the trainingBatch
     * @param trainingBatch An experience training batch
     * @return A INDArray filled with the observations from the trainingBatch
     */
    public INDArray createFeatures(List<StateActionPair<Integer>> trainingBatch) {
        int size = trainingBatch.size();
        long[] observationShape = trainingBatch.get(0).getObservation().getData().shape();
        INDArray features = createFeatureArray(size, observationShape);
        for(int i = 0; i < size; ++i) {
            setFeature(features, i, trainingBatch.get(i).getObservation().getData());
        }

        return features;
    }
    protected abstract INDArray createFeatureArray(int size, long[] observationShape);
    protected abstract void setFeature(INDArray features, long idx, INDArray data);

    /**
     * Get the expected Q value given a training batch index from the pre-computed Q values
     * @param allExpectedQValues A INDArray containg all pre-computed Q values
     * @param idx The training batch index
     * @return The expected Q value
     */
    public abstract INDArray getExpectedQValues(INDArray allExpectedQValues, int idx);

    /**
     * Create an empty INDArray to be used as the Q values array
     * @param trainingBatchSize the size of the training batch
     * @return An empty Q values array
     */
    public abstract INDArray createLabels(int trainingBatchSize);

    /**
     * Set the label in the Q values array for a given training batch index
     * @param labels The Q values array
     * @param idx The training batch index
     * @param data The updated Q values to set
     */
    public abstract void setLabels(INDArray labels, long idx, INDArray data);

    /**
     * Get the expected Q values for the last element of the training batch, estimated using the target network.
     * @param target The target network
     * @param trainingBatch An experience training batch
     * @return A INDArray filled with the observations from the trainingBatch
     * @return The expected Q values for the last element of the training batch
     */
    public abstract INDArray getTargetExpectedQValuesOfLast(IOutputNeuralNet target, List<StateActionPair<Integer>> trainingBatch, INDArray features);
}