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
package org.deeplearning4j.rl4j.network;

import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An interface defining the output aspect of a {@link NeuralNet}.
 */
public interface IOutputNeuralNet {
    /**
     * Compute the output for the supplied observation. Multiple calls to output() with the same observation will
     * give the same output, even if the internal state has changed, until the network is reset or an operation
     * that modifies it is performed (See {@link ITrainableNeuralNet#fit}, {@link ITrainableNeuralNet#applyGradients},
     * and {@link ITrainableNeuralNet#copyFrom}).
     * @param observation An {@link Observation}
     * @return The ouptut of the network
     */
    NeuralNetOutput output(Observation observation);

    /**
     * Compute the output for the supplied batch.
     * @param batch
     * @return The ouptut of the network
     */
    NeuralNetOutput output(INDArray batch); // FIXME: Remove once legacy classes are gone

    /**
     * Compute the output for the supplied batch.
     * @param features A {@link Features} instance
     * @return The ouptut of the network
     */
    NeuralNetOutput output(Features features);

    /**
     * Clear the neural net of any previous state
     */
    void reset();

    /**
     * @return True if the neural net is a RNN
     */
    boolean isRecurrent();
}