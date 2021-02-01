/*
 *  ******************************************************************************
 *  *
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
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An interface defining operations that {@link BaseNetwork} need to do on different network implementations
 * (see {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork}, {@link org.deeplearning4j.nn.graph.ComputationGraph})
 * and networks composed of other networks (see {@link CompoundNetworkHandler}
 */
public interface INetworkHandler {
    /**
     * @return true if the network is recurrent
     */
    boolean isRecurrent();

    /**
     * Will notify the network that a gradient calculation has been performed.
     */
    void notifyGradientCalculation();

    /**
     * Will notify the network that a gradient has been applied
     */
    void notifyIterationDone();

    /**
     * Perform a fit on the network.
     * @param featuresLabels The features-labels
     */
    void performFit(FeaturesLabels featuresLabels);

    /**
     * Compute the gradients from the features-labels
     * @param featuresLabels The features-labels
     */
    void performGradientsComputation(FeaturesLabels featuresLabels);

    /**
     * Fill the supplied gradients with the results of the last gradients computation
     * @param gradients The {@link Gradients} to fill
     */
    void fillGradientsResponse(Gradients gradients);

    /**
     * Will apply the gradients to the network
     * @param gradients The {@link Gradients} to apply
     * @param batchSize The batch size
     */
    void applyGradient(Gradients gradients, long batchSize);

    /**
     * @param observation An {@link Observation}
     * @return The output of the observation computed with the current network state. (i.e. not cached)
     */
    INDArray[] recurrentStepOutput(Observation observation);

    /**
     * @param observation An {@link Observation}
     * @return The output of the observation computed without using or updating the network state.
     */
    INDArray[] stepOutput(Observation observation);

    /**
     * Compute the output of a batch
     * @param features A {@link Features} instance
     * @return The output of the batch. The current state of the network is not used or changed.
     */
    INDArray[] batchOutput(Features features);

    /**
     * Clear all network state.
     */
    void resetState();

    /**
     * @return An identical copy of the current instance.
     */
    INetworkHandler clone();

    /**
     * Copies the parameter of another network to the instance.
     * @param from
     */
    void copyFrom(INetworkHandler from);
}
