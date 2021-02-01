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

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Value;
import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

/**
 * This abstract class is a base implementation of {@link ITrainableNeuralNet} for typical networks.
 * This implementation caches the outputs of the network, until the network is changed (fit(), applyGradients(), and copyFrom()) or reset()
 * This is not only a performance optimization; When using recurrent networks, the same observation should always give
 * the same output (in the policy and the update algorithm). Storing that output is the easiest and fastest.
 * @param <NET_TYPE>
 */
public abstract class BaseNetwork<NET_TYPE extends BaseNetwork>
        implements ITrainableNeuralNet<NET_TYPE> {

    @Getter(AccessLevel.PROTECTED)
    private final INetworkHandler networkHandler;

    private final Map<Observation, NeuralNetOutput> neuralNetOutputCache = new HashMap<Observation, NeuralNetOutput>();

    protected BaseNetwork(INetworkHandler networkHandler) {
        this.networkHandler = networkHandler;
    }

    /**
     * @return True if the network is recurrent.
     */
    public boolean isRecurrent() {
        return networkHandler.isRecurrent();
    }

    /**
     * Fit the network using the featuresLabels
     * @param featuresLabels The feature-labels
     */
    @Override
    public void fit(FeaturesLabels featuresLabels) {
        invalidateCache();
        networkHandler.performFit(featuresLabels);
    }

    /**
     * Compute the gradients from the featuresLabels
     * @param featuresLabels The feature-labels
     * @return A {@link Gradients} instance
     */
    @Override
    public Gradients computeGradients(FeaturesLabels featuresLabels) {
        networkHandler.performGradientsComputation(featuresLabels);
        networkHandler.notifyGradientCalculation();
        Gradients results = new Gradients(featuresLabels.getBatchSize());
        networkHandler.fillGradientsResponse(results);

        return results;
    }

    /**
     * Applies the {@link Gradients}
     * @param gradients the gradients to be applied
     */
    @Override
    public void applyGradients(Gradients gradients) {
        invalidateCache();
        networkHandler.applyGradient(gradients, gradients.getBatchSize());
        networkHandler.notifyIterationDone();
    }

    /**
     * Computes the output from an observation or get the previously computed one if found in the cache.
     * @param observation An {@link Observation}
     * @return a {@link NeuralNetOutput} instance
     */
    @Override
    public NeuralNetOutput output(Observation observation) {
        NeuralNetOutput result = neuralNetOutputCache.get(observation);
        if(result == null) {
            if(isRecurrent()) {
                result = packageResult(networkHandler.recurrentStepOutput(observation));
            } else {
                result = packageResult(networkHandler.stepOutput(observation));
            }

            neuralNetOutputCache.put(observation, result);
        }

        return result;
    }

    protected abstract NeuralNetOutput packageResult(INDArray[] output);

    /**
     * Compute the output for a batch.
     * Note: The current state is ignored if used with a recurrent network
     * @param batch
     * @return a {@link NeuralNetOutput} instance
     */
    public NeuralNetOutput output(INDArray batch) {
        // TODO: Remove when legacy code is gone
        throw new NotImplementedException("output(INDArray): should use output(Observation) or output(Features)");
    }

    /**
     * Compute the output for a batch.
     * Note: The current state is ignored if used with a recurrent network
     * @param features
     * @return a {@link NeuralNetOutput} instance
     */
    public NeuralNetOutput output(Features features) {
        return packageResult(networkHandler.batchOutput(features));
    }


    /**
     * Resets the cache and the state of the network
     */
    @Override
    public void reset() {
        invalidateCache();
        if(isRecurrent()) {
            networkHandler.resetState();
        }
    }

    protected void invalidateCache() {
        neuralNetOutputCache.clear();
    }

    /**
     * Copy the network parameters from the argument to the current network and clear the cache
     * @param from The network that will be the source of the copy.
     */
    public void copyFrom(BaseNetwork from) {
        reset();
        networkHandler.copyFrom(from.networkHandler);
    }

    @Value
    protected static class ModelCounters {
        int iterationCount;
        int epochCount;
    }

    public abstract NET_TYPE clone();
}
