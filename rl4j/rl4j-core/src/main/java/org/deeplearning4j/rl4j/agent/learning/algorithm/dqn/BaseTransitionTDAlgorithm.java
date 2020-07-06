/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.agent.learning.algorithm.dqn;

import org.deeplearning4j.rl4j.agent.learning.algorithm.IUpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.network.CommonLabelNames;
import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * The base of all {@link Transition Transition-based} TD algorithms.
 */
public abstract class BaseTransitionTDAlgorithm implements IUpdateAlgorithm<FeaturesLabels, Transition<Integer>> {

    protected final IOutputNeuralNet qNetwork;
    protected final double gamma;

    private final double errorClamp;
    private final boolean isClamped;

    /**
     *
     * @param qNetwork The Q-Network
     * @param gamma The discount factor
     * @param errorClamp Will prevent the new Q-Value from being farther than <i>errorClamp</i> away from the previous value. Double.NaN will disable the clamping.
     */
    protected BaseTransitionTDAlgorithm(IOutputNeuralNet qNetwork, double gamma, double errorClamp) {
        this.qNetwork = qNetwork;
        this.gamma = gamma;

        this.errorClamp = errorClamp;
        isClamped = !Double.isNaN(errorClamp);
    }

    /**
     *
     * @param qNetwork The Q-Network
     * @param gamma The discount factor
     * Note: Error clamping is disabled with this ctor
     */
    protected BaseTransitionTDAlgorithm(IOutputNeuralNet qNetwork, double gamma) {
        this(qNetwork, gamma, Double.NaN);
    }

    /**
     * Called just before the calculation starts
     * @param observations A INDArray of all observations stacked on dimension 0
     * @param nextObservations A INDArray of all next observations stacked on dimension 0
     */
    protected void initComputation(INDArray observations, INDArray nextObservations) {
        // Do nothing
    }

    /**
     * Compute the new estimated Q-Value for every transition in the batch
     *
     * @param batchIdx The index in the batch of the current transition
     * @param reward The reward of the current transition
     * @param isTerminal True if it's the last transition of the "game"
     * @return The estimated Q-Value
     */
    protected abstract double computeTarget(int batchIdx, double reward, boolean isTerminal);

    @Override
    public FeaturesLabels compute(List<Transition<Integer>> transitions) {

        int size = transitions.size();

        INDArray observations = Transition.buildStackedObservations(transitions);
        INDArray nextObservations = Transition.buildStackedNextObservations(transitions);

        initComputation(observations, nextObservations);

        INDArray updatedQValues = qNetwork.output(observations);
        for (int i = 0; i < size; ++i) {
            Transition<Integer> transition = transitions.get(i);
            double yTarget = computeTarget(i, transition.getReward(), transition.isTerminal());

            if(isClamped) {
                double previousQValue = updatedQValues.getDouble(i, transition.getAction());
                double lowBound = previousQValue - errorClamp;
                double highBound = previousQValue + errorClamp;
                yTarget = Math.min(highBound, Math.max(yTarget, lowBound));
            }
            updatedQValues.putScalar(i, transition.getAction(), yTarget);
        }

        FeaturesLabels featuresLabels = new FeaturesLabels(observations);
        featuresLabels.putLabels(CommonLabelNames.QValues, updatedQValues);

        return featuresLabels;
    }
}
