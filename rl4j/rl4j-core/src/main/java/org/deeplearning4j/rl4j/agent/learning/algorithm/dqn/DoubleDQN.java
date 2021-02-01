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

package org.deeplearning4j.rl4j.agent.learning.algorithm.dqn;

import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DoubleDQN extends BaseDQNAlgorithm {

    private static final int ACTION_DIMENSION_IDX = 1;

    // In literature, this corresponds to: max<sub>a</sub> Q(s<sub>t+1</sub>, a)
    private INDArray maxActionsFromQNetworkNextObservation;

    public DoubleDQN(IOutputNeuralNet qNetwork,
                     IOutputNeuralNet targetQNetwork,
                     BaseTransitionTDAlgorithm.Configuration configuration) {
        super(qNetwork, targetQNetwork, configuration);
    }

    @Override
    protected void initComputation(Features features, Features nextFeatures) {
        super.initComputation(features, nextFeatures);

        maxActionsFromQNetworkNextObservation = Nd4j.argMax(qNetworkNextFeatures, ACTION_DIMENSION_IDX);
    }

    /**
     * In literature, this corresponds to:<br />
     *      Q(s<sub>t</sub>, a<sub>t</sub>) = R<sub>t+1</sub> + &gamma; * Q<sub>tar</sub>(s<sub>t+1</sub>, max<sub>a</sub> Q(s<sub>t+1</sub>, a))
     * @param batchIdx The index in the batch of the current transition
     * @param reward The reward of the current transition
     * @param isTerminal True if it's the last transition of the "game"
     * @return The estimated Q-Value
     */
    @Override
    protected double computeTarget(int batchIdx, double reward, boolean isTerminal) {
        double yTarget = reward;
        if (!isTerminal) {
            yTarget += gamma * targetQNetworkNextFeatures.getDouble(batchIdx, maxActionsFromQNetworkNextObservation.getInt(batchIdx));
        }

        return yTarget;
    }
}
