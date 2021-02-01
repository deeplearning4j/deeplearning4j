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
package org.deeplearning4j.rl4j.builder;

import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;

/**
 * An interface that abstract what the different networks are depending on the setup (sync vs async)
 */
public interface INetworksHandler {
    /**
     * @return The global shared target parameters &theta;<sup>&ndash;</sup>
     */
    ITrainableNeuralNet getTargetNetwork();

    /**
     * @return The thread-specific parameters &theta;'
     */
    ITrainableNeuralNet getThreadCurrentNetwork();

    /**
     * @return The global shared parameters &theta;
     */
    ITrainableNeuralNet getGlobalCurrentNetwork();

    /**
     * Perform the required changes before a new IAgentLearner is built
     */
    void resetForNewBuild();
}
