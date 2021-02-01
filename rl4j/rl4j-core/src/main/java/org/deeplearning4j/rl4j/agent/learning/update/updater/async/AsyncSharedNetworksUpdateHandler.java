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
package org.deeplearning4j.rl4j.agent.learning.update.updater.async;

import lombok.Getter;
import lombok.NonNull;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.nd4j.common.base.Preconditions;

/**
 * A class that applies updates to the global current network and synchronize the target network
 */
public class AsyncSharedNetworksUpdateHandler {

    @Getter
    private final ITrainableNeuralNet globalCurrent;

    private final ITrainableNeuralNet target;
    private final int targetUpdateFrequency;

    private int updateCount = 0;

    public AsyncSharedNetworksUpdateHandler(@NonNull ITrainableNeuralNet globalCurrent,
                                            @NonNull NeuralNetUpdaterConfiguration configuration) {
        this.globalCurrent = globalCurrent;
        this.target = null;
        this.targetUpdateFrequency = 0;
    }

    public AsyncSharedNetworksUpdateHandler(@NonNull ITrainableNeuralNet globalCurrent,
                                            @NonNull ITrainableNeuralNet target,
                                            @NonNull NeuralNetUpdaterConfiguration configuration) {
        Preconditions.checkArgument(configuration.getTargetUpdateFrequency() > 0, "Configuration: targetUpdateFrequency must be greater than 0, got: ", configuration.getTargetUpdateFrequency());

        this.globalCurrent = globalCurrent;
        this.target = target;
        this.targetUpdateFrequency = configuration.getTargetUpdateFrequency();
    }

    /**
     * Applies the gradients to the global current and synchronize the target network if necessary
     * @param gradients
     */
    public void handleGradients(Gradients gradients) {
        globalCurrent.applyGradients(gradients);
        ++updateCount;

        if(target != null) {
            syncTargetNetwork();
        }
    }

    private void syncTargetNetwork() {
        if(updateCount % targetUpdateFrequency == 0) {
            target.copyFrom(globalCurrent);
        }
    }
}
