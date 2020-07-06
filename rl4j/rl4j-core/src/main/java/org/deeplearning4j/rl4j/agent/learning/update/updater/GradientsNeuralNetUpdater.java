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
package org.deeplearning4j.rl4j.agent.learning.update.updater;

import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;

/**
 * A {@link INeuralNetUpdater} that updates a neural network and sync a target network at defined intervals
 */
public class GradientsNeuralNetUpdater implements INeuralNetUpdater<Gradients> {

    private final ITrainableNeuralNet current;
    private final ITrainableNeuralNet target;

    private int updateCount = 0;
    private final int targetUpdateFrequency;

    // TODO: Add async support
    /**
     * @param current The current {@link ITrainableNeuralNet network}
     * @param target The target {@link ITrainableNeuralNet network}
     * @param targetUpdateFrequency Will synchronize the target network at every <i>targetUpdateFrequency</i> updates
     *
     * Note: Presently async is not supported
     */
    public GradientsNeuralNetUpdater(ITrainableNeuralNet current,
                                     ITrainableNeuralNet target,
                                     int targetUpdateFrequency) {
        this.current = current;
        this.target = target;

        this.targetUpdateFrequency = targetUpdateFrequency;
    }

    /**
     * Update the current network
     * @param gradients A {@link Gradients} that will be used to update the network.
     */
    @Override
    public void update(Gradients gradients) {
        current.applyGradients(gradients);
        syncTargetNetwork();
    }

    private void syncTargetNetwork() {
        if(++updateCount % targetUpdateFrequency == 0) {
            target.copy(current);
        }
    }

}
