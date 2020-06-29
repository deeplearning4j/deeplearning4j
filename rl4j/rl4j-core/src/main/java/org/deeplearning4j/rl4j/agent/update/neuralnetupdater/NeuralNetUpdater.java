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
package org.deeplearning4j.rl4j.agent.update.neuralnetupdater;

import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.nd4j.linalg.dataset.api.DataSet;

/**
 * A {@link INeuralNetUpdater} that updates a neural network and sync a target network at defined intervals
 */
public class NeuralNetUpdater implements INeuralNetUpdater {

    private final ITrainableNeuralNet current;
    private final ITrainableNeuralNet target;

    private int updateCount = 0;
    private final int targetUpdateFrequency;

    /**
     * @param current The current {@link ITrainableNeuralNet network}
     * @param target The target {@link ITrainableNeuralNet network}
     * @param targetUpdateFrequency Will synchronize the target network at every <i>targetUpdateFrequency</i> updates
     */
    public NeuralNetUpdater(ITrainableNeuralNet current,
                            ITrainableNeuralNet target,
                            int targetUpdateFrequency) {
        this.current = current;
        this.target = target;

        this.targetUpdateFrequency = targetUpdateFrequency;
    }

    /**
     * Update the current network
     * @param featuresLabels A Dataset that will be used to update the network.
     */
    @Override
    public void update(DataSet featuresLabels) {
        current.fit(featuresLabels);
        syncTargetNetwork();
    }

    private void syncTargetNetwork() {
        if(++updateCount % targetUpdateFrequency == 0) {
            target.copy(current);
        }
    }

}
