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

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.nd4j.common.base.Preconditions;

/**
 * A {@link INeuralNetUpdater} that updates a neural network and sync a target network at defined intervals
 */
public class LabelsNeuralNetUpdater implements INeuralNetUpdater<FeaturesLabels> {

    private final ITrainableNeuralNet current;
    private final ITrainableNeuralNet target;

    private int updateCount = 0;
    private final int targetUpdateFrequency;

    // TODO: Add async support
    /**
     * @param current The current {@link ITrainableNeuralNet network}
     * @param target The target {@link ITrainableNeuralNet network}
     * @param configuration The {@link Configuration} to use
     *
     * Note: Presently async is not supported
     */
    public LabelsNeuralNetUpdater(@NonNull ITrainableNeuralNet current,
                                  @NonNull ITrainableNeuralNet target,
                                  @NonNull Configuration configuration) {
        Preconditions.checkArgument(configuration.getTargetUpdateFrequency() > 0, "Configuration: targetUpdateFrequency must be greater than 0, got: ", configuration.getTargetUpdateFrequency());
        this.current = current;
        this.target = target;

        this.targetUpdateFrequency = configuration.getTargetUpdateFrequency();
    }

    /**
     * Update the current network
     * @param featuresLabels A {@link FeaturesLabels} that will be used to update the network.
     */
    @Override
    public void update(FeaturesLabels featuresLabels) {
        current.fit(featuresLabels);
        syncTargetNetwork();
    }

    private void syncTargetNetwork() {
        if(++updateCount % targetUpdateFrequency == 0) {
            target.copy(current);
        }
    }

    @SuperBuilder
    @Data
    public static class Configuration {
        /**
         * Will synchronize the target network at every <i>targetUpdateFrequency</i> updates (default: no update)
         */
        @Builder.Default
        int targetUpdateFrequency = Integer.MAX_VALUE;
    }

}
