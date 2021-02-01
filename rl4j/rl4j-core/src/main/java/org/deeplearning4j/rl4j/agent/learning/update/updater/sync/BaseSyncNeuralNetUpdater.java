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
package org.deeplearning4j.rl4j.agent.learning.update.updater.sync;

import lombok.NonNull;
import org.deeplearning4j.rl4j.agent.learning.update.updater.INeuralNetUpdater;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.nd4j.common.base.Preconditions;

/**
 * A {@link INeuralNetUpdater} that updates a neural network and sync a target network at defined intervals
 */
public abstract class BaseSyncNeuralNetUpdater<DATA_TYPE> implements INeuralNetUpdater<DATA_TYPE> {
    protected final ITrainableNeuralNet current;
    private final ITrainableNeuralNet target;

    private final int targetUpdateFrequency;
    private int updateCount = 0;

    protected BaseSyncNeuralNetUpdater(@NonNull ITrainableNeuralNet current,
                                       @NonNull ITrainableNeuralNet target,
                                       @NonNull NeuralNetUpdaterConfiguration configuration) {
        Preconditions.checkArgument(configuration.getTargetUpdateFrequency() > 0, "Configuration: targetUpdateFrequency must be greater than 0, got: ", configuration.getTargetUpdateFrequency());

        this.current = current;
        this.target = target;
        this.targetUpdateFrequency = configuration.getTargetUpdateFrequency();
    }

    @Override
    public abstract void update(DATA_TYPE dataType);

    protected void syncTargetNetwork() {
        if(++updateCount % targetUpdateFrequency == 0) {
            target.copyFrom(current);
        }
    }

    @Override
    public void synchronizeCurrent() {
        // Do nothing; there is only one current network in the sync setup.
    }
}
