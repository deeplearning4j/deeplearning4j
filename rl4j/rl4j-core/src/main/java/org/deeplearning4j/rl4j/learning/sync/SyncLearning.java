/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.rl4j.learning.sync;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.learning.sync.SyncLearningEpochListener;

import java.util.ArrayList;
import java.util.List;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/3/16.
 *
 * Mother class and useful factorisations for all training methods that
 * are not asynchronous.
 *
 */
@Slf4j
public abstract class SyncLearning<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet>
                extends Learning<O, A, AS, NN> {

    private final List<SyncLearningEpochListener> syncLearningEpochListeners = new ArrayList<SyncLearningEpochListener>();

    public SyncLearning(LConfiguration conf) {
        super(conf);
    }

    public void addEpochListener(SyncLearningEpochListener listener) {
        syncLearningEpochListeners.add(listener);
    }

    public void train() {

        try {
            log.info("training starting.");

            signalTrainingStarted();

            while (getStepCounter() < getConfiguration().getMaxStep()) {
                signalBeforeEpoch();
                preEpoch();

                DataManager.StatEntry statEntry = trainEpoch();
                postEpoch();
                incrementEpoch();

                signalAfterEpoch(statEntry);

                log.info("Epoch: " + getEpochCounter() + ", reward: " + statEntry.getReward());
            }
        } catch (Exception e) {
            log.error("Training failed.", e);
            e.printStackTrace();
        }
    }

    private void signalTrainingStarted() {
        for (SyncLearningEpochListener listener : syncLearningEpochListeners) {
            listener.onTrainingStarted(this);
        }
    }

    private void signalBeforeEpoch() {
        for (SyncLearningEpochListener listener : syncLearningEpochListeners) {
            listener.onBeforeEpoch(this, getEpochCounter(), getStepCounter());
        }
    }

    private void signalAfterEpoch(DataManager.StatEntry statEntry) {
        for (SyncLearningEpochListener listener : syncLearningEpochListeners) {
            listener.onAfterEpoch(this, statEntry, getEpochCounter(), getStepCounter());
        }
    }

    protected abstract void preEpoch();

    protected abstract void postEpoch();

    protected abstract DataManager.StatEntry trainEpoch();

}
