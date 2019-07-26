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
import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingEpochEndEvent;
import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingEvent;
import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingListener;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.IDataManager;

import java.util.ArrayList;
import java.util.List;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/3/16.
 * @author Alexandre Boulanger
 *
 * Mother class and useful factorisations for all training methods that
 * are not asynchronous.
 *
 */
@Slf4j
public abstract class SyncLearning<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet>
                extends Learning<O, A, AS, NN> {

    public SyncLearning(LConfiguration conf) {
        super(conf);
    }

    private List<SyncTrainingListener> listeners = new ArrayList<>();
    public void addListener(SyncTrainingListener listener) {
        listeners.add(listener);
    }

    public void train() {

        log.info("training starting.");

        boolean canContinue = notifyTrainingStarted();
        if(canContinue) {
            while (getStepCounter() < getConfiguration().getMaxStep()) {
                preEpoch();
                canContinue = notifyEpochStarted();
                if(!canContinue) {
                    break;
                }

                IDataManager.StatEntry statEntry = trainEpoch();

                postEpoch();
                canContinue = notifyEpochFinished(statEntry);
                if(!canContinue) {
                    break;
                }

                log.info("Epoch: " + getEpochCounter() + ", reward: " + statEntry.getReward());

                incrementEpoch();
            }
        }

        notifyTrainingFinished();
    }

    private boolean notifyTrainingStarted() {
        SyncTrainingEvent event = new SyncTrainingEvent(this);
        for(SyncTrainingListener listener : listeners) {
            listener.onTrainingStart(event);

            if(!event.getCanContinue()) {
                return false;
            }
        }

        return true;
    }

    private void notifyTrainingFinished() {
        for(SyncTrainingListener listener : listeners) {
            listener.onTrainingEnd();
        }
    }

    private boolean notifyEpochStarted() {
        SyncTrainingEvent event = new SyncTrainingEvent(this);
        for(SyncTrainingListener listener : listeners) {
            listener.onEpochStart(event);

            if(!event.getCanContinue()) {
                return false;
            }
        }

        return true;
    }

    private boolean notifyEpochFinished(IDataManager.StatEntry statEntry) {
        SyncTrainingEpochEndEvent event = new SyncTrainingEpochEndEvent(this, statEntry);
        for(SyncTrainingListener listener : listeners) {
            listener.onEpochEnd(event);

            if(!event.getCanContinue()) {
                return false;
            }
        }

        return true;
    }

    protected abstract void preEpoch();

    protected abstract void postEpoch();

    protected abstract IDataManager.StatEntry trainEpoch(); // TODO: finish removal of IDataManager from Learning
}
