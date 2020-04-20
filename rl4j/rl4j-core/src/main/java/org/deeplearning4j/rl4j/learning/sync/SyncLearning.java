/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.learning.sync;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.listener.*;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.IDataManager;

/**
 * Mother class and useful factorisations for all training methods that
 * are not asynchronous.
 *
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/3/16.
 * @author Alexandre Boulanger
 */
@Slf4j
public abstract class SyncLearning<OBSERVATION extends Encodable, ACTION, ACTION_SPACE extends ActionSpace<ACTION>, NN extends NeuralNet>
        extends Learning<OBSERVATION, ACTION, ACTION_SPACE, NN> implements IEpochTrainer {

    private final TrainingListenerList listeners = new TrainingListenerList();

    /**
     * Add a listener at the end of the listener list.
     *
     * @param listener The listener to add
     */
    public void addListener(TrainingListener listener) {
        listeners.add(listener);
    }

    /**
     * Number of epochs between calls to onTrainingProgress. Default is 5
     */
    @Getter
    private int progressMonitorFrequency = 5;

    public void setProgressMonitorFrequency(int value) {
        if(value == 0) throw new IllegalArgumentException("The progressMonitorFrequency cannot be 0");

        progressMonitorFrequency = value;
    }

    /**
     * This method will train the model<p>
     * The training stop when:<br>
     * - the number of steps reaches the maximum defined in the configuration (see {@link ILearningConfiguration#getMaxStep() LConfiguration.getMaxStep()})<br>
     * OR<br>
     * - a listener explicitly stops it<br>
     * <p>
     * Listeners<br>
     * For a given event, the listeners are called sequentially in same the order as they were added. If one listener
     * returns {@link TrainingListener.ListenerResponse SyncTrainingListener.ListenerResponse.STOP}, the remaining listeners in the list won't be called.<br>
     * Events:
     * <ul>
     *   <li>{@link TrainingListener#onTrainingStart() onTrainingStart()} is called once when the training starts.</li>
     *   <li>{@link TrainingListener#onNewEpoch(IEpochTrainer) onNewEpoch()} and {@link TrainingListener#onEpochTrainingResult(IEpochTrainer, IDataManager.StatEntry) onEpochTrainingResult()}  are called for every epoch. onEpochTrainingResult will not be called if onNewEpoch stops the training</li>
     *   <li>{@link TrainingListener#onTrainingProgress(ILearning) onTrainingProgress()} is called after onEpochTrainingResult()</li>
     *   <li>{@link TrainingListener#onTrainingEnd() onTrainingEnd()} is always called at the end of the training, even if the training was cancelled by a listener.</li>
     * </ul>
     */
    public void train() {

        log.info("training starting.");

        boolean canContinue = listeners.notifyTrainingStarted();
        if (canContinue) {
            while (this.getStepCount() < getConfiguration().getMaxStep()) {
                preEpoch();
                canContinue = listeners.notifyNewEpoch(this);
                if (!canContinue) {
                    break;
                }

                IDataManager.StatEntry statEntry = trainEpoch();
                canContinue = listeners.notifyEpochTrainingResult(this, statEntry);
                if (!canContinue) {
                    break;
                }

                postEpoch();

                if(getEpochCount() % progressMonitorFrequency == 0) {
                    canContinue = listeners.notifyTrainingProgress(this);
                    if (!canContinue) {
                        break;
                    }
                }

                log.info("Epoch: " + getEpochCount() + ", reward: " + statEntry.getReward());
                incrementEpoch();
            }
        }

        listeners.notifyTrainingFinished();
    }

    protected abstract void preEpoch();

    protected abstract void postEpoch();

    protected abstract IDataManager.StatEntry trainEpoch(); // TODO: finish removal of IDataManager from Learning
}
