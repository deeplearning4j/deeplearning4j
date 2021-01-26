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

package org.deeplearning4j.rl4j.learning.async;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.configuration.AsyncQLearningConfiguration;
import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.factory.Nd4j;

/**
 * The entry point for async training. This class will start a number ({@link AsyncQLearningConfiguration#getNumThreads()
 * configuration.getNumThread()}) of worker threads. Then, it will monitor their progress at regular intervals
 * (see setProgressEventInterval(int))
 *
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/25/16.
 * @author Alexandre Boulanger
 */
@Slf4j
public abstract class AsyncLearning<OBSERVATION extends Encodable, ACTION, ACTION_SPACE extends ActionSpace<ACTION>, NN extends NeuralNet>
                extends Learning<OBSERVATION, ACTION, ACTION_SPACE, NN>
                implements IAsyncLearning {

    private Thread monitorThread = null;

    @Getter(AccessLevel.PROTECTED)
    private final TrainingListenerList listeners = new TrainingListenerList();

    /**
     * Add a {@link TrainingListener} listener at the end of the listener list.
     *
     * @param listener the listener to be added
     */
    public void addListener(TrainingListener listener) {
        listeners.add(listener);
    }

    /**
     * Returns the configuration
     *
     * @return the configuration (see {@link AsyncQLearningConfiguration})
     */
    public abstract IAsyncLearningConfiguration getConfiguration();

    protected abstract AsyncThread newThread(int i, int deviceAffinity);

    protected abstract IAsyncGlobal<NN> getAsyncGlobal();

    protected boolean isTrainingComplete() {
        return getAsyncGlobal().isTrainingComplete();
    }

    private boolean canContinue = true;

    /**
     * Number of milliseconds between calls to onTrainingProgress
     */
    @Getter
    @Setter
    private int progressMonitorFrequency = 20000;

    private void launchThreads() {
        for (int i = 0; i < getConfiguration().getNumThreads(); i++) {
            Thread t = newThread(i, i % Nd4j.getAffinityManager().getNumberOfDevices());
            t.start();
        }
        log.info("Threads launched.");
    }

    /**
     * @return The current step
     */
    @Override
    public int getStepCount() {
        return getAsyncGlobal().getStepCount();
    }

    /**
     * This method will train the model<p>
     * The training stop when:<br>
     * - A worker thread terminate the AsyncGlobal thread (see {@link AsyncGlobal})<br>
     * OR<br>
     * - a listener explicitly stops it<br>
     * <p>
     * Listeners<br>
     * For a given event, the listeners are called sequentially in same the order as they were added. If one listener
     * returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse TrainingListener.ListenerResponse.STOP}, the remaining listeners in the list won't be called.<br>
     * Events:
     * <ul>
     *   <li>{@link TrainingListener#onTrainingStart() onTrainingStart()} is called once when the training starts.</li>
     *   <li>{@link TrainingListener#onTrainingEnd() onTrainingEnd()}  is always called at the end of the training, even if the training was cancelled by a listener.</li>
     * </ul>
     */
    public void train() {

        log.info("AsyncLearning training starting.");

        canContinue = listeners.notifyTrainingStarted();
        if (canContinue) {
            launchThreads();
            monitorTraining();
        }

        listeners.notifyTrainingFinished();
    }

    protected void monitorTraining() {
        try {
            monitorThread = Thread.currentThread();
            while (canContinue && !isTrainingComplete()) {
                canContinue = listeners.notifyTrainingProgress(this);
                if (!canContinue) {
                    return;
                }

                synchronized (this) {
                    wait(progressMonitorFrequency);
                }
            }
        } catch (InterruptedException e) {
            log.error("Training interrupted.", e);
        }
        monitorThread = null;
    }

    /**
     * Force the immediate termination of the learning. All learning threads, the AsyncGlobal thread and the monitor thread will be terminated.
     */
    public void terminate() {
        if (canContinue) {
            canContinue = false;

            Thread safeMonitorThread = monitorThread;
            if (safeMonitorThread != null) {
                safeMonitorThread.interrupt();
            }
        }
    }
}
