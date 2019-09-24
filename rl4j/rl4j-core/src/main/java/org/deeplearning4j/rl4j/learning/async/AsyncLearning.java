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

package org.deeplearning4j.rl4j.learning.async;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.listener.*;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.factory.Nd4j;

/**
 * The entry point for async training. This class will start a number ({@link AsyncConfiguration#getNumThread()
 * configuration.getNumThread()}) of worker threads. Then, it will monitor their progress at regular intervals
 * (see setProgressEventInterval(int))
 *
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/25/16.
 * @author Alexandre Boulanger
 */
@Slf4j
public abstract class AsyncLearning<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet>
                extends Learning<O, A, AS, NN> {

    @Getter(AccessLevel.PROTECTED)
    private final TrainingListenerList listeners = new TrainingListenerList();

    public AsyncLearning(AsyncConfiguration conf) {
        super(conf);
    }

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
     * @return the configuration (see {@link AsyncConfiguration})
     */
    public abstract AsyncConfiguration getConfiguration();

    protected abstract AsyncThread newThread(int i, int deviceAffinity);

    protected abstract IAsyncGlobal<NN> getAsyncGlobal();

    protected void startGlobalThread() {
        getAsyncGlobal().start();
    }

    protected boolean isTrainingComplete() {
        return getAsyncGlobal().isTrainingComplete();
    }

    private boolean canContinue = true;

    /**
     * Number of milliseconds between calls to onTrainingProgress
     */
    @Getter @Setter
    private int progressMonitorFrequency = 20000;

    private void launchThreads() {
        startGlobalThread();
        for (int i = 0; i < getConfiguration().getNumThread(); i++) {
            Thread t = newThread(i, i % Nd4j.getAffinityManager().getNumberOfDevices());
            t.start();
        }
        log.info("Threads launched.");
    }

    /**
     * @return The current step
     */
    @Override
    public int getStepCounter() {
        return getAsyncGlobal().getT().get();
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

        cleanupPostTraining();
        listeners.notifyTrainingFinished();
    }

    protected void monitorTraining() {
        try {
            while (canContinue && !isTrainingComplete() && getAsyncGlobal().isRunning()) {
                canContinue = listeners.notifyTrainingProgress(this);
                if(!canContinue) {
                    return;
                }

                synchronized (this) {
                    wait(progressMonitorFrequency);
                }
            }
        } catch (InterruptedException e) {
            log.error("Training interrupted.", e);
        }
    }

    protected void cleanupPostTraining() {
        // Worker threads stops automatically when the global thread stops
        getAsyncGlobal().terminate();
    }
}
