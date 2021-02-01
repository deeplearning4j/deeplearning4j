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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;
import org.deeplearning4j.rl4j.network.NeuralNet;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 * <p>
 * In the original paper, the authors uses Asynchronous
 * Gradient Descent: Hogwild! It is a way to apply gradients
 * and modify a model in a lock-free manner.
 * <p>
 * As a way to implement this with dl4j, it is unfortunately
 * necessary at the time of writing to apply the gradient
 * (update the parameters) on a single separate global thread.
 * <p>
 * This Central thread for Asynchronous Method of reinforcement learning
 * enqueue the gradients coming from the different threads and update its
 * model and target. Those neurals nets are then synced by the other threads.
 * <p>
 * The benefits of this thread is that the updater is "shared" between all thread
 * we have a single updater which is the single updater of the model contained here
 * <p>
 * This is similar to RMSProp with shared g and momentum
 * <p>
 * When Hogwild! is implemented, this could be replaced by a simple data
 * structure
 */
@Slf4j
public class AsyncGlobal<NN extends NeuralNet> implements IAsyncGlobal<NN> {

    final private NN current;

    private NN target;

    final private IAsyncLearningConfiguration configuration;

    @Getter
    private final Lock updateLock;

    /**
     * The number of times the gradient has been updated by worker threads
     */
    @Getter
    private int workerUpdateCount;

    @Getter
    private int stepCount;

    public AsyncGlobal(NN initial, IAsyncLearningConfiguration configuration) {
        this.current = initial;
        target = (NN) initial.clone();
        this.configuration = configuration;

        // This is used to sync between
        updateLock = new ReentrantLock();
    }

    public boolean isTrainingComplete() {
        return stepCount >= configuration.getMaxStep();
    }

    public void applyGradient(Gradient[] gradient, int nstep) {

        if (isTrainingComplete()) {
            return;
        }

        try {
            updateLock.lock();

            current.applyGradient(gradient, nstep);

            stepCount += nstep;
            workerUpdateCount++;

            int targetUpdateFrequency = configuration.getLearnerUpdateFrequency();

            // If we have a target update frequency, this means we only want to update the workers after a certain number of async updates
            // This can lead to more stable training
            if (targetUpdateFrequency != -1 && workerUpdateCount % targetUpdateFrequency == 0) {
                log.info("Updating target network at updates={} steps={}", workerUpdateCount, stepCount);
            } else {
                target.copyFrom(current);
            }
        } finally {
            updateLock.unlock();
        }

    }

    @Override
    public NN getTarget() {
        try {
            updateLock.lock();
            return target;
        } finally {
            updateLock.unlock();
        }
    }

}
