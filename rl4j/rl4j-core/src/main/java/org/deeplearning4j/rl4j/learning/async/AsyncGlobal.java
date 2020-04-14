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

package org.deeplearning4j.rl4j.learning.async;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.configuration.AsyncQLearningConfiguration;
import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.nd4j.linalg.primitives.Pair;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

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
public class AsyncGlobal<NN extends NeuralNet> extends Thread implements IAsyncGlobal<NN> {

    @Getter
    final private NN current;
    final private ConcurrentLinkedQueue<Pair<Gradient[], Integer>> queue;
    final private IAsyncLearningConfiguration configuration;
    private final IAsyncLearning learning;
    @Getter
    private AtomicInteger T = new AtomicInteger(0);
    @Getter
    private NN target;
    @Getter
    private boolean running = true;

    public AsyncGlobal(NN initial, IAsyncLearningConfiguration configuration, IAsyncLearning learning) {
        this.current = initial;
        target = (NN) initial.clone();
        this.configuration = configuration;
        this.learning = learning;
        queue = new ConcurrentLinkedQueue<>();
    }

    public boolean isTrainingComplete() {
        return T.get() >= configuration.getMaxStep();
    }

    public void enqueue(Gradient[] gradient, Integer nstep) {
        if (running && !isTrainingComplete()) {
            queue.add(new Pair<>(gradient, nstep));
        }
    }

    @Override
    public void run() {

        while (!isTrainingComplete() && running) {
            if (!queue.isEmpty()) {
                Pair<Gradient[], Integer> pair = queue.poll();
                T.addAndGet(pair.getSecond());
                Gradient[] gradient = pair.getFirst();
                synchronized (this) {
                    current.applyGradient(gradient, pair.getSecond());
                }
                if (configuration.getLearnerUpdateFrequency() != -1  && T.get() / configuration.getLearnerUpdateFrequency() > (T.get() - pair.getSecond())
                        / configuration.getLearnerUpdateFrequency()) {
                    log.info("TARGET UPDATE at T = " + T.get());
                    synchronized (this) {
                        target.copy(current);
                    }
                }
            }
        }

    }

    /**
     * Force the immediate termination of the AsyncGlobal instance. Queued work items will be discarded and the AsyncLearning instance will be forced to terminate too.
     */
    public void terminate() {
        if (running) {
            running = false;
            queue.clear();
            learning.terminate();
        }
    }

}
