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

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.nd4j.linalg.primitives.Pair;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * In the original paper, the authors uses Asynchronous
 * Gradient Descent: Hogwild! It is a way to apply gradients
 * and modify a model in a lock-free manner.
 *
 * As a way to implement this with dl4j, it is unfortunately
 * necessary at the time of writing to apply the gradient
 * (update the parameters) on a single separate global thread.
 *
 * This Central thread for Asynchronous Method of reinforcement learning
 * enqueue the gradients coming from the different threads and update its
 * model and target. Those neurals nets are then synced by the other threads.
 *
 * The benefits of this thread is that the updater is "shared" between all thread
 * we have a single updater which is the single updater of the model contained here
 *
 * This is similar to RMSProp with shared g and momentum
 *
 * When Hogwild! is implemented, this could be replaced by a simple data
 * structure
 *
 *
 */
@Slf4j
public class AsyncGlobal<NN extends NeuralNet> extends Thread {

    @Getter
    final private NN current;
    final private ConcurrentLinkedQueue<Pair<Gradient[], Integer>> queue;
    final private AsyncConfiguration a3cc;
    @Getter
    private AtomicInteger T = new AtomicInteger(0);
    @Getter
    private NN target;
    @Getter
    @Setter
    private boolean running = true;

    public AsyncGlobal(NN initial, AsyncConfiguration a3cc) {
        this.current = initial;
        target = (NN) initial.clone();
        this.a3cc = a3cc;
        queue = new ConcurrentLinkedQueue<>();
    }

    public boolean isTrainingComplete() {
        return T.get() >= a3cc.getMaxStep();
    }

    public void enqueue(Gradient[] gradient, Integer nstep) {
        queue.add(new Pair<>(gradient, nstep));
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
                if (a3cc.getTargetDqnUpdateFreq() != -1
                                && T.get() / a3cc.getTargetDqnUpdateFreq() > (T.get() - pair.getSecond())
                                                / a3cc.getTargetDqnUpdateFreq()) {
                    log.info("TARGET UPDATE at T = " + T.get());
                    synchronized (this) {
                        target.copy(current);
                    }
                }
            }
        }

    }

}
