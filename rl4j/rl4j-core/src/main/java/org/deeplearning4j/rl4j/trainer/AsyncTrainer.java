/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.rl4j.trainer;

import lombok.NonNull;
import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.IAgentLearner;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;

// TODO: Add listeners & events

/**
 * A {@link ITrainer} implementation that will create a single {@link IAgentLearner} and perform the training in a
 * synchronized setup, until a stopping condition is met.
 *
 * @param <ACTION> The type of the actions expected by the environment
 */
public class AsyncTrainer<ACTION> implements ITrainer {

    private final Builder<IAgentLearner<ACTION>> agentLearnerBuilder;
    private final Predicate<AsyncTrainer<ACTION>> stoppingCondition;

    private final int numThreads;
    private final AtomicInteger episodeCount = new AtomicInteger();
    private final AtomicInteger stepCount = new AtomicInteger();

    private boolean shouldStop = false;

    /**
     * Build a AsyncTrainer that will train until a stopping condition is met.
     * @param agentLearnerBuilder the builder that will be used to create the agent-learner instances.
     *                            Can be a descendant of {@link org.deeplearning4j.rl4j.builder.BaseAgentLearnerBuilder BaseAgentLearnerBuilder}
     *                            for common scenario, or any class or lambda that implements <tt>Builder&lt;IAgentLearner&lt;ACTION&gt;&gt;</tt> if any specific
     *                            need is not met by BaseAgentLearnerBuilder.
     * @param stoppingCondition the training will stop when this condition evaluates to true
     * @param numThreads the number of threads to run in parallel
     */
    @lombok.Builder
    public AsyncTrainer(@NonNull Builder<IAgentLearner<ACTION>> agentLearnerBuilder,
                        @NonNull Predicate<AsyncTrainer<ACTION>> stoppingCondition,
                        int numThreads) {
        Preconditions.checkArgument(numThreads > 0, "numThreads must be greater than 0, got: ", numThreads);

        this.agentLearnerBuilder = agentLearnerBuilder;
        this.stoppingCondition = stoppingCondition;
        this.numThreads = numThreads;
    }

    public void train() {
        reset();
        Thread[] threads = new Thread[numThreads];

        for(int i = 0; i < numThreads; ++i) {
            AgentLearnerThread thread = new AgentLearnerThread(agentLearnerBuilder.build(), i);
            threads[i] = thread;
            thread.start();
        }

        // Wait for all threads to finish
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                // Ignore
            }
        }
    }

    private void reset() {
        episodeCount.set(0);
        stepCount.set(0);
        shouldStop = false;
    }

    public int getEpisodeCount() {
        return episodeCount.get();
    }

    public int getStepCount() {
        return stepCount.get();
    }

    private void onEpisodeEnded(int numStepsInEpisode) {
        episodeCount.incrementAndGet();
        stepCount.addAndGet(numStepsInEpisode);
        if(stoppingCondition.test(this)) {
            shouldStop = true;
        }
    }

    private class AgentLearnerThread extends Thread {
        private final IAgentLearner<ACTION> agentLearner;
        private final int deviceNum;

        public AgentLearnerThread(IAgentLearner<ACTION> agentLearner, int deviceNum) {
            this.agentLearner = agentLearner;
            this.deviceNum = deviceNum;
        }

        @Override
        public void run() {
            Nd4j.getAffinityManager().unsafeSetDevice(deviceNum);
            while(!shouldStop) {
                agentLearner.run();
                onEpisodeEnded(agentLearner.getEpisodeStepCount());
            }
        }

    }
}