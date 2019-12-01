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

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.*;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This represent a local thread that explore the environment
 * and calculate a gradient to enqueue to the global thread/model
 *
 * It has its own version of a model that it syncs at the start of every
 * sub epoch
 *
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 * @author Alexandre Boulanger
 */
@Slf4j
public abstract class AsyncThread<O, A, AS extends ActionSpace<A>, NN extends NeuralNet>
                extends Thread implements StepCountable, IEpochTrainer {

    @Getter
    private int threadNumber;
    @Getter
    protected final int deviceNum;
    @Getter @Setter
    private int stepCounter = 0;
    @Getter @Setter
    private int epochCounter = 0;
    @Getter
    private MDP<O, A, AS> mdp;
    @Getter @Setter
    private IHistoryProcessor historyProcessor;

    private boolean isEpochStarted = false;

    private final TrainingListenerList listeners;

    public AsyncThread(IAsyncGlobal<NN> asyncGlobal, MDP<O, A, AS> mdp, TrainingListenerList listeners, int threadNumber, int deviceNum) {
        this.mdp = mdp;
        this.listeners = listeners;
        this.threadNumber = threadNumber;
        this.deviceNum = deviceNum;
    }

    public void setHistoryProcessor(IHistoryProcessor.Configuration conf) {
        historyProcessor = new HistoryProcessor(conf);
    }

    public void setHistoryProcessor(IHistoryProcessor historyProcessor) {
        this.historyProcessor = historyProcessor;
    }

    protected void postEpoch() {
        if (getHistoryProcessor() != null)
            getHistoryProcessor().stopMonitor();

    }

    protected void preEpoch() {
        // Do nothing
    }

    /**
     * This method will start the worker thread<p>
     * The thread will stop when:<br>
     * - The AsyncGlobal thread terminates or reports that the training is complete
     * (see {@link AsyncGlobal#isTrainingComplete()}). In such case, the currently running epoch will still be handled normally and
     * events will also be fired normally.<br>
     * OR<br>
     * - a listener explicitly stops it, in which case, the AsyncGlobal thread will be terminated along with
     * all other worker threads <br>
     * <p>
     * Listeners<br>
     * For a given event, the listeners are called sequentially in same the order as they were added. If one listener
     * returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse
     * TrainingListener.ListenerResponse.STOP}, the remaining listeners in the list won't be called.<br>
     * Events:
     * <ul>
     *   <li>{@link TrainingListener#onNewEpoch(IEpochTrainer) onNewEpoch()} is called when a new epoch is started.</li>
     *   <li>{@link TrainingListener#onEpochTrainingResult(IEpochTrainer, IDataManager.StatEntry) onEpochTrainingResult()} is called at the end of every
     *   epoch. It will not be called if onNewEpoch() stops the training.</li>
     * </ul>
     */
    @Override
    public void run() {
        RunContext<O> context = new RunContext<>();
        Nd4j.getAffinityManager().unsafeSetDevice(deviceNum);

        log.info("ThreadNum-" + threadNumber + " Started!");

        while (!getAsyncGlobal().isTrainingComplete() && getAsyncGlobal().isRunning()) {
            if(!isEpochStarted) {
                boolean canContinue = startNewEpoch(context);
                if (!canContinue) {
                    break;
                }
            }

            handleTraining(context);

            if (context.epochElapsedSteps >= getConf().getMaxEpochStep() || getMdp().isDone()) {
                boolean canContinue = finishEpoch(context);
                if (!canContinue) {
                    break;
                }

                ++epochCounter;
            }
        }
        terminateWork();
    }

    private void initNewEpoch(RunContext context) {
        getCurrent().reset();
        Learning.InitMdp<O>  initMdp = Learning.initMdp(getMdp(), historyProcessor);

        context.obs = initMdp.getLastObs();
        context.rewards = initMdp.getReward();
        context.epochElapsedSteps = initMdp.getSteps();
    }

    private void handleTraining(RunContext<O> context) {
        int maxSteps = Math.min(getConf().getNstep(), getConf().getMaxEpochStep() - context.epochElapsedSteps);
        SubEpochReturn<O> subEpochReturn = trainSubEpoch(context.obs, maxSteps);

        context.obs = subEpochReturn.getLastObs();
        stepCounter += subEpochReturn.getSteps();
        context.epochElapsedSteps += subEpochReturn.getSteps();
        context.rewards += subEpochReturn.getReward();
        context.score = subEpochReturn.getScore();
    }

    private boolean startNewEpoch(RunContext context) {
        isEpochStarted = true;
        initNewEpoch(context);
        preEpoch();
        return listeners.notifyNewEpoch(this);
    }

    private boolean finishEpoch(RunContext context) {
        isEpochStarted = false;
        postEpoch();
        IDataManager.StatEntry statEntry = new AsyncStatEntry(getStepCounter(), epochCounter, context.rewards, context.epochElapsedSteps, context.score);

        log.info("ThreadNum-" + threadNumber + " Epoch: " + getEpochCounter() + ", reward: " + context.rewards);

        return listeners.notifyEpochTrainingResult(this, statEntry);
    }

    private void terminateWork() {
        if(!isEpochStarted) {
            return;
        }

        postEpoch();
        getAsyncGlobal().terminate();
    }

    protected abstract NN getCurrent();

    protected abstract IAsyncGlobal<NN> getAsyncGlobal();

    protected abstract AsyncConfiguration getConf();

    protected abstract IPolicy<O, A> getPolicy(NN net);

    protected abstract SubEpochReturn<O> trainSubEpoch(O obs, int nstep);

    @AllArgsConstructor
    @Value
    public static class SubEpochReturn<O> {
        int steps;
        O lastObs;
        double reward;
        double score;
    }

    @AllArgsConstructor
    @Value
    public static class AsyncStatEntry implements IDataManager.StatEntry {
        int stepCounter;
        int epochCounter;
        double reward;
        int episodeLength;
        double score;
    }

    private static class RunContext<O> {
        private O obs;
        private double rewards;
        private int epochElapsedSteps;
        private double score;
    }

}
