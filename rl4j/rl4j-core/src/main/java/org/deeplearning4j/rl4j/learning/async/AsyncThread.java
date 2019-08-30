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
import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.learning.listener.*;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;
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
public abstract class AsyncThread<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet>
                extends Thread implements StepCountable {

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
     *   <li>{@link TrainingListener#onNewEpoch(IEpochTrainingEvent) onNewEpoch()} is called when a new epoch is started.</li>
     *   <li>{@link TrainingListener#onEpochTrainingResult(IEpochTrainingResultEvent) onEpochTrainingResult()} is called at the end of every
     *   epoch. It will not be called if onNewEpoch() stops the training.</li>
     * </ul>
     */
    @Override
    public void run() {
        RunContext<O> context = new RunContext<>();
        Nd4j.getAffinityManager().unsafeSetDevice(deviceNum);

        log.info("ThreadNum-" + threadNumber + " Started!");

        boolean canContinue = initWork(context);
        if (canContinue) {

            while (!getAsyncGlobal().isTrainingComplete() && getAsyncGlobal().isRunning()) {
                handleTraining(context);
                if (context.length >= getConf().getMaxEpochStep() || getMdp().isDone()) {
                    canContinue = finishEpoch(context) && startNewEpoch(context);
                    if (!canContinue) {
                        break;
                    }
                }
            }
        }
        terminateWork();
    }

    private void initNewEpoch(RunContext context) {
        getCurrent().reset();
        Learning.InitMdp<O>  initMdp = Learning.initMdp(getMdp(), historyProcessor);

        context.obs = initMdp.getLastObs();
        context.rewards = initMdp.getReward();
        context.length = initMdp.getSteps();
    }

    private void handleTraining(RunContext<O> context) {
        int maxSteps = Math.min(getConf().getNstep(), getConf().getMaxEpochStep() - context.length);
        SubEpochReturn<O> subEpochReturn = trainSubEpoch(context.obs, maxSteps);

        context.obs = subEpochReturn.getLastObs();
        stepCounter += subEpochReturn.getSteps();
        context.length += subEpochReturn.getSteps();
        context.rewards += subEpochReturn.getReward();
        context.score = subEpochReturn.getScore();
    }

    private boolean initWork(RunContext context) {
        initNewEpoch(context);
        preEpoch();
        return listeners.notifyNewEpoch(buildonNewEpochEvent());
    }

    private boolean startNewEpoch(RunContext context) {
        initNewEpoch(context);
        epochCounter++;
        preEpoch();
        return listeners.notifyNewEpoch(buildonNewEpochEvent());
    }

    private boolean finishEpoch(RunContext context) {
        postEpoch();
        IDataManager.StatEntry statEntry = new AsyncStatEntry(getStepCounter(), epochCounter, context.rewards, context.length, context.score);

        log.info("ThreadNum-" + threadNumber + " Epoch: " + getEpochCounter() + ", reward: " + context.rewards);

        return listeners.notifyEpochTrainingResult(buildEpochTrainingResultEvent(statEntry));
    }

    private void terminateWork() {
        postEpoch();
        getAsyncGlobal().terminate();
    }

    /**
     * An overridable method that builds the event passed to notifyNewEpoch
     * @return The event that will be passed to notifyNewEpoch
     */
    protected IEpochTrainingEvent buildonNewEpochEvent() {
        return new EpochTrainingEvent(getEpochCounter(), getStepCounter());
    }

    /**
     * An overridable method that builds the event passed to notifyEpochTrainingResult
     * @param statEntry An instance of IDataManager.StatEntry
     * @return The event that will be passed to notifyEpochTrainingResult
     */
    protected IEpochTrainingResultEvent buildEpochTrainingResultEvent(IDataManager.StatEntry statEntry) {
        return new EpochTrainingResultEvent(getEpochCounter(), getStepCounter(), statEntry);
    }


    protected abstract NN getCurrent();

    protected abstract int getThreadNumber();

    protected abstract IAsyncGlobal<NN> getAsyncGlobal();

    protected abstract AsyncConfiguration getConf();

    protected abstract Policy<O, A> getPolicy(NN net);

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

    private static class RunContext<O extends Encodable> {
        private O obs;
        private double rewards;
        private int length;
        private double score;
    }

}
