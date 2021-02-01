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

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
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
public abstract class AsyncThread<OBSERVATION extends Encodable, ACTION, ACTION_SPACE extends ActionSpace<ACTION>, NN extends NeuralNet>
                extends Thread implements IEpochTrainer {

    @Getter
    private int threadNumber;

    @Getter
    protected final int deviceNum;

    /**
     * The number of steps that this async thread has produced
     */
    @Getter @Setter
    protected int stepCount = 0;

    /**
     * The number of epochs (updates) that this thread has sent to the global learner
     */
    @Getter @Setter
    protected int epochCount = 0;

    /**
     * The number of environment episodes that have been played out
     */
    @Getter @Setter
    protected int episodeCount = 0;

    /**
     * The number of steps in the current episode
     */
    @Getter
    protected int currentEpisodeStepCount = 0;

    /**
     * If the current episode needs to be reset
     */
    boolean episodeComplete = true;

    @Getter @Setter
    private IHistoryProcessor historyProcessor;

    private boolean isEpisodeStarted = false;
    private final LegacyMDPWrapper<OBSERVATION, ACTION, ACTION_SPACE> mdp;

    private final TrainingListenerList listeners;

    public AsyncThread(MDP<OBSERVATION, ACTION, ACTION_SPACE> mdp, TrainingListenerList listeners, int threadNumber, int deviceNum) {
        this.mdp = new LegacyMDPWrapper<OBSERVATION, ACTION, ACTION_SPACE>(mdp, null);
        this.listeners = listeners;
        this.threadNumber = threadNumber;
        this.deviceNum = deviceNum;
    }

    public MDP<OBSERVATION, ACTION, ACTION_SPACE> getMdp() {
        return mdp.getWrappedMDP();
    }
    protected LegacyMDPWrapper<OBSERVATION, ACTION, ACTION_SPACE> getLegacyMDPWrapper() {
        return mdp;
    }

    public void setHistoryProcessor(IHistoryProcessor.Configuration conf) {
        setHistoryProcessor(new HistoryProcessor(conf));
    }

    public void setHistoryProcessor(IHistoryProcessor historyProcessor) {
        this.historyProcessor = historyProcessor;
        mdp.setHistoryProcessor(historyProcessor);
    }

    protected void postEpisode() {
        if (getHistoryProcessor() != null)
            getHistoryProcessor().stopMonitor();

    }

    protected void preEpisode() {
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
        RunContext context = new RunContext();
        Nd4j.getAffinityManager().unsafeSetDevice(deviceNum);

        log.info("ThreadNum-" + threadNumber + " Started!");

        while (!getAsyncGlobal().isTrainingComplete()) {

            if (episodeComplete) {
                startEpisode(context);
            }

            if(!startEpoch(context)) {
                break;
            }

            episodeComplete = handleTraining(context);

            if(!finishEpoch(context)) {
                break;
            }

            if(episodeComplete) {
                finishEpisode(context);
            }
        }
    }

    private boolean finishEpoch(RunContext context) {
        epochCount++;
        IDataManager.StatEntry statEntry = new AsyncStatEntry(stepCount, epochCount, context.rewards, currentEpisodeStepCount, context.score);
        return listeners.notifyEpochTrainingResult(this, statEntry);
    }

    private boolean startEpoch(RunContext context) {
        return listeners.notifyNewEpoch(this);
    }

    private boolean handleTraining(RunContext context) {
        int maxTrainSteps = Math.min(getConfiguration().getNStep(), getConfiguration().getMaxEpochStep() - currentEpisodeStepCount);
        SubEpochReturn subEpochReturn = trainSubEpoch(context.obs, maxTrainSteps);

        context.obs = subEpochReturn.getLastObs();
        context.rewards += subEpochReturn.getReward();
        context.score = subEpochReturn.getScore();

        return subEpochReturn.isEpisodeComplete();
    }

    private void startEpisode(RunContext context) {
        getCurrent().reset();
        Learning.InitMdp<Observation>  initMdp = refacInitMdp();

        context.obs = initMdp.getLastObs();
        context.rewards = initMdp.getReward();

        preEpisode();
        episodeCount++;
    }

    private void finishEpisode(RunContext context) {
        postEpisode();

        log.info("ThreadNum-{} Episode step: {}, Episode: {}, Epoch: {}, reward: {}", threadNumber, currentEpisodeStepCount, episodeCount, epochCount, context.rewards);
    }

    protected abstract NN getCurrent();

    protected abstract IAsyncGlobal<NN> getAsyncGlobal();

    protected abstract IAsyncLearningConfiguration getConfiguration();

    protected abstract IPolicy<ACTION> getPolicy(NN net);

    protected abstract SubEpochReturn trainSubEpoch(Observation obs, int nstep);

    private Learning.InitMdp<Observation> refacInitMdp() {
        currentEpisodeStepCount = 0;

        double reward = 0;

        LegacyMDPWrapper<OBSERVATION, ACTION, ACTION_SPACE> mdp = getLegacyMDPWrapper();
        Observation observation = mdp.reset();

        ACTION action = mdp.getActionSpace().noOp(); //by convention should be the NO_OP
        while (observation.isSkipped() && !mdp.isDone()) {
            StepReply<Observation> stepReply = mdp.step(action);

            reward += stepReply.getReward();
            observation = stepReply.getObservation();

            incrementSteps();
        }

        return new Learning.InitMdp(0, observation, reward);

    }

    public void incrementSteps() {
        stepCount++;
        currentEpisodeStepCount++;
    }

    @AllArgsConstructor
    @Value
    public static class SubEpochReturn {
        int steps;
        Observation lastObs;
        double reward;
        double score;
        boolean episodeComplete;
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

    private static class RunContext {
        private Observation obs;
        private double rewards;
        private double score;
    }

}
