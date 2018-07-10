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
import lombok.Value;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * This represent a local thread that explore the environment
 * and calculate a gradient to enqueue to the global thread/model
 *
 * It has its own version of a model that it syncs at the start of every
 * sub epoch
 *
 */
@Slf4j
public abstract class AsyncThread<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet>
                extends Thread implements StepCountable {

    private int threadNumber;
    @Getter
    private int stepCounter = 0;
    @Getter
    private int epochCounter = 0;
    @Getter
    private IHistoryProcessor historyProcessor;
    @Getter
    private int lastMonitor = -Constants.MONITOR_FREQ;

    public AsyncThread(AsyncGlobal<NN> asyncGlobal, int threadNumber) {
        this.threadNumber = threadNumber;
    }

    public void setHistoryProcessor(IHistoryProcessor.Configuration conf) {
        historyProcessor = new HistoryProcessor(conf);
    }

    protected void postEpoch() {
        if (getHistoryProcessor() != null)
            getHistoryProcessor().stopMonitor();

    }

    protected void preEpoch() {
        if (getStepCounter() - lastMonitor >= Constants.MONITOR_FREQ && getHistoryProcessor() != null
                        && getDataManager().isSaveData()) {
            lastMonitor = getStepCounter();
            int[] shape = getMdp().getObservationSpace().getShape();
            getHistoryProcessor().startMonitor(getDataManager().getVideoDir() + "/video-" + threadNumber + "-"
                            + getEpochCounter() + "-" + getStepCounter() + ".mp4", shape);
        }
    }

    @Override
    public void run() {


        try {
            log.info("ThreadNum-" + threadNumber + " Started!");
            getCurrent().reset();
            Learning.InitMdp<O> initMdp = Learning.initMdp(getMdp(), historyProcessor);
            O obs = initMdp.getLastObs();
            double rewards = initMdp.getReward();
            int length = initMdp.getSteps();

            preEpoch();
            while (!getAsyncGlobal().isTrainingComplete() && getAsyncGlobal().isRunning()) {
                int maxSteps = Math.min(getConf().getNstep(), getConf().getMaxEpochStep() - length);
                SubEpochReturn<O> subEpochReturn = trainSubEpoch(obs, maxSteps);
                obs = subEpochReturn.getLastObs();
                stepCounter += subEpochReturn.getSteps();
                length += subEpochReturn.getSteps();
                rewards += subEpochReturn.getReward();
                double score = subEpochReturn.getScore();
                if (length >= getConf().getMaxEpochStep() || getMdp().isDone()) {
                    postEpoch();

                    DataManager.StatEntry statEntry = new AsyncStatEntry(getStepCounter(), epochCounter, rewards, length, score);
                    getDataManager().appendStat(statEntry);
                    log.info("ThreadNum-" + threadNumber + " Epoch: " + getEpochCounter() + ", reward: " + statEntry.getReward());

                    getCurrent().reset();
                    initMdp = Learning.initMdp(getMdp(), historyProcessor);
                    obs = initMdp.getLastObs();
                    rewards = initMdp.getReward();
                    length = initMdp.getSteps();
                    epochCounter++;

                    preEpoch();
                }
            }
        } catch (Exception e) {
            log.error("Thread crashed: " + e.getCause());
            getAsyncGlobal().setRunning(false);
            e.printStackTrace();
        } finally {
            postEpoch();
        }
    }

    protected abstract NN getCurrent();

    protected abstract int getThreadNumber();

    protected abstract AsyncGlobal<NN> getAsyncGlobal();

    protected abstract MDP<O, A, AS> getMdp();

    protected abstract AsyncConfiguration getConf();

    protected abstract DataManager getDataManager();

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
    public static class AsyncStatEntry implements DataManager.StatEntry {
        int stepCounter;
        int epochCounter;
        double reward;
        int episodeLength;
        double score;
    }

}
