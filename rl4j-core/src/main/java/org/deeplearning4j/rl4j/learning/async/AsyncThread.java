package org.deeplearning4j.rl4j.learning.async;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Value;
import org.deeplearning4j.gym.space.ActionSpace;
import org.deeplearning4j.gym.space.Encodable;
import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;

import org.deeplearning4j.rl4j.util.DataManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
public abstract class AsyncThread<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet> extends Thread implements StepCountable {

    final protected Logger log;
    @Getter
    private int stepCounter = 0;
    @Getter
    private int epochCounter = 0;
    @Getter
    private IHistoryProcessor historyProcessor;

    public AsyncThread(AsyncGlobal<NN> asyncGlobal, int threadNumber) {

        log = LoggerFactory.getLogger("ThreadNum-" + threadNumber);

    }

    public void setHistoryProcessor(IHistoryProcessor.Configuration conf) {
        historyProcessor = new HistoryProcessor(conf);
    }

    @Override
    public void run() {


        try {
            log.info("Started!");
            Learning.InitMdp<O> initMdp = Learning.initMdp(getMdp(), historyProcessor);
            O obs = initMdp.getLastObs();
            double rewards = initMdp.getReward();
            int length = initMdp.getSteps();

            while (!getAsyncGlobal().isTrainingComplete() && getAsyncGlobal().isRunning()) {
                SubEpochReturn<O> subEpochReturn = trainSubEpoch(obs, getConf().getNstep());
                obs = subEpochReturn.getLastObs();
                stepCounter += subEpochReturn.getSteps();
                length += subEpochReturn.getSteps();
                rewards += subEpochReturn.getReward();
                double score = subEpochReturn.getScore();
                if (getMdp().isDone()) {

                    if (getThreadNumber() == 1)
                        getDataManager().appendStat(new AsyncStatEntry(getStepCounter(), epochCounter, rewards, length, score));

                    initMdp = Learning.initMdp(getMdp(), historyProcessor);
                    obs = initMdp.getLastObs();
                    rewards = initMdp.getReward();
                    length = initMdp.getSteps();
                    epochCounter++;
                }
            }
        } catch (Exception e) {
            log.error("Thread crashed");
            getAsyncGlobal().setRunning(false);
            e.printStackTrace();
        }
    }

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
