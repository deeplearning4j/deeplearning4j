package org.deeplearning4j.rl4j.learning.async;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/25/16.
 */
public abstract class AsyncLearning<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet> extends Learning<O, A, AS, NN> {


    public AsyncLearning(AsyncConfiguration conf) {
        super(conf);
    }

    public abstract AsyncConfiguration getConfiguration();

    protected abstract AsyncThread newThread(int i);

    protected abstract AsyncGlobal<NN> getAsyncGlobal();

    protected void startGlobalThread() {
        getAsyncGlobal().start();
    }

    protected boolean isTrainingComplete() {
        return getAsyncGlobal().isTrainingComplete();
    }

    public void launchThreads() {
        startGlobalThread();
        for (int i = 0; i < getConfiguration().getNumThread(); i++) {
            newThread(i).start();
        }
        getLogger().info("Threads launched.");
    }

    @Override
    public int getStepCounter() {
        return getAsyncGlobal().getT().get();
    }

    public void train() {

        getLogger().info("A3CDiscrete training starting.");
        launchThreads();

        //this is simply for stat purpose
        getDataManager().writeInfo(this);
        synchronized (this) {
            while (!isTrainingComplete() && getAsyncGlobal().isRunning()) {
                getPolicy().play(getMdp(), getHistoryProcessor());
                getDataManager().writeInfo(this);
                try {
                    wait(20000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }

    }

    @Data
    @AllArgsConstructor
    @EqualsAndHashCode(callSuper = false)
    public static class AsyncConfiguration implements LConfiguration {
        int seed;
        int maxEpochStep;
        int maxStep;
        int numThread;
        int nstep;
        double gamma;
        int updateStart;
        double rewardFactor;
        int targetDqnUpdateFreq;
        double errorClamp;
        float minEpsilon;
        int epsilonNbStep;

        public AsyncConfiguration() {

            numThread = 5;
            maxStep = 1000;
            maxEpochStep = 1000;
            targetDqnUpdateFreq = 10;
            nstep = 10;
            errorClamp = 2.0;
            minEpsilon = 0.1f;
            epsilonNbStep = 20000;

        }
    }

}
