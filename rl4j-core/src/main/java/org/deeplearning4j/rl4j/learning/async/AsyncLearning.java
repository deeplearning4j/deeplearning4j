package org.deeplearning4j.rl4j.learning.async;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.learning.Learning;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/25/16.
 */
public abstract class AsyncLearning<O extends Encodable, A, AS extends ActionSpace<A>> extends Learning<O, A, AS> {



    public AsyncLearning(AsyncConfiguration conf) {
        super(conf);
    }

    public abstract AsyncConfiguration getConfiguration();
    protected abstract AsyncThread newThread(int i);

    protected abstract AsyncGlobal getAsyncGlobal();

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


    public void train() {

        getLogger().info("A3CDiscrete training starting.");
        launchThreads();

        //this is simply for stat purpose
        synchronized (this) {
            while (!isTrainingComplete()) {
                getPolicy().play(getMdp(), getHistoryProcessor());
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
        int updateStart;
        int numThread;
        int nstep;
        double gamma;
        double learningRate;
        int targetDqnUpdateFreq;
        double errorClamp;
        float minEpsilon;
        float epsilonDecreaseRate;

        public AsyncConfiguration() {

            updateStart = 1000;
            numThread = 5;
            maxStep = 1000;
            maxEpochStep = 1000;
            learningRate = 0.01;
            targetDqnUpdateFreq = 10;
            nstep = 10;

            errorClamp = 2.0;
            minEpsilon = 0.1f;
            epsilonDecreaseRate = 1f / 20000f;

        }
    }

}
