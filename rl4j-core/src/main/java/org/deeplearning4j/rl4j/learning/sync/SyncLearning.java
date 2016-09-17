package org.deeplearning4j.rl4j.learning.sync;

import org.deeplearning4j.gym.space.ActionSpace;
import org.deeplearning4j.gym.space.Encodable;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.network.NeuralNet;

import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/3/16.
 *
 * Mother class and useful factorisations for all training methods that
 * are not asynchronous.
 *
 */
public abstract class SyncLearning<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet> extends Learning<O, A, AS, NN> {

    private int lastSave = -Constants.MODEL_SAVE_FREQ;

    public SyncLearning(LConfiguration conf) {
        super(conf);
    }

    public void train() {

        getLogger().info("training starting.");

        getDataManager().writeInfo(this);


        while (getStepCounter() < getConfiguration().getMaxStep()) {

            getLogger().info("Epoch: " + getEpochCounter());

            preEpoch();
            DataManager.StatEntry statEntry = trainEpoch();
            postEpoch();

            incrementEpoch();

            if (getStepCounter() - lastSave >= Constants.MODEL_SAVE_FREQ) {
                getDataManager().save(this);
                lastSave = getStepCounter();
            }

            getDataManager().appendStat(statEntry);
            getLogger().info("reward:" + statEntry.getReward());
            getDataManager().writeInfo(this);

        }


    }


    protected abstract void preEpoch();

    protected abstract void postEpoch();

    protected abstract DataManager.StatEntry trainEpoch();

}
