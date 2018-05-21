package org.deeplearning4j.rl4j.learning.sync;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/3/16.
 *
 * Mother class and useful factorisations for all training methods that
 * are not asynchronous.
 *
 */
@Slf4j
public abstract class SyncLearning<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet>
                extends Learning<O, A, AS, NN> {

    private int lastSave = -Constants.MODEL_SAVE_FREQ;

    public SyncLearning(LConfiguration conf) {
        super(conf);
    }

    public void train() {

        try {
            log.info("training starting.");

            getDataManager().writeInfo(this);


            while (getStepCounter() < getConfiguration().getMaxStep()) {
                preEpoch();
                DataManager.StatEntry statEntry = trainEpoch();
                postEpoch();

                incrementEpoch();

                if (getStepCounter() - lastSave >= Constants.MODEL_SAVE_FREQ) {
                    getDataManager().save(this);
                    lastSave = getStepCounter();
                }

                getDataManager().appendStat(statEntry);
                getDataManager().writeInfo(this);

                log.info("Epoch: " + getEpochCounter() + ", reward: " + statEntry.getReward());
            }
        } catch (Exception e) {
            log.error("Training failed.", e);
            e.printStackTrace();
        }


    }


    protected abstract void preEpoch();

    protected abstract void postEpoch();

    protected abstract DataManager.StatEntry trainEpoch();

}
