package org.deeplearning4j.rl4j.learning.sync;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.util.IDataManager;

import java.io.IOException;

@Slf4j
public class DataManagerSyncLearningEpochListener implements SyncLearningEpochListener {
    protected final IDataManager dataManager;

    private int lastSave = -Constants.MODEL_SAVE_FREQ;

    public DataManagerSyncLearningEpochListener(IDataManager dataManager) {

        this.dataManager = dataManager;
    }

    @Override
    public void onTrainingStarted(ILearning learning) {
        try {
            dataManager.writeInfo(learning);
        } catch (IOException e) {
            log.error("Failed to write info.", e);
            e.printStackTrace();
        }
    }

    @Override
    public void onBeforeEpoch(ILearning learning, int currentEpoch, int currentStep) {
        // Do nothing
    }

    @Override
    public void onAfterEpoch(ILearning learning, DataManager.StatEntry statEntry, int currentEpoch, int currentStep) {
        try {
            if (currentStep - lastSave >= Constants.MODEL_SAVE_FREQ) {
                dataManager.save(learning);
                lastSave = currentStep;
            }

            dataManager.appendStat(statEntry);
            dataManager.writeInfo(learning);
        } catch (IOException e) {
            log.error(e.getMessage());
            e.printStackTrace();
        }

    }
}
