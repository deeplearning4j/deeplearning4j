package org.deeplearning4j.rl4j.util;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.listener.*;

/**
 * DataManagerSyncTrainingListener can be added to the listeners of SyncLearning so that the
 * training process can be fed to the DataManager
 */
@Slf4j
public class DataManagerTrainingListener implements TrainingListener {
    private final IDataManager dataManager;
    public DataManagerTrainingListener(IDataManager dataManager) {
        this.dataManager = dataManager;
    }

    @Override
    public ListenerResponse onTrainingStart() {
        return ListenerResponse.CONTINUE;
    }

    @Override
    public void onTrainingEnd() {

    }

    @Override
    public ListenerResponse onNewEpoch(IEpochTrainingEvent event) {
        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onEpochTrainingResult(IEpochTrainingResultEvent event) {
        try {
            dataManager.appendStat(event.getStatEntry());
        } catch (Exception e) {
            log.error("Training failed.", e);
            return ListenerResponse.STOP;
        }

        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onTrainingProgress(ITrainingProgressEvent event) {
        try {
            dataManager.writeInfo(event.getLearning());
        } catch (Exception e) {
            log.error("Training failed.", e);
            return ListenerResponse.STOP;
        }

        return ListenerResponse.CONTINUE;
    }
}
