package org.deeplearning4j.rl4j.util;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingEpochEndEvent;
import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingEvent;
import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingListener;

/**
 * DataManagerSyncTrainingListener can be added to the listeners of SyncLearning so that the
 * training process can be fed to the DataManager
 */
@Slf4j
public class DataManagerSyncTrainingListener implements SyncTrainingListener {
    private final IDataManager dataManager;
    private final int saveFrequency;
    private final int monitorFrequency;

    private int lastSave;
    private int lastMonitor;

    private DataManagerSyncTrainingListener(Builder builder) {
        this.dataManager = builder.dataManager;

        this.saveFrequency = builder.saveFrequency;
        this.lastSave = -builder.saveFrequency;

        this.monitorFrequency = builder.monitorFrequency;
        this.lastMonitor = -builder.monitorFrequency;
    }

    @Override
    public ListenerResponse onTrainingStart(SyncTrainingEvent event) {
        try {
            dataManager.writeInfo(event.getLearning());
        } catch (Exception e) {
            log.error("Training failed.", e);
            return ListenerResponse.STOP;
        }
        return ListenerResponse.CONTINUE;
    }

    @Override
    public void onTrainingEnd() {
        // Do nothing
    }

    @Override
    public ListenerResponse onEpochStart(SyncTrainingEvent event) {
        int stepCounter = event.getLearning().getStepCounter();

        if (stepCounter - lastMonitor >= monitorFrequency
                && event.getLearning().getHistoryProcessor() != null
                && dataManager.isSaveData()) {
            lastMonitor = stepCounter;
            int[] shape = event.getLearning().getMdp().getObservationSpace().getShape();
            event.getLearning().getHistoryProcessor().startMonitor(dataManager.getVideoDir() + "/video-" + event.getLearning().getEpochCounter() + "-"
                    + stepCounter + ".mp4", shape);
        }

        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onEpochEnd(SyncTrainingEpochEndEvent event) {
        try {
            int stepCounter = event.getLearning().getStepCounter();
            if (stepCounter - lastSave >= saveFrequency) {
                dataManager.save(event.getLearning());
                lastSave = stepCounter;
            }

            dataManager.appendStat(event.getStatEntry());
            dataManager.writeInfo(event.getLearning());
        } catch (Exception e) {
            log.error("Training failed.", e);
            return ListenerResponse.STOP;
        }

        return ListenerResponse.CONTINUE;
    }

    public static Builder builder(IDataManager dataManager) {
        return new Builder(dataManager);
    }

    public static class Builder {
        private final IDataManager dataManager;
        private int saveFrequency = Constants.MODEL_SAVE_FREQ;
        private int monitorFrequency = Constants.MONITOR_FREQ;

        /**
         * Create a Builder with the given DataManager
         * @param dataManager
         */
        public Builder(IDataManager dataManager) {
            this.dataManager = dataManager;
        }

        /**
         * A number that represent the number of steps since the last call to DataManager.save() before can it be called again.
         * @param saveFrequency (Default: 100000)
         */
        public Builder saveFrequency(int saveFrequency) {
            this.saveFrequency = saveFrequency;
            return this;
        }

        /**
         * A number that represent the number of steps since the last call to HistoryProcessor.startMonitor() before can it be called again.
         * @param monitorFrequency (Default: 10000)
         */
        public Builder monitorFrequency(int monitorFrequency) {
            this.monitorFrequency = monitorFrequency;
            return this;
        }

        /**
         * Creates a DataManagerSyncTrainingListener with the configured parameters
         * @return An instance of DataManagerSyncTrainingListener
         */
        public DataManagerSyncTrainingListener build() {
            return new DataManagerSyncTrainingListener(this);
        }

    }
}
