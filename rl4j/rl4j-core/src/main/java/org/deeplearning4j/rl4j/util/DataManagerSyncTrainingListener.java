package org.deeplearning4j.rl4j.util;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingEpochEndEvent;
import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingEvent;
import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingListener;

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
    public void onTrainingStart(SyncTrainingEvent event) {
        try {
            dataManager.writeInfo(event.getLearning());
        } catch (Exception e) {
            log.error("Training failed.", e);
            e.printStackTrace();

            event.setCanContinue(false);
        }
    }

    @Override
    public void onTrainingEnd() {
        // Do nothing
    }

    @Override
    public void onEpochStart(SyncTrainingEvent event) {
        int stepCounter = event.getLearning().getStepCounter();

        if (stepCounter - lastMonitor >= monitorFrequency
                && event.getLearning().getHistoryProcessor() != null
                && dataManager.isSaveData()) {
            lastMonitor = stepCounter;
            int[] shape = event.getLearning().getMdp().getObservationSpace().getShape();
            event.getLearning().getHistoryProcessor().startMonitor(dataManager.getVideoDir() + "/video-" + event.getLearning().getEpochCounter() + "-"
                    + stepCounter + ".mp4", shape);
        }
    }

    @Override
    public void onEpochEnd(SyncTrainingEpochEndEvent event) {
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
            e.printStackTrace();

            event.setCanContinue(false);
        }
    }

    public static Builder builder(IDataManager dataManager) {
        return new Builder(dataManager);
    }

    public static class Builder {
        private final IDataManager dataManager;
        private int saveFrequency = Constants.MODEL_SAVE_FREQ;
        private int monitorFrequency = Constants.MONITOR_FREQ;

        public Builder(IDataManager dataManager) {
            this.dataManager = dataManager;
        }

        public Builder saveFrequency(int saveFrequency) {
            this.saveFrequency = saveFrequency;
            return this;
        }

        public Builder monitorFrequency(int monitorFrequency) {
            this.monitorFrequency = monitorFrequency;
            return this;
        }

        public DataManagerSyncTrainingListener build() {
            return new DataManagerSyncTrainingListener(this);
        }

    }
}
