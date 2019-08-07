package org.deeplearning4j.rl4j.util;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.async.listener.AsyncTrainingEpochEndEvent;
import org.deeplearning4j.rl4j.learning.async.listener.AsyncTrainingEpochEvent;
import org.deeplearning4j.rl4j.learning.async.listener.AsyncTrainingEvent;
import org.deeplearning4j.rl4j.learning.async.listener.AsyncTrainingListener;

/**
 * DataManagerSyncTrainingListener can be added to the listeners of SyncLearning so that the
 * training process can be fed to the DataManager
 */
@Slf4j
public class DataManagerAsyncTrainingListener implements AsyncTrainingListener {
    private final IDataManager dataManager;
    private final int monitorFrequency;

    private int lastMonitor;

    private DataManagerAsyncTrainingListener(Builder builder) {
        this.dataManager = builder.dataManager;

        this.monitorFrequency = builder.monitorFrequency;
        this.lastMonitor = -builder.monitorFrequency;
    }

    @Override
    public ListenerResponse onTrainingStart(AsyncTrainingEvent event) {
        try {
            dataManager.writeInfo(event.getLearning());
        } catch (Exception e) {
            log.error("Training failed.", e);
            return ListenerResponse.STOP;
        }
        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onTrainingProgress(AsyncTrainingEvent event) {
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
    public ListenerResponse onEpochStart(AsyncTrainingEpochEvent event) {
        int stepCounter = event.getAsyncThread().getStepCounter();

        if (stepCounter - lastMonitor >= monitorFrequency
                && event.getAsyncThread().getHistoryProcessor() != null
                && dataManager.isSaveData()) {
            lastMonitor = stepCounter;
            int[] shape = event.getAsyncThread().getMdp().getObservationSpace().getShape();
            event.getAsyncThread().getHistoryProcessor().startMonitor(dataManager.getVideoDir() + "/video-" + event.getAsyncThread().getEpochCounter() + "-"
                    + stepCounter + ".mp4", shape);
        }

        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onEpochEnd(AsyncTrainingEpochEndEvent event) {
        try {
            dataManager.appendStat(event.getStatEntry());
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
        private int monitorFrequency = Constants.MONITOR_FREQ;

        /**
         * Create a Builder with the given DataManager
         * @param dataManager
         */
        public Builder(IDataManager dataManager) {
            this.dataManager = dataManager;
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
        public DataManagerAsyncTrainingListener build() {
            return new DataManagerAsyncTrainingListener(this);
        }

    }
}
