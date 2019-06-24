package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.rl4j.observation.transforms.TransformListener;
import org.deeplearning4j.rl4j.observation.Observation;

public class RecorderTransformListener implements TransformListener {

    private final IHistoryProcessor historyProcessor;

    public RecorderTransformListener(IHistoryProcessor historyProcessor) {

        this.historyProcessor = historyProcessor;
    }

    @Override
    public void onReset() {
        // Do Nothing
    }

    @Override
    public void onGetObservation(Observation observation) {
        historyProcessor.record(observation.toNDArray());
    }
}
