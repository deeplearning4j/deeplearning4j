package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.rl4j.observation.transforms.TransformListener;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.VoidObservation;

public class HistoryTransformListener implements TransformListener {

    private final IHistoryProcessor historyProcessor;

    public HistoryTransformListener(IHistoryProcessor historyProcessor) {
        this.historyProcessor = historyProcessor;
    }

    @Override
    public void onReset() {
        // Do Nothing
    }

    @Override
    public void onGetObservation(Observation observation) {
        if(!(observation instanceof VoidObservation)) {
            historyProcessor.add(observation.toNDArray());
        }
    }
}
