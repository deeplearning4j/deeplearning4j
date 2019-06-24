package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;

public interface TransformListener {
    void onReset();
    void onGetObservation(Observation observation);
}
