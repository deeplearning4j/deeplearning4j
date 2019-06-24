package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;

public interface ObservationTransform {
    void reset();
    Observation getObservation(Observation input);
    boolean isReady();
}
