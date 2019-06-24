package org.deeplearning4j.rl4j.observation.transforms;

import lombok.Setter;
import org.deeplearning4j.rl4j.observation.Observation;

public abstract class PassthroughTransform implements ObservationTransform {

    @Setter
    private ObservationTransform previous;

    public PassthroughTransform() {
    }

    public PassthroughTransform(ObservationTransform previous) {
        this.previous = previous;
    }

    public void reset() {
        if(previous != null) {
            previous.reset();
        }
    }

    public Observation getObservation(Observation input) {
        return handle(previous == null ? input : previous.getObservation(input));
    }

    @Override
    public boolean isReady() {
        return (previous == null || previous.isReady()) && getIsReady();
    }

    protected abstract Observation handle(Observation input);
    protected abstract boolean getIsReady();
}
