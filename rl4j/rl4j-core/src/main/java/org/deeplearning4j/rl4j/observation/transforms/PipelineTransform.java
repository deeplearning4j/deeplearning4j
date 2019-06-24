package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;

import java.util.ArrayList;
import java.util.List;

public class PipelineTransform implements ObservationTransform {

    private final ObservationTransform outputObservationTransform;

    private PipelineTransform(Builder builder) {
        ObservationTransform previous = builder.previous;

        for (PassthroughTransform observable : builder.observables) {
            observable.setPrevious(previous);
            previous = observable;
        }

        outputObservationTransform = previous;
    }

    public void reset() {
        if(outputObservationTransform != null) {
            outputObservationTransform.reset();
        }
    }

    public Observation getObservation(Observation input) {
        return outputObservationTransform == null ? input : outputObservationTransform.getObservation(input);
    }

    @Override
    public boolean isReady() {
        return outputObservationTransform.isReady();
    }

    public static Builder builder(ObservationTransform previous) {
        return new Builder(previous);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final ObservationTransform previous;
        private List<PassthroughTransform> observables = new ArrayList<>();

        public Builder() {
            previous = null;
        }

        public Builder(ObservationTransform previous) {
            this.previous = previous;
        }

        public Builder flowTo(PassthroughTransform observable) {
            observables.add(observable);
            return this;
        }

        public PipelineTransform build() {
            return new PipelineTransform(this);
        }

    }
}
