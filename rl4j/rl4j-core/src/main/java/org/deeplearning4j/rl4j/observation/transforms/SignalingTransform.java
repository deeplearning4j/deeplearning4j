package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.Observation;

import java.util.ArrayList;
import java.util.List;

public class SignalingTransform extends PassthroughTransform {

    private final List<TransformListener> listeners;

    public SignalingTransform() {
        super();
        listeners = new ArrayList<>();
    }

    public SignalingTransform(ObservationTransform source) {
        super(source);
        listeners = new ArrayList<>();
    }

    private SignalingTransform(Builder builder) {
        listeners = builder.listeners;
    }

    public void addListener(TransformListener listener) {
        listeners.add(listener);
    }

    @Override
    public void reset() {
        super.reset();
        signalOnReset();
    }

    @Override
    protected Observation handle(Observation input) {
        signalOnGetObservation(input);
        return input;
    }

    @Override
    protected boolean getIsReady() {
        return true;
    }

    private void signalOnGetObservation(Observation observation) {
        for (TransformListener listener : listeners) {
            listener.onGetObservation(observation);
        }
    }

    private void signalOnReset() {
        for (TransformListener listener : listeners) {
            listener.onReset();
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private List<TransformListener> listeners = new ArrayList<>();

        public Builder listener(TransformListener listener) {
            listeners.add(listener);

            return this;
        }

        public SignalingTransform build() {
            return new SignalingTransform(this);
        }
    }
}
