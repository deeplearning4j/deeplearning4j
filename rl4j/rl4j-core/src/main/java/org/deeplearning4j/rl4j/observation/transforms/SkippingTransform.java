package org.deeplearning4j.rl4j.observation.transforms;

import lombok.Builder;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.VoidObservation;

public class SkippingTransform extends PassthroughTransform {

    private int skipFrame = 1;
    private int currentIdx = 0;

    public SkippingTransform(ObservationTransform source, int skipFrame) {
        super(source);
        this.skipFrame = skipFrame;
    }


    @Builder
    public SkippingTransform(int skipFrame) {
        this.skipFrame = skipFrame;
    }

    @Override
    public void reset() {
        super.reset();
        currentIdx = 0;
    }

    @Override
    public boolean getIsReady() {
        return true;
    }

    @Override
    protected Observation handle(Observation input) {
        if(currentIdx++ % skipFrame == 0) {
            return input;
        }

        return VoidObservation.getInstance();
    }
}
