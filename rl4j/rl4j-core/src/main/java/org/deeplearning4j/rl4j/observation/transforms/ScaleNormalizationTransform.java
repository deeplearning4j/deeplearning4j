package org.deeplearning4j.rl4j.observation.transforms;

import lombok.Builder;
import org.deeplearning4j.rl4j.observation.BasicObservation;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

@Builder
public class ScaleNormalizationTransform extends PassthroughTransform {

    double scale = 1.0;

    @Override
    protected Observation handle(Observation input) {
        INDArray ndArray = input.toNDArray();
        ndArray.muli(1.0 / scale);

        return new BasicObservation(ndArray);
    }

    @Override
    protected boolean getIsReady() {
        return true;
    }
}
