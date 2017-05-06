package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.SgdUpdater;

/**
 * Created by Alex on 06/05/2017.
 */
@AllArgsConstructor
@Data
@EqualsAndHashCode
@Builder
public class Sgd implements Updater {

    private double learningRate;

    @Override
    public long stateSize(long numParams) {
        return numParams;
    }

    @Override
    public void applySchedules(int iteration, double newLearningRate) {
        this.learningRate = newLearningRate;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        if (viewArray != null) {
            throw new IllegalStateException("View arrays are not supported/required for SGD updater");
        }
        return new SgdUpdater(this);
    }
}
