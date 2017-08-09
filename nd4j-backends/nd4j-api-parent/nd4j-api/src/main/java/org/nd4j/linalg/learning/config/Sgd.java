package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.SgdUpdater;

/**
 * SGD updater applies a learning rate only
 * @author Adam Gibson
 */
@AllArgsConstructor
@Data
@EqualsAndHashCode
@Builder(builderClassName = "Builder")
public class Sgd implements IUpdater {
    public static final double DEFAULT_SGD_LR = 1e-3;

    private double learningRate;

    public Sgd() {
        this(DEFAULT_SGD_LR);
    }

    @Override
    public long stateSize(long numParams) {
        return 0;
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

    @Override
    public Sgd clone() {
        return new Sgd(learningRate);
    }

    public static class Builder {
        private double learningRate = DEFAULT_SGD_LR;

        public Builder() {}
    }
}
