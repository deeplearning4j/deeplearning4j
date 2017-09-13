package org.nd4j.linalg.learning.config;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.NoOpUpdater;

/**
 * NoOp updater: gradient updater that makes no changes to the gradient
 *
 * @author Alex Black
 */
@Data
public class NoOp implements IUpdater {
    @Override
    public long stateSize(long numParams) {
        return 0;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        if (viewArray != null) {
            throw new IllegalStateException("Cannot use view array with NoOp updater");
        }
        return new NoOpUpdater(this);
    }

    @Override
    public NoOp clone() {
        return new NoOp();
    }
}
