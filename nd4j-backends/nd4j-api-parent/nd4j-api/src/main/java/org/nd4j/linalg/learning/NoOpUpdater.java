package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.NoOp;

/**
 * Created by Alex on 08/09/2016.
 */
@Data
public class NoOpUpdater implements GradientUpdater<NoOp> {

    private final NoOp config;

    public NoOpUpdater(NoOp config){
        this.config = config;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, int[] shape, char order, boolean initialize) {
        //No op
    }

    @Override
    public void applyUpdater(INDArray gradient, int iteration) {
        //No op
    }
}
