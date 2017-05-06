package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Sgd;

/**
 * @author Adam Gibson
 */
@Data
//@NoArgsConstructor
public class SgdUpdater implements GradientUpdater<Sgd> {

    private final Sgd config;

    public SgdUpdater(Sgd config){
        this.config = config;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, int[] gradientShape, char gradientOrder, boolean initialize) {
        //No op
    }

    @Override
    public void applyUpdater(INDArray gradient, int iteration) {
        gradient.muli(config.getLearningRate());
    }
}
