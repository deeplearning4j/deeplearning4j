package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class Sgd implements GradientUpdater {
    private double learningRate = 1e-1;

    public Sgd(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public int stateSizeForInputSize(int inputSize) {
        return 0;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, int[] gradientShape, char gradientOrder, boolean initialize) {
        //No op
    }

    @Override
    public void update(Object... args) {
        if (args.length > 0) {
            learningRate = (Double) args[0];
        }

    }

    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        return gradient.muli(learningRate);
    }
}
