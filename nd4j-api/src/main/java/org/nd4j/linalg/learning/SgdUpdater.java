package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class SgdUpdater implements GradientUpdater {
    private double learningRate = 1e-1;
    private double momentum = 0.5;

    public SgdUpdater(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        return gradient.mul(learningRate);
    }
}
