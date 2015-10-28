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
    public void update(Object... args) {
        if(args.length > 0) {
            learningRate = (Double) args[0];
        }

    }

    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        return gradient.mul(learningRate);
    }
}
