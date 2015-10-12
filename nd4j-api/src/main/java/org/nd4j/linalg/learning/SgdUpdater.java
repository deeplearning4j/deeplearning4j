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
    private double lr = 1e-1;

    public SgdUpdater(double lr) {
        this.lr = lr;
    }

    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        return gradient.mul(lr);
    }
}
