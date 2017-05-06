package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;

/**
 * The AdaMax updater, a variant of Adam.
 * http://arxiv.org/abs/1412.6980
 *
 * @author Justin Long
 */
@Data
@NoArgsConstructor
public class AdaMax implements Serializable, GradientUpdater {
    public static final double DEFAULT_ADAMAX_EPSILON = 1e-8;
    public static final double DEFAULT_ADAMAX_BETA1_MEAN_DECAY = 0.9;
    public static final double DEFAULT_ADAMAX_BETA2_VAR_DECAY = 0.999;


    private double learningRate = 1e-3; // learning rate
    private double beta1 = DEFAULT_ADAMAX_BETA1_MEAN_DECAY; // gradient moving avg decay rate
    private double beta2 = DEFAULT_ADAMAX_BETA2_VAR_DECAY; // gradient sqrd decay rate
    private double epsilon = DEFAULT_ADAMAX_EPSILON;
    private INDArray m, u; // moving avg & exponentially weighted infinity norm

    private char gradientReshapeOrder;

    @Override
    public int stateSizeForInputSize(int inputSize) {
        return 2 * inputSize;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, int[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(0);
        int length = viewArray.length();
        this.m = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, length / 2));
        this.u = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(length / 2, length));

        //Reshape to match the expected shape of the input gradient arrays
        this.m = Shape.newShapeNoCopy(this.m, gradientShape, gradientOrder == 'f');
        this.u = Shape.newShapeNoCopy(this.u, gradientShape, gradientOrder == 'f');
        if (m == null || u == null)
            throw new IllegalStateException("Could not correctly reshape gradient view arrays");

        this.gradientReshapeOrder = gradientOrder;
    }

    public AdaMax(double alpha, double beta1, double beta2, double epsilon) {
        this.learningRate = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon; // fudge factor to avoid zeros
    }

    public AdaMax(double alpha, double beta1, double beta2) {
        this.learningRate = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    public AdaMax(double alpha) {
        this.learningRate = alpha;
    }

    @Override
    public void update(Object... args) {
        if (args.length > 0) {
            learningRate = (Double) args[0];
        }
    }

    /**
     * Calculate the update based on the given gradient
     *
     * @param gradient  the gradient to get the update for
     * @param iteration
     * @return the gradient
     */
    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if (m == null || u == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        INDArray oneMinusBeta1Grad = gradient.mul(1.0 - beta1);
        m.muli(beta1).addi(oneMinusBeta1Grad);

        u.assign(Transforms.max(u.muli(beta2), Transforms.abs(gradient)));

        double beta1t = FastMath.pow(beta1, iteration + 1);
        double beta2t = FastMath.pow(beta2, iteration + 1);

        double alphat = learningRate * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
        if (Double.isNaN(alphat) || alphat == 0.0)
            alphat = epsilon;

        gradient.assign(m).muli(alphat).divi(u);
        return gradient;
    }
}
