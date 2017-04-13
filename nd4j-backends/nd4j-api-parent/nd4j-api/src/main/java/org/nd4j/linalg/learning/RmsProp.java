package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * RMS Prop updates:
 * <p>
 * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 * http://cs231n.github.io/neural-networks-3/#ada
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class RmsProp implements GradientUpdater {
    public static final double DEFAULT_RMSPROP_EPSILON = 1e-8;
    public static final double DEFAULT_RMSPROP_RMSDECAY = 0.95;

    private INDArray lastGradient;
    private double rmsDecay = DEFAULT_RMSPROP_RMSDECAY;
    private double learningRate = 1e-1;
    private double epsilon = DEFAULT_RMSPROP_EPSILON;

    private char gradientReshapeOrder;

    public RmsProp(double learningRate, double rmsDecay) {
        this(learningRate, rmsDecay, DEFAULT_RMSPROP_EPSILON);
    }

    public RmsProp(double learningRate, double rmsDecay, double epsilon) {
        this.learningRate = learningRate;
        this.rmsDecay = rmsDecay;
        this.epsilon = epsilon;
    }

    @Override
    public int stateSizeForInputSize(int inputSize) {
        return inputSize;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, int[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(epsilon);
        this.lastGradient = viewArray;

        //Reshape to match the expected shape of the input gradient arrays
        this.lastGradient = Shape.newShapeNoCopy(this.lastGradient, gradientShape, gradientOrder == 'f');
        if (lastGradient == null)
            throw new IllegalStateException("Could not correctly reshape gradient view array");

        gradientReshapeOrder = gradientOrder;
    }

    @Override
    public void update(Object... args) {
        if (args.length > 0) {
            learningRate = (Double) args[0];
        }
    }

    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if (lastGradient == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        lastGradient.muli(rmsDecay).addi(gradient.mul(gradient).muli(1 - rmsDecay));
        // lr * gradient / (sqrt(cache) + 1e-8)
        return gradient.muli(learningRate).divi(Transforms.sqrt(lastGradient.dup(gradientReshapeOrder), false).addi(epsilon));
    }
}
