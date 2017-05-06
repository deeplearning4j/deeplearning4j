package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.learning.config.RmsProp;
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
public class RmsPropUpdater implements GradientUpdater<RmsProp> {

    private final RmsProp config;

    private INDArray lastGradient;
    private char gradientReshapeOrder;

    public RmsPropUpdater(RmsProp config) {
        this.config = config;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, int[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(config.getEpsilon());
        this.lastGradient = viewArray;

        //Reshape to match the expected shape of the input gradient arrays
        this.lastGradient = Shape.newShapeNoCopy(this.lastGradient, gradientShape, gradientOrder == 'f');
        if (lastGradient == null)
            throw new IllegalStateException("Could not correctly reshape gradient view array");

        gradientReshapeOrder = gradientOrder;
    }

    @Override
    public void applyUpdater(INDArray gradient, int iteration) {
        if (lastGradient == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        double learningRate = config.getLearningRate();
        double rmsDecay = config.getRmsDecay();
        double epsilon = config.getEpsilon();

        lastGradient.muli(rmsDecay).addi(gradient.mul(gradient).muli(1 - rmsDecay));
        // lr * gradient / (sqrt(cache) + 1e-8)
        gradient.muli(learningRate).divi(Transforms.sqrt(lastGradient.dup(gradientReshapeOrder), false).addi(epsilon));
    }
}
