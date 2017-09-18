package org.nd4j.linalg.learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.learning.config.Momentum;

public class MomentumUpdater implements GradientUpdater<Momentum> {
    private final Momentum config;
    private INDArray v;
    private char gradientReshapeOrder;

    public MomentumUpdater(final Momentum config) {
        this.config = config;
    }

    @Override
    public Momentum getConfig() {
        return config;
    }

    @Override
    public void setStateViewArray(final INDArray viewArray, final int[] gradientShape, final char gradientOrder,
                                  final boolean initialize) {
        if (!viewArray.isRowVector()) {
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        }

        if (initialize) {
            viewArray.assign(0);
        }

        this.v = viewArray;

        // Reshape to match the expected shape of the input gradient arrays
        this.v = Shape.newShapeNoCopy(this.v, gradientShape, gradientOrder == 'f');

        if (this.v == null) {
            throw new IllegalStateException("Could not correctly reshape gradient view array");
        }

        this.gradientReshapeOrder = gradientOrder;
    }

    @Override
    public void applyUpdater(final INDArray gradient, final int iteration) {
        if (v == null) {
            throw new IllegalStateException("Updater has not been initialized with view state");
        }

        final double momentum = config.getMomentum();
        final double learningRate = config.getLearningRate();

        // Standard momentum: v = mu * v_prev - learning_rate * gradient(x)
        v.muli(momentum).addi(gradient.dup(gradientReshapeOrder).muli(learningRate));

        gradient.assign(vPrev);
    }
}
