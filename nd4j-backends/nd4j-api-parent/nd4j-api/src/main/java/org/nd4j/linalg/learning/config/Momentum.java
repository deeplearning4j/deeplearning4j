package org.nd4j.linalg.learning.config;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.MomentumUpdater;

public class Momentum implements IUpdater {
    public static final double DEFAULT_MOMENTUM = 0.9;
    public static final double DEFAULT_LEARNING_RATE = 0.1;

    private final double momentum;
    private double learningRate;

    public Momentum() {
        this(DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM);
    }

    public Momentum(double momentum) {
        this(DEFAULT_LEARNING_RATE, momentum);
    }

    public Momentum(final double learningRate, final double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
    }

    @Override
    public long stateSize(final long numParams) {
        return numParams;
    }

    @Override
    public void applySchedules(final int iteration, final double newLearningRate) {
        this.learningRate = newLearningRate;
        // TODO: LR schedule?
    }

    @Override
    public GradientUpdater instantiate(final INDArray viewArray, final boolean initializeViewArray) {
        final MomentumUpdater momentumUpdater = new MomentumUpdater(this);
        momentumUpdater.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);

        return momentumUpdater;
    }

    @Override
    @SuppressWarnings("MethodDoesntCallSuperMethod")
    public IUpdater clone() {
        return new Momentum(learningRate, momentum);
    }

    public double getMomentum() {
        return momentum;
    }

    public double getLearningRate() {
        return learningRate;
    }
}
