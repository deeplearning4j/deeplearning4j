package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.RmsPropUpdater;

/**
 * RMS Prop updates:
 * <p>
 * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 * http://cs231n.github.io/neural-networks-3/#ada
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
@Builder(builderClassName = "Builder")
public class RmsProp implements IUpdater {
    public static final double DEFAULT_RMSPROP_LEARNING_RATE = 1e-1;
    public static final double DEFAULT_RMSPROP_EPSILON = 1e-8;
    public static final double DEFAULT_RMSPROP_RMSDECAY = 0.95;

    private double learningRate = 1e-1;
    private double rmsDecay = DEFAULT_RMSPROP_RMSDECAY;
    private double epsilon = DEFAULT_RMSPROP_EPSILON;

    public RmsProp() {
        this(DEFAULT_RMSPROP_LEARNING_RATE, DEFAULT_RMSPROP_RMSDECAY, DEFAULT_RMSPROP_EPSILON);
    }

    public RmsProp(double rmsDecay) {
        this(DEFAULT_RMSPROP_LEARNING_RATE, rmsDecay, DEFAULT_RMSPROP_EPSILON);
    }

    @Override
    public long stateSize(long numParams) {
        return numParams;
    }

    @Override
    public void applySchedules(int iteration, double newLearningRate) {
        this.learningRate = newLearningRate;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        RmsPropUpdater u = new RmsPropUpdater(this);
        u.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public RmsProp clone() {
        return new RmsProp(learningRate, rmsDecay, epsilon);
    }

    //Partial builder class implementation for default values & public no-arg constructor
    //https://reinhard.codes/2016/07/13/using-lomboks-builder-annotation-with-default-values/
    public static class Builder {
        private double learningRate = DEFAULT_RMSPROP_LEARNING_RATE;
        private double rmsDecay = DEFAULT_RMSPROP_RMSDECAY;
        private double epsilon = DEFAULT_RMSPROP_EPSILON;

        public Builder() {}
    }
}
