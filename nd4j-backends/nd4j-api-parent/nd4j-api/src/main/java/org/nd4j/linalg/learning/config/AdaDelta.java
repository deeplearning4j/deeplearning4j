package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaDeltaUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.Arrays;

/**
 * http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
 * https://arxiv.org/pdf/1212.5701v1.pdf
 * <p>
 * Ada delta updater. More robust adagrad that keeps track of a moving window
 * average of the gradient rather than the every decaying learning rates of adagrad
 * <p>
 * Note: AdaDelta is unique in that it doesn't require manual setting of a learning rate
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
@Builder(builderClassName = "Builder")
public class AdaDelta implements IUpdater {
    public static final double DEFAULT_ADADELTA_RHO = 0.95;
    public static final double DEFAULT_ADADELTA_EPSILON = 1e-6;

    @lombok.Builder.Default private double rho = DEFAULT_ADADELTA_RHO;
    @lombok.Builder.Default private double epsilon = DEFAULT_ADADELTA_EPSILON;

    public AdaDelta() {
        this(DEFAULT_ADADELTA_RHO, DEFAULT_ADADELTA_EPSILON);
    }

    @Override
    public long stateSize(long numParams) {
        return 2 * numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        AdaDeltaUpdater u = new AdaDeltaUpdater(this);
        int[] gradientShape = viewArray.shape();
        gradientShape = Arrays.copyOf(gradientShape, gradientShape.length);
        gradientShape[1] /= 2;
        u.setStateViewArray(viewArray, gradientShape, viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public AdaDelta clone() {
        return new AdaDelta(rho, epsilon);
    }

    @Override
    public double getLearningRate(int iteration, int epoch) {
        return Double.NaN;  //No LR for  this updater
    }

    @Override
    public boolean hasLearningRate() {
        return false;
    }

    @Override
    public void setLrAndSchedule(double lr, ISchedule lrSchedule) {
        throw new UnsupportedOperationException("Cannot set learning rate or LR schedule: AdaDelta does not have a learning rate");
    }

    //Partial builder implementation to give public no-arg constructor
    public static class Builder {
        public Builder(){ }
    }
}
