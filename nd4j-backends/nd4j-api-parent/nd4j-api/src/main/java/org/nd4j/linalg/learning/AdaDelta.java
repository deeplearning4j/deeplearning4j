package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;

/**
 * http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
 * https://arxiv.org/pdf/1212.5701v1.pdf
 * <p>
 * Ada delta updater. More robust adagrad that keeps track of a moving window
 * average of the gradient rather than the every decaying learning rates of adagrad
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class AdaDelta implements Serializable, GradientUpdater {
    public static final double DEFAULT_ADADELTA_EPSILON = 1e-6;
    public static final double DEFAULT_ADADELTA_RHO = 0.95;

    private INDArray msg; //E[g^2]_t by arxiv paper, algorithm 1
    private INDArray msdx; //E[delta x^2]_t by arxiv paper, algorithm 1
    private double rho = DEFAULT_ADADELTA_RHO;
    private double epsilon = DEFAULT_ADADELTA_EPSILON;


    public AdaDelta(double rho) {
        this.rho = rho;
    }

    public AdaDelta(double rho, double epsilon) {
        this.rho = rho;
        this.epsilon = epsilon;
    }

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
        this.msg = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, length / 2));
        this.msdx = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(length / 2, length));

        //Reshape to match the expected shape of the input gradient arrays
        this.msg = Shape.newShapeNoCopy(this.msg, gradientShape, gradientOrder == 'f');
        this.msdx = Shape.newShapeNoCopy(this.msdx, gradientShape, gradientOrder == 'f');
        if (msg == null || msdx == null)
            throw new IllegalStateException("Could not correctly reshape gradient view arrays");
    }

    @Override
    public void update(Object... args) {
        //no op
    }

    /**
     * Get the updated gradient for the given gradient
     * and also update the state of ada delta.
     *
     * @param gradient  the gradient to get the
     *                  updated gradient for
     * @param iteration
     * @return the update gradient
     */
    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if (msg == null || msdx == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        //Line 4 of Algorithm 1: https://arxiv.org/pdf/1212.5701v1.pdf
        //E[g^2]_t = rho * E[g^2]_{t−1} + (1-rho)*g^2_t
        msg.muli(rho).addi(gradient.mul(gradient).muli(1 - rho));

        //Calculate update:
        //dX = - g * RMS[delta x]_{t-1} / RMS[g]_t
        //Note: negative is applied in the DL4J step function: params -= update rather than params += update
        INDArray rmsdx_t1 = Transforms.sqrt(msdx.add(epsilon), false);
        INDArray rmsg_t = Transforms.sqrt(msg.add(epsilon), false);
        INDArray update = gradient.muli(rmsdx_t1.divi(rmsg_t));

        //Accumulate gradients: E[delta x^2]_t = rho * E[delta x^2]_{t-1} + (1-rho)* (delta x_t)^2
        msdx.muli(rho).addi(update.mul(update).muli(1 - rho));

        return update;
    }

    @Override
    public GradientUpdaterAggregator getAggregator(boolean addThis) {
        AdaDeltaAggregator ag = new AdaDeltaAggregator();
        if (addThis)
            ag.aggregate(this);
        return ag;
    }

    public static class AdaDeltaAggregator implements GradientUpdaterAggregator {
        private INDArray msgSum;
        private INDArray msdxSum;
        private double rhoSum;
        private int count = 0;

        @Override
        public GradientUpdater getUpdater() {
            AdaDelta adaDelta = new AdaDelta(rhoSum / count);
            adaDelta.setMsg(msgSum.div(count));
            adaDelta.setMsdx(msdxSum.div(count));
            adaDelta.setRho(rhoSum / count);
            return adaDelta;
        }

        @Override
        public void aggregate(GradientUpdater updater) {
            if (!(updater instanceof AdaDelta))
                throw new UnsupportedOperationException("Cannot aggregate AdaDelta with updater: " + updater);
            AdaDelta adaDelta = (AdaDelta) updater;
            if (msgSum == null) {
                msgSum = adaDelta.msg.dup();
                msdxSum = adaDelta.msdx.dup();
                rhoSum = adaDelta.rho;
            } else {
                msgSum.addi(adaDelta.msg);
                msdxSum.addi(adaDelta.msdx);
                rhoSum += adaDelta.rho;
            }
            count++;
        }

        @Override
        public GradientUpdaterAggregator combine(GradientUpdaterAggregator other) {
            if (!(other instanceof AdaDeltaAggregator))
                throw new IllegalArgumentException("Cannot combine AdaDeltaAggregator with aggregator: " + other);
            AdaDeltaAggregator aggregator = (AdaDeltaAggregator) other;
            msgSum.addi(aggregator.msgSum);
            msdxSum.addi(aggregator.msdxSum);
            rhoSum += aggregator.rhoSum;
            count += aggregator.count;
            return this;
        }
    }
}
