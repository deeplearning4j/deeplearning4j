package org.nd4j.linalg.learning;

import lombok.Data;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;

import lombok.NoArgsConstructor;

/**
 * The Adam updater.
 * http://arxiv.org/abs/1412.6980
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class Adam implements Serializable, GradientUpdater {

    private double learningRate = 1e-3; // learning rate
    private double beta1 = 0.9; // gradient moving avg decay rate
    private double beta2 = 0.999; // gradient sqrd decay rate
    private double epsilon = 1e-8;
    private INDArray m, v; // moving avg & sqrd gradients

    public Adam(double alpha, double beta1, double beta2, double epsilon) {
        this.learningRate = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon; // fudge factor to avoid zeros
    }

    public Adam(double alpha, double beta1, double beta2) {
        this.learningRate = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    public Adam(double alpha) {
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
        if (m == null) m = Nd4j.zeros(gradient.shape());
        if (v == null) v = Nd4j.zeros(gradient.shape());

        INDArray oneMinusBeta1Grad = gradient.mul(1.0 - beta1);
        m.muli(beta1).addi(oneMinusBeta1Grad);

        INDArray oneMinusBeta2GradSquared = gradient.mul(gradient).muli(1 - beta2);
        v.muli(beta2).addi(oneMinusBeta2GradSquared);

        double beta1t = FastMath.pow(beta1, iteration);
        double beta2t = FastMath.pow(beta2, iteration);

        double alphat = learningRate * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
        if (Double.isNaN(alphat) || alphat == 0.0) alphat = Nd4j.EPS_THRESHOLD;
        INDArray sqrtV = Transforms.sqrt(v).addi(epsilon);
        INDArray ret = m.mul(alphat).divi(sqrtV);
        return ret;
    }

    @Override
    public GradientUpdaterAggregator getAggregator(boolean addThis) {
        AdamAggregator ag = new AdamAggregator();
        if (addThis) ag.aggregate(this);
        return ag;
    }

    public static class AdamAggregator implements GradientUpdaterAggregator {
        private INDArray mSum;
        private INDArray vSum;
        private double lrSum;
        private double beta1Sum;
        private double beta2Sum;
        private double epsilonSum;
        private int count = 0;

        @Override
        public GradientUpdater getUpdater() {
            Adam adam = new Adam(lrSum / count, beta1Sum / count, beta2Sum / count, epsilonSum / count);
            adam.setM(mSum.div(count));
            adam.setV(vSum.div(count));
            return adam;
        }

        @Override
        public void aggregate(GradientUpdater updater) {
            if (!(updater instanceof Adam))
                throw new UnsupportedOperationException("Cannot aggregate Adam with updater: " + updater);
            Adam adam = (Adam) updater;
            if (mSum == null) {
                mSum = adam.m.dup();
                vSum = adam.v.dup();
                lrSum = adam.learningRate;
                beta1Sum = adam.beta1;
                beta2Sum = adam.beta2;
                epsilonSum = adam.epsilon;
            } else {
                mSum.addi(adam.m);
                vSum.addi(adam.v);
                lrSum += adam.learningRate;
                beta1Sum += adam.beta1;
                beta2Sum += adam.beta2;
                epsilonSum += adam.epsilon;
            }
            count++;
        }

        @Override
        public GradientUpdaterAggregator combine(GradientUpdaterAggregator other) {
            if (!(other instanceof AdamAggregator))
                throw new IllegalArgumentException("Cannot combine AdamAggregator with aggregator: " + other);
            AdamAggregator aggregator = (AdamAggregator) other;
            mSum.addi(aggregator.mSum);
            vSum.addi(aggregator.vSum);
            lrSum += aggregator.lrSum;
            beta1Sum += aggregator.beta1Sum;
            beta2Sum += aggregator.beta2Sum;
            epsilonSum += aggregator.epsilonSum;
            count += aggregator.count;
            return this;
        }
    }

}
