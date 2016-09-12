package org.deeplearning4j.arbiter.layers;

import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;

import java.util.List;

/**
 * LayerSpace for batch normalization layers
 *
 * @author Alex Black
 */
public class BatchNormalizationSpace extends FeedForwardLayerSpace<BatchNormalization> {

    protected ParameterSpace<Double> decay;
    protected ParameterSpace<Double> eps;
    protected ParameterSpace<Boolean> isMinibatch;
    protected ParameterSpace<Boolean> lockGammaBeta;
    protected ParameterSpace<Double> gamma;
    protected ParameterSpace<Double> beta;

    private BatchNormalizationSpace(Builder builder) {
        super(builder);
        this.decay = builder.decay;
        this.eps = builder.eps;
        this.isMinibatch = builder.isMinibatch;
        this.lockGammaBeta = builder.lockGammaBeta;
        this.gamma = builder.gamma;
        this.beta = builder.beta;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        List<ParameterSpace> list = super.collectLeaves();
        if (decay != null) list.addAll(decay.collectLeaves());
        if (eps != null) list.addAll(eps.collectLeaves());
        if (isMinibatch != null) list.addAll(isMinibatch.collectLeaves());
        if (lockGammaBeta != null) list.addAll(lockGammaBeta.collectLeaves());
        if (gamma != null) list.addAll(gamma.collectLeaves());
        if (beta != null) list.addAll(beta.collectLeaves());
        return list;
    }


    @Override
    public BatchNormalization getValue(double[] parameterValues) {
        BatchNormalization.Builder b = new BatchNormalization.Builder();
        setLayerOptionsBuilder(b, parameterValues);
        return b.build();
    }

    protected void setLayerOptionsBuilder(BatchNormalization.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (decay != null) builder.decay(decay.getValue(values));
        if (eps != null) builder.eps(eps.getValue(values));
        if (isMinibatch != null) builder.minibatch(isMinibatch.getValue(values));
        if (lockGammaBeta != null) builder.lockGammaBeta(lockGammaBeta.getValue(values));
        if (gamma != null) builder.gamma(gamma.getValue(values));
        if (beta != null) builder.beta(beta.getValue(values));
    }


    public static class Builder extends FeedForwardLayerSpace.Builder<Builder> {

        protected ParameterSpace<Double> decay;
        protected ParameterSpace<Double> eps;
        protected ParameterSpace<Boolean> isMinibatch;
        protected ParameterSpace<Boolean> lockGammaBeta;
        protected ParameterSpace<Double> gamma;
        protected ParameterSpace<Double> beta;

        public Builder minibatch(boolean minibatch) {
            return minibatch(new FixedValue<>(minibatch));
        }

        public Builder minibatch(ParameterSpace<Boolean> minibatch) {
            this.isMinibatch = minibatch;
            return this;
        }

        public Builder gamma(double gamma) {
            return gamma(new FixedValue<>(gamma));
        }

        public Builder gamma(ParameterSpace<Double> gamma) {
            this.gamma = gamma;
            return this;
        }

        public Builder beta(double beta) {
            return beta(new FixedValue<>(beta));
        }

        public Builder beta(ParameterSpace<Double> beta) {
            this.beta = beta;
            return this;
        }

        public Builder eps(double eps) {
            return eps(new FixedValue<>(eps));
        }

        public Builder eps(ParameterSpace<Double> eps) {
            this.eps = eps;
            return this;
        }

        public Builder decay(double decay) {
            return decay(new FixedValue<Double>(decay));
        }

        public Builder decay(ParameterSpace<Double> decay) {
            this.decay = decay;
            return this;
        }

        public Builder lockGammaBeta(boolean lockGammaBeta) {
            return lockGammaBeta(new FixedValue<>(lockGammaBeta));
        }

        public Builder lockGammaBeta(ParameterSpace<Boolean> lockGammaBeta) {
            this.lockGammaBeta = lockGammaBeta;
            return this;
        }

        @Override
        public BatchNormalizationSpace build() {
            return new BatchNormalizationSpace(this);
        }
    }
}
