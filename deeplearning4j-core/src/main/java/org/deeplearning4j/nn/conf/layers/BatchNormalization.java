package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Batch normalization configuration
 *
 * @author Adam Gibson
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
@Builder
public class BatchNormalization extends FeedForwardLayer {
    protected double decay;
    protected double eps = Nd4j.EPS_THRESHOLD;
    protected boolean useBatchMean;
    protected double gamma;
    protected double beta;
    protected boolean lockGammaBeta;

    private BatchNormalization(Builder builder){
        super(builder);
        this.decay = builder.decay;
        this.useBatchMean = builder.useBatchMean;
        this.gamma = builder.gamma;
        this.beta = builder.beta;
        this.lockGammaBeta = builder.lockGammaBeta;
    }

    @Override
    public BatchNormalization clone() {
        BatchNormalization clone = (BatchNormalization) super.clone();
        return clone;
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {
        protected double decay = 0.9;
        protected boolean useBatchMean = true; // TODO auto set this if layer conf is batch
        protected boolean lockGammaBeta = false;
        protected double gamma = 1;
        protected double beta = 0;

        public Builder(double decay, boolean useBatchMean) {
            this.decay = decay;
            this.useBatchMean = useBatchMean;
        }

        public Builder(double gamma, double beta) {
            this.gamma = gamma;
            this.beta = beta;
        }

        public Builder(double gamma, double beta, boolean lockGammaBeta) {
            this.gamma = gamma;
            this.beta = beta;
            this.lockGammaBeta = lockGammaBeta;
        }

        public Builder(boolean lockGammaBeta) {
            this.lockGammaBeta = lockGammaBeta;
        }

        public Builder(){}

        public Builder gamma(double gamma){
            this.gamma = gamma;
            return this;
        }

        public Builder beta(double beta){
            this.beta = beta;
            return this;
        }

        public Builder decay(double decay){
            this.decay = decay;
            return this;
        }

        public Builder lockGammaBeta(boolean lockGammaBeta){
            this.lockGammaBeta = lockGammaBeta;
            return this;
        }

        @Override
        public BatchNormalization build() {
            return new BatchNormalization(this);
        }
    }

}
