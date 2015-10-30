package org.deeplearning4j.nn.conf.layers;

import lombok.*;

/**
 * Created by nyghtowl on 10/29/15.
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class LocalResponseNormalization extends FeedForwardLayer{
    // hyper-parameters defined using a validation set
    protected double n; // # adjacent kernal maps
    protected double k; // constant (e.g. scale)
    protected double beta; // decay rate
    protected double alpha; // decay rate

    private LocalResponseNormalization(Builder builder) {
        super(builder);
        this.k = builder.k;
        this.n = builder.n;
        this.alpha = builder.alpha;
        this.beta = builder.beta;
    }

    @Override
    public LocalResponseNormalization clone() {
        LocalResponseNormalization clone = (LocalResponseNormalization) super.clone();
        return clone;
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {
        // defaults based on AlexNet model
        protected double k=2;
        protected double n=5;
        protected double alpha=10e-4;
        protected double beta=0.75;


    public Builder(double k, double alpha, double beta) {
        this.k = k;
        this.alpha = alpha;
        this.beta = beta;
    }

    @Override
    public LocalResponseNormalization build() {
        return new LocalResponseNormalization(this);
    }

    }

}
