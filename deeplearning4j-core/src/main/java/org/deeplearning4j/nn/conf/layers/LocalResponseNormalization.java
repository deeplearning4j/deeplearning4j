package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.conf.inputs.InputType;

/**
 * Created by nyghtowl on 10/29/15.
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class LocalResponseNormalization extends Layer {
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


    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input type: Expected input of type CNN, got " + inputType);
        }
        return inputType;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @AllArgsConstructor
    public static class Builder extends Layer.Builder<Builder> {
        // defaults based on AlexNet model
        private double k = 2;
        private double n = 5;
        private double alpha = 1e-4;
        private double beta = 0.75;

        public Builder(double k, double alpha, double beta) {
            this.k = k;
            this.alpha = alpha;
            this.beta = beta;
        }

        public Builder() {
        }

        public Builder k(double k) {
            this.k = k;
            return this;
        }

        public Builder n(double n) {
            this.n = n;
            return this;
        }

        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public Builder beta(double beta) {
            this.beta = beta;
            return this;
        }

        @Override
        public LocalResponseNormalization build() {
            return new LocalResponseNormalization(this);
        }

    }

}
