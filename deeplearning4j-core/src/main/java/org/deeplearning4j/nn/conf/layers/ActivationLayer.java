package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

/**
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ActivationLayer extends FeedForwardLayer {


    private ActivationLayer(Builder builder) {
    	super(builder);
    }

    @Override
    public ActivationLayer clone() {
        ActivationLayer clone = (ActivationLayer) super.clone();
        return clone;
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {
        
        @Override
        @SuppressWarnings("unchecked")
        public ActivationLayer build() {
            return new ActivationLayer(this);
        }
    }
}
