package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.conf.inputs.InputType;

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

    @Override
    public InputType getOutputType(InputType inputType) {
        if(inputType == null) throw new IllegalStateException("Invalid input type: null");
        return inputType;
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
