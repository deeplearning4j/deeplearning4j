package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.conf.inputs.InputType;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class BaseRecurrentLayer extends FeedForwardLayer {

    protected BaseRecurrentLayer(Builder builder) {
        super(builder);
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN || ((InputType.InputTypeRecurrent) inputType).getSize() <= 0) {
            throw new IllegalStateException("Invalid input for RNN layer: expect RNN input type with size > 0. Got: " + inputType);
        }

        return InputType.recurrent(nOut);
    }

    @AllArgsConstructor
    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<Builder<T>> {

    }

}
