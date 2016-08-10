package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.conf.inputs.InputType;

/**
 * Created by jeffreytang on 7/21/15.
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class FeedForwardLayer extends Layer {
    protected int nIn;
    protected int nOut;

    public FeedForwardLayer(Builder builder) {
        super(builder);
        this.nIn = builder.nIn;
        this.nOut = builder.nOut;
    }


    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.FF) {
            throw new IllegalStateException("Invalid input type (layer name=\"" + getLayerName() + "\"): expected FeedForward input type. Got: " + inputType);
        }

        return InputType.feedForward(nOut);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.FF) {
            throw new IllegalStateException("Invalid input type (layer name=\"" + getLayerName() + "\"): expected FeedForward input type. Got: " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeFeedForward f = (InputType.InputTypeFeedForward) inputType;
            this.nIn = f.getSize();
        }
    }

    public abstract static class Builder<T extends Builder<T>> extends Layer.Builder<T> {
        protected int nIn = 0;
        protected int nOut = 0;

        public T nIn(int nIn) {
            this.nIn = nIn;
            return (T) this;
        }

        public T nOut(int nOut) {
            this.nOut = nOut;
            return (T) this;
        }
    }
}
