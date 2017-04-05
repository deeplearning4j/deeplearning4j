package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class BaseRecurrentLayer extends FeedForwardLayer {

    protected BaseRecurrentLayer(Builder builder) {
        super(builder);
    }

    protected boolean useLayerNormalization;

    /**
     * Returns true when this recurrent layer should use layer normalization.
     * @return True or False.
     */
    public boolean getUseLayerNormalization() {
        return useLayerNormalization;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for RNN layer (layer index = " + layerIndex
                            + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                            + inputType);
        }

        return InputType.recurrent(nOut);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for RNN layer (layer name = \"" + getLayerName()
                            + "\"): expect RNN input type with size > 0. Got: " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
            this.nIn = r.getSize();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, getLayerName());
    }
    //class Builder<T extends FeedForwardLayer.Builder<T>> extends Layer.Builder<T>

    @AllArgsConstructor
    public static abstract class Builder<T extends BaseRecurrentLayer.Builder<T>> extends FeedForwardLayer.Builder<T> {
        protected boolean useLayerNormalization = false;


        public Builder(Builder<T> builder) {
            super(builder);
            this.useLayerNormalization=builder.useLayerNormalization;
        }

        public Builder() {

        }

        /**
         * Use layer normalization, as described in https://arxiv.org/pdf/1607.06450.pdf. This can substantially speed
         * up training (faster convergence).
         *
         * @param useLayerNormalization
         * @return A Builder (for chain configuration)
         */
        public T setUseLayerNormalization(boolean useLayerNormalization) {
            this.useLayerNormalization = useLayerNormalization;
            return (T)this;
        }

    }

}
