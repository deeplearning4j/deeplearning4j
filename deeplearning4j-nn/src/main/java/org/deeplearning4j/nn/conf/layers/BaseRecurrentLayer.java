package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.layers.recurrent.LayerNormalization;
import org.deeplearning4j.nn.layers.recurrent.LayerNormalizationPerRecord;
import org.deeplearning4j.nn.layers.recurrent.LayerNormalizationWholeMinibatch;
import org.deeplearning4j.nn.layers.recurrent.NoNormalization;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class BaseRecurrentLayer extends FeedForwardLayer {
    protected boolean useLayerNormalization;
    /**
     * When true, compute mean and stddev across a single record. When false, compute across an entire minibatch.
     */
    protected boolean normalizePerRecord;


    protected BaseRecurrentLayer(Builder builder) {
        super(builder);
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


    @AllArgsConstructor
    public static abstract class Builder<T extends BaseRecurrentLayer.Builder<T>> extends FeedForwardLayer.Builder<T> {
        protected boolean useLayerNormalization = false;
        protected boolean normalizePerRecord = false;

        public Builder(Builder<T> builder) {
            super(builder);
            this.useLayerNormalization = builder.useLayerNormalization;
            this.normalizePerRecord = builder.normalizePerRecord;
        }

        public Builder() {

        }

        /**
         * Use layer normalization, as described in https://arxiv.org/pdf/1607.06450.pdf. This can substantially speed
         * up training (faster convergence).
         * @param useLayerNormalization Turn on layer normalization when this parameter is true.
         * @return A Builder (for chain configuration)
         */
        public T setUseLayerNormalization(boolean useLayerNormalization) {
            this.useLayerNormalization = useLayerNormalization;
            this.normalizePerRecord = false;
            return (T) this;
        }
        /**
         * Use layer normalization, but normalizing per record (as suggested in the pull request). Does not seem to work (possibly would need changes to
         * gradient calculation), so not recommended, only for demonstration purposes.
         *
         * @param useLayerNormalization Turn on layer normalization when this parameter is true.
         * @param normalizePerRecord When true, uses LayerNormalizationPerRecord, otherwise, uses LayerNormalizationWholeMinibatch
         * @return A Builder (for chain configuration)
         */
        public T setUseLayerNormalization(boolean useLayerNormalization, boolean normalizePerRecord) {
            this.useLayerNormalization = useLayerNormalization;
            this.normalizePerRecord = normalizePerRecord;
            return (T) this;
        }

    }

    public LayerNormalization getLayerNormalization() {
        LayerNormalization layerNormalization = null;
        if (useLayerNormalization) {
            if (normalizePerRecord) {
                layerNormalization = new LayerNormalizationPerRecord();
            } else {
                layerNormalization = new LayerNormalizationWholeMinibatch();
            }
        } else {
            layerNormalization = new NoNormalization();
        }
        return layerNormalization;
    }

}
