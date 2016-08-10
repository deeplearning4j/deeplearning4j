package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class RnnOutputLayer extends BaseOutputLayer {

    private RnnOutputLayer(Builder builder) {
        super(builder);
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input type for RnnOutputLayer (layer name=\"" + getLayerName() + "\"): Expected RNN input, got " + inputType);
        }
        return InputType.recurrent(nOut);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input type for RnnOutputLayer (layer name=\"" + getLayerName() + "\"): Expected RNN input, got " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
            this.nIn = r.getSize();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for RNN layer (layer name = \"" + getLayerName() + "\"): input type is null");
        }

        switch (inputType.getType()) {
            case FF:
                //FF -> RNN
                return new FeedForwardToRnnPreProcessor();
            case RNN:
                //RNN -> RNN: No preprocessor necessary
                return null;
            case CNN:
                //CNN -> RNN
                InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional)inputType;
                return new CnnToRnnPreProcessor(c.getHeight(),c.getWidth(),c.getDepth());
            default:
                throw new RuntimeException("Unknown input type: " + inputType);
        }
    }


    public static class Builder extends BaseOutputLayer.Builder<Builder> {

        public Builder() {
            this.lossFunction = LossFunction.MCXENT;
        }

        public Builder(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
        }

        @Override
        @SuppressWarnings("unchecked")
        public RnnOutputLayer build() {
            return new RnnOutputLayer(this);
        }
    }
}
