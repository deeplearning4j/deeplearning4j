package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class RnnOutputLayer extends BaseOutputLayer {

    private RnnOutputLayer(Builder builder) {
        super(builder);
    }

    @Override
    public Layer instantiate(Collection<IterationListener> iterationListeners,
                             String name, int layerIndex, int numInputs, INDArray layerParamsView,
                             boolean initializeParams) {
        LayerValidation.assertNInNOutSet("RnnOutputLayer", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer ret =
                        new org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer(this);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(this, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(this);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return DefaultParamInitializer.getInstance();
    }

    @Override
    public InputType[] getOutputType(int layerIndex, InputType... inputType) {
        if (preProcessor != null) {
            inputType = preProcessor.getOutputType(inputType);
        }
        if (inputType == null || inputType[0].getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input type for RnnOutputLayer (layer index = " + layerIndex
                            + ", layer name=\"" + getLayerName() + "\"): Expected RNN input, got " + (inputType == null ? null : inputType[0]));
        }
        InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent) inputType[0];

        return new InputType[]{InputType.recurrent(nOut, itr.getTimeSeriesLength())};
    }

    @Override
    public void setNIn(InputType[] inputType, boolean override) {
        if(preProcessor != null){
            inputType = preProcessor.getOutputType(inputType);
        }
        if (inputType == null || inputType.length != 1 || inputType[0].getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input type for RnnOutputLayer (layer name=\"" + getLayerName()
                            + "\"): Expected RNN input, got "
                    + (inputType == null ? null : Arrays.toString(inputType)));
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType[0];
            this.nIn = r.getSize();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, getLayerName());
    }


    public static class Builder extends BaseOutputLayer.Builder<Builder> {

        public Builder() {

        }

        public Builder(LossFunction lossFunction) {
            lossFunction(lossFunction);
        }

        public Builder(ILossFunction lossFunction) {
            this.lossFn = lossFunction;
        }

        @Override
        @SuppressWarnings("unchecked")
        public RnnOutputLayer build() {
            return new RnnOutputLayer(this);
        }
    }
}
