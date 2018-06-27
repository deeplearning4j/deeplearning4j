package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.params.LocallyConnected2DParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collection;
import java.util.Map;

/**
 * A 2D locally connected layer computes a convolution with unshared weights. In a regular
 * convolution operation for each input filter there is one kernel that moves over patches
 * of the filter. In a locally connected layer, there is a separate kernel for each patch.
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class LocallyConnected2D extends ConvolutionLayer {

    int[] inputSize;
    int[] outputSize;

    private LocallyConnected2D(Builder builder) {
        super(builder);
        initializeConstraints(builder);
        this.inputSize = builder.inputSize;
        computeOutputSize();
    }

    private void computeOutputSize() {
        int nIn = (int) getNIn();

        int[] inputShape = new int[] {1, nIn, inputSize[0], inputSize[1]};
        INDArray dummyInputForShapeInference = Nd4j.ones(inputShape);

        if (convolutionMode == ConvolutionMode.Same) {
            this.outputSize = ConvolutionUtils.getOutputSize(
                    dummyInputForShapeInference, kernelSize, stride, null, convolutionMode, dilation);
            this.padding = ConvolutionUtils.getSameModeTopLeftPadding(outputSize, inputSize, kernelSize, stride, dilation);
        } else {
            this.outputSize = ConvolutionUtils.getOutputSize(dummyInputForShapeInference, kernelSize, stride, padding, convolutionMode, dilation); //Also performs validation
        }
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet(
                "LocallyConnected2D", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.local.LocallyConnected2DLayer ret =
                new org.deeplearning4j.nn.layers.local.LocallyConnected2DLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return LocallyConnected2DParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input for 2D locally connected layer (layer name=\""
                    + getLayerName() + "\"): Expected CNN input, got " + inputType);
        }
        return InputTypeUtil.getOutputTypeCnnLayers(inputType, kernelSize, stride, padding, dilation,
                convolutionMode, nOut, layerIndex, getLayerName(), ConvolutionLayer.class);
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return null;
    }

    public static class Builder extends ConvolutionLayer.BaseConvBuilder<LocallyConnected2D.Builder> {

        int[] inputSize;

        protected Builder(int[] kernelSize, int[] stride, int[] padding, int[] inputSize) {
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
            this.inputSize = inputSize;
        }

        /**
         * Set input filter size (h,w) for this locally connected 2D layer
         *
         * @param inputSize pair of height and width of the input filters to this layer
         * @return Builder
         */
        public Builder setInputSize(int[] inputSize){
            Preconditions.checkState(inputSize.length == 2, "Input size argument of a locally connected" +
                    "layer has to have length 2, got " + inputSize.length);
            this.inputSize = inputSize;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public LocallyConnected2D build() {
            ConvolutionUtils.validateConvolutionModePadding(convolutionMode, padding);
            ConvolutionUtils.validateCnnKernelStridePadding(kernelSize, stride, padding);
            return new LocallyConnected2D(this);
        }
    }

}
