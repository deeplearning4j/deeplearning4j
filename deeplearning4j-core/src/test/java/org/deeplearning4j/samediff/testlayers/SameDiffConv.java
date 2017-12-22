package org.deeplearning4j.samediff.testlayers;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffLayer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;

import java.util.*;

public class SameDiffConv extends BaseSameDiffLayer {

    private static final List<String> WEIGHT_KEYS = Collections.singletonList(ConvolutionParamInitializer.WEIGHT_KEY);
    private static final List<String> BIAS_KEYS = Collections.singletonList(ConvolutionParamInitializer.BIAS_KEY);

    private int nIn;
    private int nOut;
    private Activation activation;
    private int[] kernel;
    private int[] stride;
    private int[] padding;
    private ConvolutionMode cm;

    private Map<String,int[]> paramShapes;

    protected SameDiffConv(Builder b) {
        super(b);
        this.nIn = b.nIn;
        this.nOut = b.nOut;
        this.activation = b.activation;
        this.kernel = b.kernel;
        this.stride = b.stride;
        this.padding = b.padding;
        this.cm = b.cm;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional)inputType;
        return InputTypeUtil.getOutputTypeCnnLayers(inputType, kernel, stride, padding, new int[]{1,1},
                cm, nOut, layerIndex, getLayerName(), SameDiffConv.class);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (nIn <= 0 || override) {
            InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
            this.nIn = c.getDepth();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public List<String> weightKeys() {
        return WEIGHT_KEYS;
    }

    @Override
    public List<String> biasKeys() {
        return BIAS_KEYS;
    }

    @Override
    public Map<String, int[]> paramShapes() {
        if(paramShapes == null) {
            int[] weightsShape = new int[]{nIn, nOut, kernel[0], kernel[1]};
            int[] biasShape = new int[]{1, nOut};
            Map<String,int[]> m = new HashMap<>();
            m.put(ConvolutionParamInitializer.WEIGHT_KEY, weightsShape);
            m.put(ConvolutionParamInitializer.BIAS_KEY, biasShape);
            paramShapes = m;
        }
        return paramShapes;
    }

    @Override
    public List<String> defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable) {
//        sameDiff.conv2d()
        return null;
    }

    public static class Builder extends BaseSameDiffLayer.Builder<Builder> {

        private int nIn;
        private int nOut;
        private Activation activation = Activation.TANH;
        private int[] kernel = new int[]{2,2};
        private int[] stride = new int[]{1,1};
        private int[] padding = new int[]{0,0};
        private ConvolutionMode cm = ConvolutionMode.Same;

        public Builder nIn(int nIn){
            this.nIn = nIn;
            return this;
        }

        public Builder nOut(int nOut){
            this.nOut = nOut;
            return this;
        }

        public Builder activation(Activation activation){
            this.activation = activation;
            return this;
        }

        public Builder kernel(int... k){
            this.kernel = k;
            return this;
        }

        public Builder stride(int... s){
            this.stride = s;
            return this;
        }

        public Builder padding(int... p){
            this.padding = p;
            return this;
        }

        public Builder convolutionMode(ConvolutionMode cm){
            this.cm = cm;
            return this;
        }

        @Override
        public SameDiffConv build() {
            return new SameDiffConv(this);
        }
    }
}
