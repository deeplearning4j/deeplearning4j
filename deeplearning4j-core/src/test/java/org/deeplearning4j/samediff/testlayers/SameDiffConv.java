package org.deeplearning4j.samediff.testlayers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayerUtils;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.*;

@Data
@EqualsAndHashCode(callSuper = true, exclude = {"paramShapes"})
@JsonIgnoreProperties({"paramShapes"})
public class SameDiffConv extends BaseSameDiffLayer {

    private static final List<String> WEIGHT_KEYS = Collections.singletonList(ConvolutionParamInitializer.WEIGHT_KEY);
    private static final List<String> BIAS_KEYS = Collections.singletonList(ConvolutionParamInitializer.BIAS_KEY);
    //Order to match 'vanilla' conv layer implementation, for easy comparison
    private static final List<String> PARAM_KEYS = Arrays.asList(ConvolutionParamInitializer.BIAS_KEY, ConvolutionParamInitializer.WEIGHT_KEY);

    private int nIn;
    private int nOut;
    private Activation activation;
    private int[] kernel;
    private int[] stride;
    private int[] padding;
    private ConvolutionMode cm;
    private int[] dilation;
    private boolean hasBias;

    private Map<String, int[]> paramShapes;

    protected SameDiffConv(Builder b) {
        super(b);
        this.nIn = b.nIn;
        this.nOut = b.nOut;
        this.activation = b.activation;
        this.kernel = b.kernel;
        this.stride = b.stride;
        this.padding = b.padding;
        this.cm = b.cm;
        this.dilation = b.dilation;
        this.hasBias = b.hasBias;
    }

    private SameDiffConv(){
        //No arg constructor for Jackson/JSON serialization
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
        return InputTypeUtil.getOutputTypeCnnLayers(inputType, kernel, stride, padding, new int[]{1, 1},
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
        if(hasBias) {
            return BIAS_KEYS;
        } else {
            return Collections.emptyList();
        }
    }

    @Override
    public List<String> paramKeys() {
        if(hasBias) {
            return PARAM_KEYS;
        } else {
            return WEIGHT_KEYS;
        }
    }

    @Override
    public char paramReshapeOrder(String param) {
        //To match DL4J
        return 'c';
    }

    @Override
    public Map<String, int[]> paramShapes() {
        if (paramShapes == null) {
            int[] weightsShape = new int[]{nOut, nIn, kernel[0], kernel[1]};
            Map<String, int[]> m = new HashMap<>();
            m.put(ConvolutionParamInitializer.WEIGHT_KEY, weightsShape);
            if(hasBias) {
                int[] biasShape = new int[]{1, nOut};
                m.put(ConvolutionParamInitializer.BIAS_KEY, biasShape);
            }
            paramShapes = m;
        }
        return paramShapes;
    }

    @Override
    public List<String> defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable) {

        SDVariable w = paramTable.get(ConvolutionParamInitializer.WEIGHT_KEY);

        SDVariable[] vars;
        if(hasBias){
            SDVariable b = paramTable.get(ConvolutionParamInitializer.BIAS_KEY);
            vars = new SDVariable[]{layerInput, w, b};
        } else {
            vars = new SDVariable[]{layerInput, w};
        }

        Conv2DConfig c = Conv2DConfig.builder()
                .kh(kernel[0]).kw(kernel[1])
                .ph(padding[0]).pw(padding[1])
                .sy(stride[0]).sx(stride[1])
                .dh(dilation[0]).dw(dilation[1])
                .isSameMode(this.cm == ConvolutionMode.Same)
                .build();

        SDVariable conv = sameDiff.conv2d(vars, c);    //TODO can't set name

        SDVariable out = activation.asSameDiff("out", sameDiff, conv);

        return Collections.singletonList("out");
    }

    @Override
    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig) {
        if (activation == null) {
            activation = SameDiffLayerUtils.fromIActivation(globalConfig.getActivationFn());
        }
        if (cm == null) {
            cm = globalConfig.getConvolutionMode();
        }
    }

    public static class Builder extends BaseSameDiffLayer.Builder<Builder> {

        private int nIn;
        private int nOut;
        private Activation activation = Activation.TANH;
        private int[] kernel = new int[]{2, 2};
        private int[] stride = new int[]{1, 1};
        private int[] padding = new int[]{0, 0};
        private int[] dilation = new int[]{1, 1};
        private ConvolutionMode cm = ConvolutionMode.Same;
        private boolean hasBias = true;

        public Builder nIn(int nIn) {
            this.nIn = nIn;
            return this;
        }

        public Builder nOut(int nOut) {
            this.nOut = nOut;
            return this;
        }

        public Builder activation(Activation activation) {
            this.activation = activation;
            return this;
        }

        public Builder kernelSize(int... k) {
            this.kernel = k;
            return this;
        }

        public Builder stride(int... s) {
            this.stride = s;
            return this;
        }

        public Builder padding(int... p) {
            this.padding = p;
            return this;
        }

        public Builder convolutionMode(ConvolutionMode cm) {
            this.cm = cm;
            return this;
        }

        public Builder dilation(int... d) {
            this.dilation = d;
            return this;
        }

        public Builder hasBias(boolean hasBias){
            this.hasBias = hasBias;
            return this;
        }

        @Override
        public SameDiffConv build() {
            return new SameDiffConv(this);
        }
    }
}
