package org.deeplearning4j.samediff.testlayers;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;

import java.util.*;

public class SameDiffDense extends BaseSameDiffLayer {

    private static final List<String> W_KEYS = Collections.singletonList(DefaultParamInitializer.WEIGHT_KEY);
    private static final List<String> B_KEYS = Collections.singletonList(DefaultParamInitializer.BIAS_KEY);
    private static final List<String> PARAM_KEYS = Arrays.asList(DefaultParamInitializer.WEIGHT_KEY, DefaultParamInitializer.BIAS_KEY);

    private final Map<String,int[]> paramShapes;

    private int nIn;
    private int nOut;
    private Activation activation;

    protected SameDiffDense(Builder builder) {
        super(builder);

        nIn = builder.nIn;
        nOut = builder.nOut;
        activation = builder.activation;

        paramShapes = new HashMap<>();
        paramShapes.put(DefaultParamInitializer.WEIGHT_KEY, new int[]{nIn, nOut});
        paramShapes.put(DefaultParamInitializer.BIAS_KEY, new int[]{1, nOut});
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return null;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if(override){
            this.nIn = ((InputType.InputTypeFeedForward)inputType).getSize();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return null;
    }

    @Override
    public List<String> weightKeys() {
        return W_KEYS;
    }

    @Override
    public List<String> biasKeys() {
        return B_KEYS;
    }

    @Override
    public Map<String, int[]> paramShapes() {
        return paramShapes;
    }

    @Override
    public List<String> defineLayer(SameDiff sd, SDVariable layerInput, Map<String, SDVariable> paramTable) {
        SDVariable weights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);
        SDVariable bias = paramTable.get(DefaultParamInitializer.BIAS_KEY);

        SDVariable mmul = sd.mmul("mmul", layerInput, weights);
        SDVariable z = mmul.add("z", bias);
//        SDVariable out = sd.sigmoid("out", z);
        SDVariable out = activation.asSameDiff("out", sd, z);

        return Collections.singletonList("out");
    }

    public static class Builder extends BaseSameDiffLayer.Builder<Builder> {

        private int nIn;
        private int nOut;
        private Activation activation = Activation.TANH;

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

        @Override
        public SameDiffDense build() {
            return new SameDiffDense(this);
        }
    }
}
