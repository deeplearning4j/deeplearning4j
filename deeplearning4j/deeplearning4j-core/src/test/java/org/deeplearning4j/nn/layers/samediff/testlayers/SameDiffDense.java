package org.deeplearning4j.nn.layers.samediff.testlayers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayerUtils;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.*;

@Data
@EqualsAndHashCode(callSuper = true, exclude = {"paramShapes"})
@JsonIgnoreProperties("paramShapes")
public class SameDiffDense extends SameDiffLayer {

    private static final List<String> W_KEYS = Collections.singletonList(DefaultParamInitializer.WEIGHT_KEY);
    private static final List<String> B_KEYS = Collections.singletonList(DefaultParamInitializer.BIAS_KEY);
    private static final List<String> PARAM_KEYS = Arrays.asList(DefaultParamInitializer.WEIGHT_KEY, DefaultParamInitializer.BIAS_KEY);

    private Map<String,long[]> paramShapes;

    private long nIn;
    private long nOut;
    private Activation activation;

    protected SameDiffDense(Builder builder) {
        super(builder);

        nIn = builder.nIn;
        nOut = builder.nOut;
        activation = builder.activation;
    }

    private SameDiffDense(){
        //No op constructor for Jackson
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
    public void defineParameters(SDLayerParams params) {
        params.clear();
        params.addWeightParam(DefaultParamInitializer.WEIGHT_KEY, new long[]{nIn, nOut});
        params.addBiasParam(DefaultParamInitializer.BIAS_KEY, new long[]{1, nOut});
    }

    @Override
    public void initializeParameters(Map<String,INDArray> params){
        for(Map.Entry<String,INDArray> e : params.entrySet()){
            if(DefaultParamInitializer.BIAS_KEY.equals(e.getKey())){
                e.getValue().assign(0.0);
            } else {
                //Normally use 'c' order, but use 'f' for direct comparison to DL4J DenseLayer
                WeightInitUtil.initWeights(nIn, nOut, new long[]{nIn, nOut}, weightInit, null, 'f', e.getValue());
            }
        }
    }

    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput, Map<String, SDVariable> paramTable) {
        SDVariable weights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);
        SDVariable bias = paramTable.get(DefaultParamInitializer.BIAS_KEY);

        SDVariable mmul = sd.mmul("mmul", layerInput, weights);
        SDVariable z = mmul.add("z", bias);
        return activation.asSameDiff("out", sd, z);
    }

    @Override
    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig) {
        if(activation == null){
            activation = SameDiffLayerUtils.fromIActivation(globalConfig.getActivationFn());
        }
    }

    public char paramReshapeOrder(String param){
        //To match DL4J
        return 'f';
    }

    public static class Builder extends SameDiffLayer.Builder<Builder> {

        private int nIn;
        private int nOut;
        private Activation activation;

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
