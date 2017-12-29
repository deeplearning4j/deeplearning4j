package org.deeplearning4j.samediff.testlayers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffOutputLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayerUtils;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.*;

@Data
@EqualsAndHashCode(callSuper = true, exclude = {"paramShapes"})
@JsonIgnoreProperties("paramShapes")
public class SameDiffOutput extends BaseSameDiffOutputLayer {

    private static final List<String> W_KEYS = Collections.singletonList(DefaultParamInitializer.WEIGHT_KEY);
    private static final List<String> B_KEYS = Collections.singletonList(DefaultParamInitializer.BIAS_KEY);
    private static final List<String> PARAM_KEYS = Arrays.asList(DefaultParamInitializer.WEIGHT_KEY, DefaultParamInitializer.BIAS_KEY);

    private Map<String,int[]> paramShapes;

    private int nIn;
    private int nOut;
    private Activation activation;
    private LossFunctions.LossFunction lossFn;

    protected SameDiffOutput(Builder builder) {
        super(builder);

        nIn = builder.nIn;
        nOut = builder.nOut;
        activation = builder.activation;
        lossFn = builder.lossFn;
    }

    private SameDiffOutput(){
        //No op constructor for Jackson
    }

    @Override
    public String outputActivationsKey() {
        return "out";
    }

    @Override
    public String lossKey() {
        return "loss";
    }

    @Override
    public int[] labelShape() {
        return new int[]{-1, nOut};
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
        if(paramShapes == null){
            paramShapes = new HashMap<>();
            paramShapes.put(DefaultParamInitializer.WEIGHT_KEY, new int[]{nIn, nOut});
            paramShapes.put(DefaultParamInitializer.BIAS_KEY, new int[]{1, nOut});
        }
        return paramShapes;
    }

    @Override
    public List<String> defineLayer(SameDiff sd, SDVariable layerInput, SDVariable layerLabel, Map<String, SDVariable> paramTable) {
        SDVariable weights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);
        SDVariable bias = paramTable.get(DefaultParamInitializer.BIAS_KEY);

        SDVariable mmul = sd.mmul("mmul", layerInput, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = activation.asSameDiff("out", sd, z);

//        //TODO for now: Calculate MSE only
//        SDVariable diff = out.sub(layerLabel);
        int[] labelShape = labelShape();
//        SDVariable sqDiff = diff.mul(diff);
//        SDVariable mse = sd.loss

        String lossKey = lossKey();
        SDVariable loss;
        int d = 1;
        switch (lossFn){
            case MSE:
                loss = sd.lossMSE( lossKey, out, layerLabel, d);
                break;
            case L1:
                loss = sd.lossL1( lossKey, out, layerLabel, d);
                break;
            case XENT:
                loss = sd.lossBinaryXENT( lossKey, out, layerLabel, d);
                break;
            case MCXENT:
                loss = sd.lossMCXENT( lossKey, out, layerLabel, d);
                break;
            case SQUARED_LOSS:
                loss = sd.lossMSE( lossKey + "-pre", out, layerLabel, d).mul( lossKey, labelShape[1]);
                break;
            case NEGATIVELOGLIKELIHOOD:
                loss = sd.lossNegativeLogLikelihood( lossKey, out, layerLabel, d);
                break;
            case HINGE:
                loss = sd.lossHinge( lossKey, out, layerLabel, d);
                break;
            case SQUARED_HINGE:
                loss = sd.lossSquaredHinge( lossKey, out, layerLabel, d);
                break;
            case KL_DIVERGENCE:
                loss = sd.lossKLD( lossKey, out, layerLabel, d);
                break;
            case MEAN_ABSOLUTE_ERROR:
                loss = sd.lossMAE( lossKey, out, layerLabel, d);
                break;
            case L2:
                loss = sd.lossL2( lossKey, out, layerLabel, d);
                break;
            case MEAN_SQUARED_LOGARITHMIC_ERROR:
                loss = sd.lossMSLE( lossKey, out, layerLabel, d);
                break;
            case POISSON:
                loss = sd.lossPoisson( lossKey, out, layerLabel, d);
                break;
            case EXPLL:
            case RMSE_XENT:
            case RECONSTRUCTION_CROSSENTROPY:
            case CUSTOM:
            case COSINE_PROXIMITY:
            case MEAN_ABSOLUTE_PERCENTAGE_ERROR:
            default:
                throw new UnsupportedOperationException("Unsupported loss function: " + lossFn);
        }


        return Collections.singletonList("out");
    }

    @Override
    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig) {
        if(activation == null){
            activation = SameDiffLayerUtils.fromIActivation(globalConfig.getActivationFn());
        }
    }

    public static class Builder extends BaseSameDiffOutputLayer.Builder<Builder> {

        private int nIn;
        private int nOut;
        private Activation activation;
        private LossFunctions.LossFunction lossFn = LossFunctions.LossFunction.MSE;

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

        public Builder lossFunction(LossFunctions.LossFunction lossFn){
            this.lossFn = lossFn;
            return this;
        }

        @Override
        public SameDiffOutput build() {
            return new SameDiffOutput(this);
        }
    }
}
