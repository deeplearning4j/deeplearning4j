package org.deeplearning4j.samediff.testlayers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffLossLayer;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffOutputLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayerUtils;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.*;

@Data
@EqualsAndHashCode(callSuper = true, exclude = {"paramShapes"})
@JsonIgnoreProperties("paramShapes")
public class SameDiffLoss extends BaseSameDiffLossLayer {

    private LossFunctions.LossFunction lossFn;

    protected SameDiffLoss(Builder builder) {
        super(builder);
        this.lossFn = builder.lossFn;
    }

    private SameDiffLoss() {
        //No op constructor for Jackson
    }

    public Pair<String,String> lossKeys() {
        return new Pair<>(lossPerExampleVar(), "score");
    }

    @Override
    public String lossPerExampleVar(){
        return "lossPerEx";
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return null;
    }

    @Override
    public void defineLayer(SameDiff sd, SDVariable input, SDVariable label) {

        Pair<String,String> lossKeys = lossKeys();
        SDVariable loss;

        switch (lossFn) {
            case MSE:
                SDVariable diff = input.sub(label);
                SDVariable sqDiff = diff.mul(diff);
                SDVariable mse = sd.mean(lossKeys.getFirst(), sqDiff, 1);
                SDVariable score = sd.mean(lossKeys.getSecond(), mse);
//                SDVariable score = sd.sum(lossKeys.getSecond(), mse);
                break;
            default:
                throw new UnsupportedOperationException("Not yet implemented: " + lossFn);
        }
    }

    @Override
    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig) {

    }

    public static class Builder extends BaseSameDiffLossLayer.Builder<Builder> {

        private LossFunctions.LossFunction lossFn = LossFunctions.LossFunction.MSE;

        public Builder lossFunction(LossFunctions.LossFunction lossFn) {
            this.lossFn = lossFn;
            return this;
        }

        @Override
        public SameDiffLoss build() {
            return new SameDiffLoss(this);
        }
    }
}
