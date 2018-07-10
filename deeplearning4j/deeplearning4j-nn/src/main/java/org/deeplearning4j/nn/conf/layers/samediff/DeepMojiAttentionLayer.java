package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
/**
 * Attention layer for DeepMoji network architecture, following the implementation
 * here: https://github.com/bfelbo/DeepMoji
 *
 *
 * @author  Max Pumperla
 */
@Data
public class DeepMojiAttentionLayer extends SameDiffLayer {

    private int timeSteps;
    private long nIn;
    private long nOut;

    private final double EPS = 1e-7;


    public DeepMojiAttentionLayer(Builder builder) {
        super(builder);

        this.nIn = builder.nIn;
        this.nOut = builder.nOut;
        this.weightInit = builder.weightInit;
        this.timeSteps = builder.timeSteps;
    }

    private DeepMojiAttentionLayer(){
        //No op (Jackson)
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for DeepMoji attention layer (layer name=\"" + getLayerName()
                    + "\"): Expected RNN input, got " + inputType);
        }
        if (override) {
            InputType.InputTypeRecurrent rnnType = (InputType.InputTypeRecurrent) inputType;
            this.nIn = (int) rnnType.getSize();
        }
    }

    /**
     * This attention layer computes a weighted average over all "channels" dimensions, i.e.
     * for an input of shape (mb, channels, timeSteps) it will compute an output of shape
     * (mb, timeSteps).
     *
     * @param sd         The SameDiff instance for this layer
     * @param layerInput An SDVariable representing the 4D inputs to this layer
     * @param paramTable Layer parameters
     * @return layer output
     */
    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput, Map<String, SDVariable> paramTable) {
        SDVariable weights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);

        SDVariable logits = sd.tensorMmul(layerInput, weights, new int[][] { {2}, {0}});
        SDVariable reshapedLogits = sd.reshape(logits, layerInput.getShape()[0], layerInput.getShape()[1]);
        SDVariable ai = sd.exp(reshapedLogits);
        SDVariable aiSum = sd.sum(ai, 1);
        SDVariable aiSumEps = sd.expandDims(aiSum.add(EPS), 1);
        SDVariable attentionWeights = ai.div(aiSumEps);
        SDVariable weightedInput = layerInput.mul(sd.expandDims(attentionWeights, 2));

        return sd.sum(weightedInput, 2);
    }


    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for DeepMoji attention layer (layer name=\"" + getLayerName()
                    + "\"): Expected RNN input, got " + inputType);
        }
        InputType.InputTypeRecurrent rnnType = (InputType.InputTypeRecurrent) inputType;
        long size = rnnType.getSize();

        // Layer will "average out" time-step dimension
        return InputType.feedForward(size);
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam(DefaultParamInitializer.WEIGHT_KEY, nIn, 1);
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        initWeights( (int) nIn, 1, weightInit,  params.get(DefaultParamInitializer.WEIGHT_KEY));
    }

    public static class Builder extends SameDiffLayer.Builder<Builder> {

        private int nIn;
        private int nOut;
        private int timeSteps;

        public Builder nIn(int nIn){
            this.nIn = nIn;
            return this;
        }

        public Builder nOut(int nOut){
            this.nOut = nOut;
            return this;
        }

        public Builder timeSteps(int timeSteps){
            this.timeSteps = timeSteps;
            return this;
        }

        @Override
        public DeepMojiAttentionLayer build() {
            return new DeepMojiAttentionLayer(this);
        }
    }
}
