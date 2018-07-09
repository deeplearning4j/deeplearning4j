package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.Map;
/**
 * Attention layer for DeepMoji network architecture.
 *
 *
 * @author  Max Pumperla
 */
@Data
public class DeepMojiAttentionLayer extends SameDiffLayer {

    // TODO: Masking
    private int channels;
    private final double EPS = 1e-7;
    private int nIn;

    public DeepMojiAttentionLayer(int channels, WeightInit weightInit) {
        this.weightInit = weightInit;
        this.channels = channels;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for DeepMoji attention layer (layer name=\"" + getLayerName()
                    + "\"): Expected RNN input, got " + inputType);
        }
        InputType.InputTypeRecurrent rnnType = (InputType.InputTypeRecurrent) inputType;
        this.nIn = (int) rnnType.getTimeSeriesLength();
    }

    public DeepMojiAttentionLayer(int channels) {
        this(channels, WeightInit.UNIFORM);
    }

    /**
     * In the defineLayer method, you define the actual layer forward pass
     * For this layer, we are returning out = activationFn( input*weights + bias)
     *
     * @param sd         The SameDiff instance for this layer
     * @param layerInput An SDVariable representing the 4D inputs to this layer
     * @param paramTable Layer parameters
     * @return layer output
     */
    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput, Map<String, SDVariable> paramTable) {
        SDVariable weights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);

        SDVariable logits = sd.mmul(layerInput, weights);
        SDVariable reshapedLogits = sd.reshape(logits, layerInput.getShape()[0], layerInput.getShape()[1]);
        SDVariable ai = sd.exp(reshapedLogits);
        SDVariable aiSum = sd.sum(ai, 1);
        SDVariable aiSumEps = aiSum.add(EPS);
        SDVariable attentionWeights = ai.div(aiSumEps);
        SDVariable weightedInput = layerInput.mul(sd.expandDims(attentionWeights, 2));

        return sd.sum(weightedInput, 1);
    }


    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for DeepMoji attention layer (layer name=\"" + getLayerName()
                    + "\"): Expected RNN input, got " + inputType);
        }
        InputType.InputTypeRecurrent rnnType = (InputType.InputTypeRecurrent) inputType;
        long tsLength = rnnType.getTimeSeriesLength();

        // Layer will "average out" second dimension
        return InputType.feedForward(tsLength);
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam(DefaultParamInitializer.WEIGHT_KEY, channels, 1);
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        initWeights(channels, 1, weightInit,  params.get(DefaultParamInitializer.WEIGHT_KEY));
    }
}
