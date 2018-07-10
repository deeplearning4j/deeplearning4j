package org.deeplearning4j.nn.modelimport.keras.layers.custom;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.DeepMojiAttentionLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Import of custom attention layer for DeepMoji application. Keras model found
 * here: https://github.com/bfelbo/DeepMoji
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasDeepMojiAttention extends KerasLayer {

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig   dictionary containing Keras layer configuration.
     *
     * @throws InvalidKerasConfigurationException Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasDeepMojiAttention(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig               dictionary containing Keras layer configuration
     * @param enforceTrainingConfig     whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasDeepMojiAttention(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        Map<String, Object> attentionParams = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);

        // NOTE: this is hard-coded for DeepMoji
        DeepMojiAttentionLayer layer = new DeepMojiAttentionLayer.Builder()
                .timeSteps(2304).nIn(30).nOut(64).build();
        this.layer = layer;
    }

    /**
     * Get DL4J DeepMoji attention layer
     *
     * @return DeepMojiAttentionLayer
     */
    public DeepMojiAttentionLayer getAttentionLayer() {
        return (DeepMojiAttentionLayer) this.layer;
    }

    /**
     * Get layer output type.
     *
     * @param  inputType    Array of InputTypes
     * @return              output type as InputType
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras DeepMojiAttentionLayer layer accepts only one input (received "
                            + inputType.length + ")");
        return this.getAttentionLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Set weights for attention layer.
     *
     * @param weights Attention layer weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();
        if (weights.containsKey("W"))
            this.weights.put(DefaultParamInitializer.WEIGHT_KEY, weights.get("W"));
        else
            throw new InvalidKerasConfigurationException(
                    "Parameter " + conf.getKERAS_PARAM_NAME_W() + " does not exist in weights");

        if (weights.size() > 1) {
            Set<String> paramNames = weights.keySet();
            paramNames.remove(conf.getKERAS_PARAM_NAME_W());
            String unknownParamNames = paramNames.toString();
            log.warn("Attemping to set weights for unknown parameters: "
                    + unknownParamNames.substring(1, unknownParamNames.length() - 1));
        }
    }
}

