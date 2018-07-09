package org.deeplearning4j.nn.modelimport.keras.layers.custom;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.DeepMojiAttentionLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;

import java.util.Map;

/**
 * Import of custom attention layer for DeepMoji application.
 *
 * @author Max Pumperla
 */
public class KerasDeepMojiAttention extends KerasLayer {

    // TODO: set weights

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
        // TODO: get channels properly

        DeepMojiAttentionLayer layer = new DeepMojiAttentionLayer(1);
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
}

