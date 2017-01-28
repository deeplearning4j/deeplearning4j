package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

import java.util.Map;

/**
 * Imports an Activation layer from Keras.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasActivation extends KerasLayer {

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig   dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasActivation(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig               dictionary containing Keras layer configuration
     * @param enforceTrainingConfig     whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasActivation(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        this.layer = new ActivationLayer.Builder()
            .name(this.layerName)
            .activation(getActivationFromConfig(layerConfig))
            .build();
    }

    /**
     * Get layer output type.
     *
     * @param  inputType    Array of InputTypes
     * @return              output type as InputType
     * @throws InvalidKerasConfigurationException
     */
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException("Keras Activation layer accepts only one input (received " + inputType.length + ")");
        return this.getActivationLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Get DL4J ActivationLayer.
     *
     * @return  ActivationLayer
     */
    public ActivationLayer getActivationLayer() {
        return (ActivationLayer)this.layer;
    }
}
