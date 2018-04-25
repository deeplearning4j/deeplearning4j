package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.getPaddingFromConfig;

/**
 * Imports a Keras Cropping 1D layer.
 *
 * @author Max Pumperla
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasCropping1D extends KerasLayer {

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasCropping1D(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasCropping1D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        String croppingField = conf.getLAYER_FIELD_CROPPING();
        int[] cropping = getPaddingFromConfig(layerConfig, conf, croppingField, 1);
        Cropping1D.Builder builder = new Cropping1D.Builder(cropping)
                .name(this.layerName).dropOut(this.dropout);
        this.layer = builder.build();
        this.vertex = null;
    }

    /**
     * Get DL4J Cropping1D layer.
     *
     * @return Cropping1D layer
     */
    public Cropping1D getCropping1DLayer() {
        return (Cropping1D) this.layer;
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Cropping layer accepts only one input (received " + inputType.length + ")");
        return this.getCropping1DLayer().getOutputType(-1, inputType[0]);
    }
}
