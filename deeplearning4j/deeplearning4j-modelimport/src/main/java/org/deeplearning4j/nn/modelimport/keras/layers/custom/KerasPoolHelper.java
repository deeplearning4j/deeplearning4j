package org.deeplearning4j.nn.modelimport.keras.layers.custom;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.graph.PoolHelperVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

/**
 * Custom PoolHelper layer developed for importing GoogLeNet. This layer strips
 * the first column and row of the input. See https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class KerasPoolHelper extends KerasLayer {

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig   dictionary containing Keras layer configuration.
     *
     * @throws InvalidKerasConfigurationException Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasPoolHelper(Map<String, Object> layerConfig)
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
    public KerasPoolHelper(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        this.vertex = new PoolHelperVertex();
    }

    /**
     * Get layer output type.
     *
     * @param  inputType    Array of InputTypes
     * @return              output type as InputType
     */
    @Override
    public InputType getOutputType(InputType... inputType) {
        return this.vertex.getOutputType(-1, inputType);
    }
}
