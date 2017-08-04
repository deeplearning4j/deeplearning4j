package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InputType.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.TensorFlowCnnToFeedForwardPreProcessor;

import java.util.Map;

/**
 * Imports a Keras Flatten layer as a DL4J {Cnn,Rnn}ToFeedForwardInputPreProcessor.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasFlatten extends KerasLayer {

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasFlatten(Map<String, Object> layerConfig)
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
    public KerasFlatten(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
    }

    /**
     * Whether this Keras layer maps to a DL4J InputPreProcessor.
     *
     * @return      true
     */
    @Override
    public boolean isInputPreProcessor() {
        return true;
    }

    /**
     * Gets appropriate DL4J InputPreProcessor for given InputTypes.
     *
     * @param  inputType    Array of InputTypes
     * @return              DL4J InputPreProcessor
     * @throws InvalidKerasConfigurationException
     * @see org.deeplearning4j.nn.conf.InputPreProcessor
     */
    @Override
    public InputPreProcessor getInputPreprocessor(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                            "Keras Flatten layer accepts only one input (received " + inputType.length + ")");
        InputPreProcessor preprocessor = null;
        if (inputType[0] instanceof InputTypeConvolutional) {
            InputTypeConvolutional it = (InputTypeConvolutional) inputType[0];
            switch (this.getDimOrder()) {
                case NONE:
                case THEANO:
                    preprocessor = new CnnToFeedForwardPreProcessor(it.getHeight(), it.getWidth(), it.getDepth());
                    break;
                case TENSORFLOW:
                    preprocessor = new TensorFlowCnnToFeedForwardPreProcessor(it.getHeight(), it.getWidth(),
                                    it.getDepth());
                    break;
                default:
                    throw new InvalidKerasConfigurationException("Unknown Keras backend " + this.getDimOrder());
            }
        } else if (inputType[0] instanceof InputType.InputTypeRecurrent) {
            preprocessor = new RnnToFeedForwardPreProcessor();
        }
        return preprocessor;
    }

    /**
     * Get layer output type.
     *
     * @param  inputType    Array of InputTypes
     * @return              output type as InputType
     * @throws InvalidKerasConfigurationException
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                            "Keras Flatten layer accepts only one input (received " + inputType.length + ")");
        return getInputPreprocessor(inputType).getOutputType(inputType[0]);
    }
}
