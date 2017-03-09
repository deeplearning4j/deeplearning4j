package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

import java.util.ArrayList;
import java.util.Map;

/**
 * Imports an Input layer from Keras. Used to set InputType of DL4J model.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasInput extends KerasLayer {

    public static final int NO_TRUNCATED_BPTT = 0;

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig   dictionary containing Keras layer configuration.
     *
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasInput(Map<String, Object> layerConfig)
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
    public KerasInput(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        if (this.inputShape.length < 0 || this.inputShape.length > 3)
            throw new UnsupportedKerasConfigurationException(
                            "Inputs with " + this.inputShape.length + " dimensions not supported");
    }

    /**
     * Constructor from layer name and input shape.
     *
     * @param layerName     layer name
     * @param inputShape    input shape as array
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasInput(String layerName, int[] inputShape) throws UnsupportedKerasConfigurationException {
        this(layerName, inputShape, true);
    }

    /**
     * Constructor from layer name and input shape.
     *
     * @param layerName                 layer name
     * @param inputShape                input shape as array
     * @param enforceTrainingConfig     whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasInput(String layerName, int[] inputShape, boolean enforceTrainingConfig)
                    throws UnsupportedKerasConfigurationException {
        this.className = LAYER_CLASS_NAME_INPUT;
        this.layerName = layerName;
        this.inputShape = inputShape;
        this.inboundLayerNames = new ArrayList<String>();
        this.layer = null;
        this.vertex = null;
        if (this.inputShape.length < 0 || this.inputShape.length > 3)
            throw new UnsupportedKerasConfigurationException(
                            "Inputs with " + this.inputShape.length + " dimensions not supported");
    }

    /**
     * Get layer output type.
     *
     * @param  inputType    Array of InputTypes
     * @return              output type as InputType
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    @Override
    public InputType getOutputType(InputType... inputType)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        if (inputType.length > 0)
            log.warn("Keras Input layer does not accept inputs (received " + inputType.length + "). Ignoring.");
        InputType myInputType;
        switch (this.inputShape.length) {
            case 1:
                myInputType = new InputType.InputTypeFeedForward(this.inputShape[0]);
                break;
            case 2:
                myInputType = new InputType.InputTypeRecurrent(this.inputShape[1]);
                break;
            case 3:
                switch (this.dimOrder) {
                    case TENSORFLOW:
                        /* TensorFlow convolutional input: # rows, # cols, # channels */
                        myInputType = new InputType.InputTypeConvolutional(this.inputShape[0], this.inputShape[1],
                                        this.inputShape[2]);
                        break;
                    case THEANO:
                        /* Theano convolutional input:     # channels, # rows, # cols */
                        myInputType = new InputType.InputTypeConvolutional(this.inputShape[1], this.inputShape[2],
                                        this.inputShape[0]);
                        break;
                    default:
                        throw new UnsupportedKerasConfigurationException(
                                        "3D (convolutional) inputs with unknown dimension ordering " + this.dimOrder
                                                        + " are not supported.");
                }
                break;
            default:
                throw new UnsupportedKerasConfigurationException(
                                "Inputs with " + this.inputShape.length + " dimensions not supported");
        }
        return myInputType;
    }

    /**
     * Returns value of truncated BPTT, if any found.
     *
     * TODO: figure out how to infer truncated BPTT value for non-sequence inputs
     *
     * In Keras, truncated BPTT is specified implicitly by specifying a fixed
     * size input and by passing in the "unroll" argument to recurrent layers.
     * Currently, the only setting in which we can confidently determine the
     * value of truncated BPTT is if the original input has two dimensions,
     * the first of which is sequence length. Hypothetically, we should be
     * able to do this for other types of inputs, but that's less straightforward.
     *
     * @return          value of truncated BPTT
     */
    public int getTruncatedBptt() {
        if (this.inputShape.length == 2 && this.inputShape[0] > 0)
            return this.inputShape[0];
        return NO_TRUNCATED_BPTT;
    }
}
