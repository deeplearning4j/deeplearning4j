package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.SpaceToDepthLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

public class KerasSpaceToDepth extends KerasLayer {

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration exception
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration exception
     */
    public KerasSpaceToDepth(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration exception
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration exception
     */
    public KerasSpaceToDepth(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        // TODO: we hard-code block size here to import YOLO9000. This size is not available as property
        // in the hdf5 file outside of the serialized lambda function (that we can't really well deserialize).
        SpaceToDepthLayer.Builder builder = new SpaceToDepthLayer.Builder()
                .blocks(2)
                .dataFormat(SpaceToDepthLayer.DataFormat.NCHW)
                .name(layerName);

        this.layer = builder.build();
        this.vertex = null;
    }

    /**
     * Get DL4J SpaceToDepth layer.
     *
     * @return SpaceToDepth layer
     */
    public SpaceToDepthLayer getSpaceToDepthLayer() {
        return (SpaceToDepthLayer) this.layer;
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
                    "Keras Space to Depth layer accepts only one input (received " + inputType.length + ")");
        return this.getSpaceToDepthLayer().getOutputType(-1, inputType[0]);
    }

}