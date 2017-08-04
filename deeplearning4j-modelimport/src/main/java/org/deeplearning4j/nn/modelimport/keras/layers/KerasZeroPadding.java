package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

import java.util.List;
import java.util.Map;

/**
 * Imports a Keras ZeroPadding layer as a DL4J Subsampling layer
 * with kernel size 1 and stride 1.
 *
 * TODO: change this to official DL4J ZeroPadding layer once it's
 * supported
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasZeroPadding extends KerasLayer {

    public static final String LAYER_FIELD_PADDING = "padding";

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig   dictionary containing Keras layer configuration.
     *
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasZeroPadding(Map<String, Object> layerConfig)
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
    public KerasZeroPadding(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        ZeroPaddingLayer.Builder builder = new ZeroPaddingLayer.Builder(getPaddingFromConfig(layerConfig))
                        .name(this.layerName).dropOut(this.dropout);
        this.layer = builder.build();
        this.vertex = null;
    }

    /**
     * Get DL4J SubsamplingLayer.
     *
     * @return  SubsamplingLayer
     */
    public ZeroPaddingLayer getZeroPaddingLayer() {
        return (ZeroPaddingLayer) this.layer;
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
                            "Keras ZeroPadding layer accepts only one input (received " + inputType.length + ")");
        return this.getZeroPaddingLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Get zero padding from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public int[] getPaddingFromConfig(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_PADDING))
            throw new InvalidKerasConfigurationException(
                            "Field " + LAYER_FIELD_PADDING + " not found in Keras ZeroPadding layer");
        List<Integer> paddingList = (List<Integer>) innerConfig.get(LAYER_FIELD_PADDING);
        switch (this.className) {
            case LAYER_CLASS_NAME_ZERO_PADDING_2D:
                if (paddingList.size() == 2) {
                    paddingList.add(paddingList.get(1));
                    paddingList.add(1, paddingList.get(0));
                }
                if (paddingList.size() != 4)
                    throw new InvalidKerasConfigurationException("Found Keras ZeroPadding2D layer with invalid "
                                    + paddingList.size() + "D padding.");
                break;
            case LAYER_CLASS_NAME_ZERO_PADDING_1D:
                throw new UnsupportedKerasConfigurationException("Keras ZeroPadding1D layer not supported");
            default:
                throw new UnsupportedKerasConfigurationException(
                                "Keras " + this.className + " padding layer not supported");
        }

        int[] padding = new int[paddingList.size()];
        for (int i = 0; i < paddingList.size(); i++)
            padding[i] = paddingList.get(i);
        return padding;
    }
}
