package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

import java.util.Map;

/**
 * Imports a Keras Pooling layer as a DL4J Subsampling layer.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasPooling extends KerasLayer {

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig   dictionary containing Keras layer configuration.
     *
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasPooling(Map<String,Object> layerConfig)
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
    public KerasPooling(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        SubsamplingLayer.Builder builder = new SubsamplingLayer.Builder(mapPoolingType(this.className))
            .name(this.layerName)
            .dropOut(this.dropout)
            .convolutionMode(getConvolutionModeFromConfig(layerConfig))
            .kernelSize(getKernelSizeFromConfig(layerConfig))
            .stride(getStrideFromConfig(layerConfig));
        int[] padding = getPaddingFromConfig(layerConfig);
        if (padding != null)
            builder.padding(padding);
        this.layer = builder.build();
        this.vertex = null;
    }

    /**
     * Get DL4J SubsamplingLayer.
     *
     * @return  SubsamplingLayer
     */
    public SubsamplingLayer getSubsamplingLayer() {
        return (SubsamplingLayer)this.layer;
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
            throw new InvalidKerasConfigurationException("Keras Subsampling layer accepts only one input (received " + inputType.length + ")");
        return this.getSubsamplingLayer().getOutputType(-1, inputType[0]);
    }
}
