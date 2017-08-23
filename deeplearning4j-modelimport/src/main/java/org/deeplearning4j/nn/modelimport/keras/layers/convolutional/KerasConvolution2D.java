package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Imports a 2D Convolution layer from Keras.
 *
 * @author dave@skymind.io
 */
@Slf4j
@Data
public class KerasConvolution2D extends KerasConvolution {

    /**
     * Pass-through constructor from KerasLayer
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasConvolution2D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasConvolution2D(Map<String, Object> layerConfig)
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
    public KerasConvolution2D(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        hasBias = getHasBiasFromConfig(layerConfig);
        numTrainableParams = hasBias ? 2 : 1;

        ConvolutionLayer.Builder builder = new ConvolutionLayer.Builder().name(this.layerName)
                        .nOut(getNOutFromConfig(layerConfig)).dropOut(this.dropout)
                        .activation(getActivationFromConfig(layerConfig))
                        .weightInit(getWeightInitFromConfig(
                                layerConfig, conf.getLAYER_FIELD_INIT(), enforceTrainingConfig))
                        .biasInit(0.0)
                        .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
                        .convolutionMode(getConvolutionModeFromConfig(layerConfig))
                        .kernelSize(getKernelSizeFromConfig(layerConfig, 2))
                        .hasBias(hasBias).stride(getStrideFromConfig(layerConfig, 2));
        int[] padding = getPaddingFromBorderModeConfig(layerConfig, 2);
        if (padding != null)
            builder.padding(padding);
        this.layer = builder.build();
    }

    /**
     * Get DL4J ConvolutionLayer.
     *
     * @return  ConvolutionLayer
     */
    public ConvolutionLayer getConvolution2DLayer() {
        return (ConvolutionLayer) this.layer;
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
                            "Keras Convolution layer accepts only one input (received " + inputType.length + ")");
        return this.getConvolution2DLayer().getOutputType(-1, inputType[0]);
    }

}
