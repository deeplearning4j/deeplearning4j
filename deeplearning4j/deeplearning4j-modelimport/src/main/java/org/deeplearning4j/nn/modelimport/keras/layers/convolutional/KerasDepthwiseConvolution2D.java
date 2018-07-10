package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DepthwiseConvolution2D;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.embeddings.KerasEmbedding;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasConstraintUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasRegularizerUtils;
import org.deeplearning4j.nn.params.SeparableConvolutionParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasConvolutionUtils.*;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils.getActivationFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getHasBiasFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getNOutFromConfig;


/**
 * Keras depth-wise convolution 2D layer support
 *
 * @author Max Pumperla
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasDepthwiseConvolution2D extends KerasConvolution {


    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasDepthwiseConvolution2D(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasDepthwiseConvolution2D(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, Collections.<String, KerasLayer>emptyMap(), true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasDepthwiseConvolution2D(Map<String, Object> layerConfig,
                                       Map<String, ? extends KerasLayer> previousLayers)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, previousLayers, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasDepthwiseConvolution2D(Map<String, Object> layerConfig,
                                       Map<String, ? extends KerasLayer> previousLayers, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, previousLayers, null, enforceTrainingConfig);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasDepthwiseConvolution2D(Map<String, Object> layerConfig,
                                       Map<String, ? extends KerasLayer> previousLayers,
                                       List<String> layerNamesToCheck, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        if (layerNamesToCheck != null) {
            inboundLayerNames.addAll(layerNamesToCheck);
        }
        hasBias = getHasBiasFromConfig(layerConfig, conf);
        numTrainableParams = hasBias ? 2 : 1;
        int[] dilationRate = getDilationRate(layerConfig, 2, conf, false);

        Pair<WeightInit, Distribution> depthWiseInit = getWeightInitFromConfig(layerConfig,
                conf.getLAYER_FIELD_DEPTH_WISE_INIT(), enforceTrainingConfig, conf, kerasMajorVersion);
        WeightInit depthWeightInit = depthWiseInit.getFirst();
        Distribution depthDistribution = depthWiseInit.getSecond();

        val nIn = getNInFromConfig(previousLayers);

        int depthMultiplier = getDepthMultiplier(layerConfig, conf);

        this.weightL1Regularization = KerasRegularizerUtils.getWeightRegularizerFromConfig(
                layerConfig, conf, conf.getLAYER_FIELD_DEPTH_WISE_REGULARIZER(), conf.getREGULARIZATION_TYPE_L1());
        this.weightL2Regularization = KerasRegularizerUtils.getWeightRegularizerFromConfig(
                layerConfig, conf, conf.getLAYER_FIELD_DEPTH_WISE_REGULARIZER(), conf.getREGULARIZATION_TYPE_L2());


        LayerConstraint biasConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_B_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint depthWiseWeightConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_DEPTH_WISE_CONSTRAINT(), conf, kerasMajorVersion);


        DepthwiseConvolution2D.Builder builder = new DepthwiseConvolution2D.Builder().name(this.layerName)
                .dropOut(this.dropout)
                .nIn(nIn)
                .nOut(nIn * depthMultiplier)
                .activation(getActivationFromConfig(layerConfig, conf))
                .weightInit(depthWeightInit)
                .depthMultiplier(depthMultiplier)
                .l1(this.weightL1Regularization).l2(this.weightL2Regularization)
                .convolutionMode(getConvolutionModeFromConfig(layerConfig, conf))
                .kernelSize(getKernelSizeFromConfig(layerConfig, 2, conf, kerasMajorVersion))
                .hasBias(hasBias)
                .stride(getStrideFromConfig(layerConfig, 2, conf));
        int[] padding = getPaddingFromBorderModeConfig(layerConfig, 2, conf, kerasMajorVersion);
        if (depthDistribution != null)
            builder.dist(depthDistribution);
        if (hasBias)
            builder.biasInit(0.0);
        if (padding != null)
            builder.padding(padding);
        if (dilationRate != null)
            builder.dilation(dilationRate);
        if (biasConstraint != null)
            builder.constrainBias(biasConstraint);
        if (depthWiseWeightConstraint != null)
            builder.constrainWeights(depthWiseWeightConstraint);
        this.layer = builder.build();
    }

    /**
     * Set weights for layer.
     *
     * @param weights Map of weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();

        INDArray dW;
        if (weights.containsKey(conf.getLAYER_PARAM_NAME_DEPTH_WISE_KERNEL()))
            dW = weights.get(conf.getLAYER_PARAM_NAME_DEPTH_WISE_KERNEL());
        else
            throw new InvalidKerasConfigurationException(
                    "Keras DepthwiseConvolution2D layer does not contain parameter "
                            + conf.getLAYER_PARAM_NAME_DEPTH_WISE_KERNEL());

        this.weights.put(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, dW);
        if (hasBias) {
            INDArray bias;
            if (kerasMajorVersion == 2 && weights.containsKey("bias"))
                bias = weights.get("bias");
            else if (kerasMajorVersion == 1 && weights.containsKey("b"))
                bias = weights.get("b");
            else
                throw new InvalidKerasConfigurationException(
                        "Keras DepthwiseConvolution2D layer does not contain bias parameter");
            this.weights.put(SeparableConvolutionParamInitializer.BIAS_KEY, bias);

        }

    }

    /**
     * Get DL4J DepthwiseConvolution2D.
     *
     * @return DepthwiseConvolution2D
     */
    public DepthwiseConvolution2D getDepthwiseConvolution2DLayer() {
        return (DepthwiseConvolution2D) this.layer;
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
                    "Keras depth-wise convolution 2D layer accepts only one input (received " + inputType.length + ")");
        return this.getDepthwiseConvolution2DLayer().getOutputType(-1, inputType[0]);
    }

}