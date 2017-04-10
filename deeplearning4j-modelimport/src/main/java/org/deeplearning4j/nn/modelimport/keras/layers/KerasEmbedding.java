package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Imports an Embedding layer from Keras.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasEmbedding extends KerasLayer {

    public static final String LAYER_FIELD_INPUT_DIM = "input_dim";

    /* Keras layer parameter names. */
    public static final int NUM_TRAINABLE_PARAMS = 1;
    public static final String KERAS_PARAM_NAME_W = "W";
    public static final String KERAS_PARAM_NAME_B = "b";

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasEmbedding(Map<String, Object> layerConfig)
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
    public KerasEmbedding(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        int inputDim = getInputDimFromConfig(layerConfig);
        int[] inputShapeOld = this.inputShape;
        this.inputShape = new int[inputShapeOld.length + 1];
        this.inputShape[0] = inputShapeOld[0];
        this.inputShape[1] = inputDim;
        /* TODO: what about mask_zero field? */

        /* TODO: fix this once we have a solution for fixing layer parameters. */
        if (enforceTrainingConfig)
            throw new UnsupportedKerasConfigurationException(
                            "DL4J EmbeddingLayer includes bias but Keras Embedding does not");
        else
            log.warn("DL4J EmbeddingLayer includes bias but Keras Embedding does not.");

        this.layer = new EmbeddingLayer.Builder().name(this.layerName).nIn(inputDim)
                        .nOut(getNOutFromConfig(layerConfig)).dropOut(this.dropout).activation(Activation.IDENTITY)
                        .weightInit(getWeightInitFromConfig(layerConfig, enforceTrainingConfig)).biasInit(0.0)
                        .l1(this.weightL1Regularization).l2(this.weightL2Regularization).build();
    }

    /**
     * Get DL4J DenseLayer.
     *
     * @return  DenseLayer
     */
    public EmbeddingLayer getEmbeddingLayer() {
        return (EmbeddingLayer) this.layer;
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
        /* Check whether layer requires a preprocessor for this InputType. */
        InputPreProcessor preprocessor = getInputPreprocessor(inputType[0]);
        if (preprocessor != null) {
            return this.getEmbeddingLayer().getOutputType(-1, preprocessor.getOutputType(inputType[0]));
        }
        return this.getEmbeddingLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return          number of trainable parameters (1)
     */
    @Override
    public int getNumParams() {
        return NUM_TRAINABLE_PARAMS;
    }

    /**
     * Set weights for layer.
     *
     * @param weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<String, INDArray>();
        if (!weights.containsKey(KERAS_PARAM_NAME_W))
            throw new InvalidKerasConfigurationException(
                            "Parameter " + KERAS_PARAM_NAME_W + " does not exist in weights");
        INDArray W = weights.get(KERAS_PARAM_NAME_W);
        if (!weights.containsKey(KERAS_PARAM_NAME_B)) {
            log.warn("Setting DL4J EmbeddingLayer bias to zero.");
            weights.put(KERAS_PARAM_NAME_B, Nd4j.zeros(W.size(1)));
        }
        this.weights.put(DefaultParamInitializer.WEIGHT_KEY, W);
        this.weights.put(DefaultParamInitializer.BIAS_KEY, weights.get(KERAS_PARAM_NAME_B));

        if (weights.size() > 2) {
            Set<String> paramNames = weights.keySet();
            paramNames.remove(KERAS_PARAM_NAME_W);
            String unknownParamNames = paramNames.toString();
            log.warn("Attemping to set weights for unknown parameters: "
                            + unknownParamNames.substring(1, unknownParamNames.length() - 1));
        }
    }

    /**
     * Get Keras input shape from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return                  input dim as int
     */
    private int getInputDimFromConfig(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_INPUT_DIM))
            throw new InvalidKerasConfigurationException(
                            "Keras Embedding layer config missing " + LAYER_FIELD_INPUT_DIM + " field");
        return (int) innerConfig.get(LAYER_FIELD_INPUT_DIM);
    }
}
