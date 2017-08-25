package org.deeplearning4j.nn.modelimport.keras.layers.embeddings;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getNOutFromConfig;

/**
 * Imports an Embedding layer from Keras.
 *
 * @author dave@skymind.io
 */
@Slf4j
@Data
public class KerasEmbedding extends KerasLayer {

    private final int NUM_TRAINABLE_PARAMS = 1;

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

        this.layer = new EmbeddingLayer.Builder().name(this.layerName).nIn(inputDim)
                        .nOut(getNOutFromConfig(layerConfig, conf)).dropOut(this.dropout).activation(Activation.IDENTITY)
                        .weightInit(getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_EMBEDDING_INIT(),
                                enforceTrainingConfig, conf, kerasMajorVersion))
                        .biasInit(0.0)
                        .l1(this.weightL1Regularization).l2(this.weightL2Regularization).hasBias(false).build();
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
        if (!weights.containsKey(conf.getLAYER_FIELD_EMBEDDING_WEIGHTS()))
            throw new InvalidKerasConfigurationException(
                            "Parameter " + conf.getLAYER_FIELD_EMBEDDING_WEIGHTS() + " does not exist in weights");
        INDArray kernel = weights.get(conf.getLAYER_FIELD_EMBEDDING_WEIGHTS());
        this.weights.put(DefaultParamInitializer.WEIGHT_KEY, kernel);

        if (weights.size() > 2) {
            Set<String> paramNames = weights.keySet();
            paramNames.remove(conf.getLAYER_FIELD_EMBEDDING_WEIGHTS());
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
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(conf.getLAYER_FIELD_INPUT_DIM()))
            throw new InvalidKerasConfigurationException(
                            "Keras Embedding layer config missing " + conf.getLAYER_FIELD_INPUT_DIM() + " field");
        return (int) innerConfig.get(conf.getLAYER_FIELD_INPUT_DIM());
    }
}
