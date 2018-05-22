package org.deeplearning4j.nn.modelimport.keras.layers.embeddings;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasConstraintUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getHasBiasFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getNOutFromConfig;

/**
 * Imports an Embedding layer from Keras.
 *
 * @author dave@skymind.io
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasEmbedding extends KerasLayer {

    private final int NUM_TRAINABLE_PARAMS = 1;
    private boolean hasZeroMasking;
    private int inputDim;
    private int inputLength;
    private boolean inferInputLength;


    /**
     * Pass through constructor for unit tests
     *
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasEmbedding() throws UnsupportedKerasConfigurationException {
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasEmbedding(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasEmbedding(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        this.inputDim = getInputDimFromConfig(layerConfig);
        this.inputLength = getInputLengthFromConfig(layerConfig);
        this.inferInputLength = this.inputLength == 0;
        if (this.inferInputLength)
            this.inputLength = 1; // set dummy value, so shape inference works

        this.hasZeroMasking = KerasLayerUtils.getZeroMaskingFromConfig(layerConfig, conf);
        if (hasZeroMasking)
            log.warn("Masking in keras and DL4J work differently. We do not completely support mask_zero flag " +
                    "on Embedding layers. Zero Masking for the Embedding layer only works with unidirectional LSTM for now."
                    + " If you want to have this behaviour for your imported model " +
                    "in DL4J, apply masking as a pre-processing step to your input." +
                    "See https://deeplearning4j.org/usingrnns#masking for more on this.");

        Pair<WeightInit, Distribution> init = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_EMBEDDING_INIT(),
                enforceTrainingConfig, conf, kerasMajorVersion);
        WeightInit weightInit = init.getFirst();
        Distribution distribution = init.getSecond();

        LayerConstraint embeddingConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_EMBEDDINGS_CONSTRAINT(), conf, kerasMajorVersion);

        EmbeddingSequenceLayer.Builder builder = new EmbeddingSequenceLayer.Builder()
                .name(this.layerName)
                .nIn(inputDim)
                .inputLength(inputLength)
                .inferInputLength(inferInputLength)
                .nOut(getNOutFromConfig(layerConfig, conf))
                .dropOut(this.dropout).activation(Activation.IDENTITY)
                .weightInit(weightInit)
                .biasInit(0.0)
                .l1(this.weightL1Regularization)
                .l2(this.weightL2Regularization)
                .hasBias(false);
        if (distribution != null)
            builder.dist(distribution);
        if (embeddingConstraint != null)
            builder.constrainWeights(embeddingConstraint);
        this.layer = builder.build();
    }

    /**
     * Get DL4J Embedding Sequence layer.
     *
     * @return Embedding Sequence layer
     */
    public EmbeddingSequenceLayer getEmbeddingLayer() {
        return (EmbeddingSequenceLayer) this.layer;
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
     * @return number of trainable parameters (1)
     */
    @Override
    public int getNumParams() {
        return NUM_TRAINABLE_PARAMS;
    }

    /**
     * Set weights for layer.
     *
     * @param weights Embedding layer weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();
        if (!weights.containsKey(conf.getLAYER_FIELD_EMBEDDING_WEIGHTS()))
            throw new InvalidKerasConfigurationException(
                    "Parameter " + conf.getLAYER_FIELD_EMBEDDING_WEIGHTS() + " does not exist in weights");
        INDArray kernel = weights.get(conf.getLAYER_FIELD_EMBEDDING_WEIGHTS());
        if (this.hasZeroMasking) {
            kernel.putRow(0, Nd4j.zeros(kernel.columns()));
        }
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
     * Get Keras input length from Keras layer configuration. In Keras input_length, if present, denotes
     * the number of indices to embed per mini-batch, i.e. input will be of of shape (mb, input_length)
     * and (mb, 1) else.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return input length as int
     */
    private int getInputLengthFromConfig(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(conf.getLAYER_FIELD_INPUT_LENGTH()))
            throw new InvalidKerasConfigurationException(
                    "Keras Embedding layer config missing " + conf.getLAYER_FIELD_INPUT_LENGTH() + " field");
        if (innerConfig.get(conf.getLAYER_FIELD_INPUT_LENGTH()) == null) {
            return 0;
        } else {
            return (int) innerConfig.get(conf.getLAYER_FIELD_INPUT_LENGTH());
        }
    }

    /**
     * Get Keras input dimension from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return input dim as int
     */
    private int getInputDimFromConfig(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(conf.getLAYER_FIELD_INPUT_DIM()))
            throw new InvalidKerasConfigurationException(
                    "Keras Embedding layer config missing " + conf.getLAYER_FIELD_INPUT_DIM() + " field");
        return (int) innerConfig.get(conf.getLAYER_FIELD_INPUT_DIM());
    }
}
