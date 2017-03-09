package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Imports a Dense layer from Keras.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasDense extends KerasLayer {

    /* Keras layer parameter names. */
    public static final int NUM_TRAINABLE_PARAMS = 2;
    public static final String KERAS_PARAM_NAME_W = "W";
    public static final String KERAS_PARAM_NAME_B = "b";

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasDense(Map<String, Object> layerConfig)
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
    public KerasDense(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        this.layer = new DenseLayer.Builder().name(this.layerName).nOut(getNOutFromConfig(layerConfig))
                        .dropOut(this.dropout).activation(getActivationFromConfig(layerConfig))
                        .weightInit(getWeightInitFromConfig(layerConfig, enforceTrainingConfig)).biasInit(0.0)
                        .l1(this.weightL1Regularization).l2(this.weightL2Regularization).build();
    }

    /**
     * Get DL4J DenseLayer.
     *
     * @return  DenseLayer
     */
    public DenseLayer getDenseLayer() {
        return (DenseLayer) this.layer;
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
            return this.getDenseLayer().getOutputType(-1, preprocessor.getOutputType(inputType[0]));
        }
        return this.getDenseLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return          number of trainable parameters (2)
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
        if (weights.containsKey(KERAS_PARAM_NAME_W))
            this.weights.put(DefaultParamInitializer.WEIGHT_KEY, weights.get(KERAS_PARAM_NAME_W));
        else
            throw new InvalidKerasConfigurationException(
                            "Parameter " + KERAS_PARAM_NAME_W + " does not exist in weights");
        if (weights.containsKey(KERAS_PARAM_NAME_B))
            this.weights.put(DefaultParamInitializer.BIAS_KEY, weights.get(KERAS_PARAM_NAME_B));
        else
            throw new InvalidKerasConfigurationException(
                            "Parameter " + KERAS_PARAM_NAME_B + " does not exist in weights");
        if (weights.size() > 2) {
            Set<String> paramNames = weights.keySet();
            paramNames.remove(KERAS_PARAM_NAME_W);
            paramNames.remove(KERAS_PARAM_NAME_B);
            String unknownParamNames = paramNames.toString();
            log.warn("Attemping to set weights for unknown parameters: "
                            + unknownParamNames.substring(1, unknownParamNames.length() - 1));
        }
    }
}
