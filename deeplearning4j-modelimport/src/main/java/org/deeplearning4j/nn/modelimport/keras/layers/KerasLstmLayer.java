package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;

import java.util.Map;

/**
 * Created by davekale on 1/5/17.
 */
@Slf4j
public class KerasLstmLayer extends KerasLayer {

    public static final String LAYER_FIELD_INNER_INIT = "inner_init";
    public static final String LAYER_FIELD_INNER_ACTIVATION = "inner_activation";
    public static final String LAYER_FIELD_FORGET_BIAS_INIT = "forget_bias_init";
    public static final String LAYER_FIELD_DROPOUT_U = "dropout_U";
    public static final String LSTM_FORGET_BIAS_INIT_ZERO = "zero";
    public static final String LSTM_FORGET_BIAS_INIT_ONE = "one";

    public KerasLstmLayer(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    public KerasLstmLayer(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        WeightInit weightInit = getWeightInitFromConfig(layerConfig, enforceTrainingConfig);
        WeightInit recurrentWeightInit = getRecurrentWeightInitFromConfig(layerConfig, enforceTrainingConfig);
        if (weightInit != recurrentWeightInit)
            if (enforceTrainingConfig)
                throw new UnsupportedKerasConfigurationException("Specifying different initialization for recurrent weights not supported.");
            else
                log.warn("Specifying different initialization for recurrent weights not supported.");
        getRecurrentDropout(layerConfig);
        this.dl4jLayer = new GravesLSTM.Builder()
            .gateActivationFunction(getGateActivationFromConfig(layerConfig))
            .forgetGateBiasInit(getForgetBiasInitFromConfig(layerConfig, enforceTrainingConfig))
            .name(this.layerName)
            .nOut(getNOutFromConfig(layerConfig))
            .dropOut(getDropoutFromConfig(layerConfig))
            .activation(getActivationFromConfig(layerConfig))
            .weightInit(weightInit)
            .biasInit(0.0)
            .l1(getWeightL1RegularizationFromConfig(layerConfig, enforceTrainingConfig))
            .l2(getWeightL2RegularizationFromConfig(layerConfig, enforceTrainingConfig))
            .build();
    }

    public GravesLSTM getGravesLSTMLayer() {
        return (GravesLSTM)this.dl4jLayer;
    }

    public static WeightInit getRecurrentWeightInitFromConfig(Map<String,Object> layerConfig, boolean train)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_INNER_INIT))
            throw new InvalidKerasConfigurationException("TODO");
        String kerasInit = (String)innerConfig.get(LAYER_FIELD_INNER_INIT);
        WeightInit init;
        try {
            init = mapWeightInitialization(kerasInit);
        } catch (UnsupportedKerasConfigurationException e) {
            if (train)
                throw e;
            else {
                init = WeightInit.XAVIER;
                log.warn("Unknown weight initializer " + kerasInit + " (Using XAVIER instead).");
            }
        }
        return init;
    }

    public static double getRecurrentDropout(Map<String,Object> layerConfig)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        /* NOTE: Keras "dropout" parameter determines dropout probability,
         * while DL4J "dropout" parameter determines retention probability.
         */
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        double dropout = 1.0;
        if (innerConfig.containsKey(LAYER_FIELD_DROPOUT_U))
            dropout = 1.0-(double)innerConfig.get(LAYER_FIELD_DROPOUT_U);
        if (dropout > 0.0)
            throw new UnsupportedKerasConfigurationException("Dropout > 0 on LSTM recurrent connections not supported.");
        return dropout;
    }

    public static IActivation getGateActivationFromConfig(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_INNER_ACTIVATION))
            throw new InvalidKerasConfigurationException("TODO");
        return mapActivation((String)innerConfig.get(LAYER_FIELD_INNER_ACTIVATION));
    }

    public static double getForgetBiasInitFromConfig(Map<String,Object> layerConfig, boolean train)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_FORGET_BIAS_INIT))
            throw new InvalidKerasConfigurationException("TODO");
        String kerasForgetBiasInit = (String)innerConfig.get(LAYER_FIELD_FORGET_BIAS_INIT);
        double init = 0;
        switch (kerasForgetBiasInit) {
            case LSTM_FORGET_BIAS_INIT_ZERO:
                init = 0.0;
                break;
            case LSTM_FORGET_BIAS_INIT_ONE:
                init = 1.0;
                break;
            default:
                if (train)
                    throw new UnsupportedKerasConfigurationException("Unsupported LSTM forget gate bias initialization: " + kerasForgetBiasInit);
                else {
                    init = 1.0;
                    log.warn("Unsupported LSTM forget gate bias initialization: " + kerasForgetBiasInit + " (using 1 instead)");
                }
                break;
        }
        return init;
    }
}
