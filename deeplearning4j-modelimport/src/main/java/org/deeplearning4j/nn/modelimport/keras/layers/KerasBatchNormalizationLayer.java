package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

import java.util.Map;

/**
 * Created by davekale on 1/5/17.
 */
@Slf4j
public class KerasBatchNormalizationLayer extends KerasLayer {

    public static final int LAYER_BATCHNORM_MODE_1 = 1;
    public static final int LAYER_BATCHNORM_MODE_2 = 2;
    public static final String LAYER_FIELD_GAMMA_REGULARIZER = "gamma_regularizer";
    public static final String LAYER_FIELD_BETA_REGULARIZER = "beta_regularizer";
    public static final String LAYER_FIELD_MODE = "mode";
    public static final String LAYER_FIELD_AXIS = "axis";
    public static final String LAYER_FIELD_MOMENTUM = "momentum";
    public static final String LAYER_FIELD_EPSILON = "eps";

    public KerasBatchNormalizationLayer(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    public KerasBatchNormalizationLayer(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        getGammaRegularizerFromConfig(layerConfig, enforceTrainingConfig);
        getBetaRegularizerFromConfig(layerConfig, enforceTrainingConfig);
        int batchNormMode = getBatchNormMode(layerConfig, enforceTrainingConfig);
        int batchNormAxis = getBatchNormAxis(layerConfig, enforceTrainingConfig);

        this.dl4jLayer = new BatchNormalization.Builder()
            .name(this.layerName)
            .dropOut(getDropoutFromConfig(layerConfig))
            .minibatch(true)
            .lockGammaBeta(false)
            .eps(getEpsFromConfig(layerConfig))
            .momentum(getMomentumFromConfig(layerConfig))
            .build();
    }

    public BatchNormalization getBatchNormalizationLayer() {
        return (BatchNormalization)this.dl4jLayer;
    }

    protected double getEpsFromConfig(Map<String,Object> layerConfig)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_EPSILON))
            throw new InvalidKerasConfigurationException("TODO");
        return (double)innerConfig.get(LAYER_FIELD_EPSILON);
    }

    protected double getMomentumFromConfig(Map<String,Object> layerConfig)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_MOMENTUM))
            throw new InvalidKerasConfigurationException("TODO");
        return (double)innerConfig.get(LAYER_FIELD_MOMENTUM);
    }

    protected void getGammaRegularizerFromConfig(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.get(LAYER_FIELD_GAMMA_REGULARIZER) != null) {
            if (enforceTrainingConfig)
                throw new UnsupportedKerasConfigurationException("Regularization for BatchNormalization gamma parameter not supported");
            else
                log.warn("Regularization for BatchNormalization gamma parameter not supported...ignoring.");
        }
    }

    protected void getBetaRegularizerFromConfig(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.get(LAYER_FIELD_BETA_REGULARIZER) != null) {
            if (enforceTrainingConfig)
                throw new UnsupportedKerasConfigurationException("Regularization for BatchNormalization beta parameter not supported");
            else
                log.warn("Regularization for BatchNormalization beta parameter not supported...ignoring.");
        }
    }

    protected int getBatchNormMode(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_MODE))
            throw new InvalidKerasConfigurationException("TODO");
        int batchNormMode = (int)layerConfig.get(LAYER_FIELD_MODE);
        switch(batchNormMode) {
            case LAYER_BATCHNORM_MODE_1:
                throw new UnsupportedKerasConfigurationException("Keras BatchNormalization mode " + LAYER_BATCHNORM_MODE_1 + " (sample-wise) not supported");
            case LAYER_BATCHNORM_MODE_2:
                throw new UnsupportedKerasConfigurationException("Keras BatchNormalization (per-batch statistics during testing) " + LAYER_BATCHNORM_MODE_2 + " not supported");
        }
        return batchNormMode;
    }

    protected int getBatchNormAxis(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        return (int)innerConfig.get(LAYER_FIELD_AXIS);
    }
}
