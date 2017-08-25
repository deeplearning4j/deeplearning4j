package org.deeplearning4j.nn.modelimport.keras.utils;

import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

public class KerasRegularizerUtils {

    /**
     * Get L1 weight regularization (if any) from Keras weight regularization configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration     Map containing Keras weight reguarlization configuration
     * @return L1 regularization strength (0.0 if none)
     */
    public static double getWeightL1RegularizationFromConfig(Map<String, Object> layerConfig, boolean willBeTrained,
                                                      KerasLayerConfiguration conf)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (innerConfig.containsKey(conf.getLAYER_FIELD_W_REGULARIZER())) {
            Map<String, Object> regularizerConfig =
                    (Map<String, Object>) innerConfig.get(conf.getLAYER_FIELD_W_REGULARIZER());
            if (regularizerConfig != null && regularizerConfig.containsKey(conf.getREGULARIZATION_TYPE_L1()))
                return (double) regularizerConfig.get(conf.getREGULARIZATION_TYPE_L1());
        }
        return 0.0;
    }

    /**
     * Get L2 weight regularization (if any) from Keras weight regularization configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return L1 regularization strength (0.0 if none)
     */
    public static double getWeightL2RegularizationFromConfig(Map<String, Object> layerConfig, boolean willBeTrained,
                                                             KerasLayerConfiguration conf)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (innerConfig.containsKey(conf.getLAYER_FIELD_W_REGULARIZER())) {
            Map<String, Object> regularizerConfig =
                    (Map<String, Object>) innerConfig.get(conf.getLAYER_FIELD_W_REGULARIZER());
            if (regularizerConfig != null && regularizerConfig.containsKey(conf.getREGULARIZATION_TYPE_L2()))
                return (double) regularizerConfig.get(conf.getREGULARIZATION_TYPE_L2());
        }
        return 0.0;
    }
}
