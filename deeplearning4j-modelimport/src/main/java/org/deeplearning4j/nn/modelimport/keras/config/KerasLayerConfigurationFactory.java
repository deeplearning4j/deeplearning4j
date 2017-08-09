package org.deeplearning4j.nn.modelimport.keras.config;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

@Slf4j
public class KerasLayerConfigurationFactory {

    public KerasLayerConfigurationFactory() {}

    public static KerasLayerConfiguration get(Integer kerasMajorVersion) throws UnsupportedKerasConfigurationException {
        if (kerasMajorVersion != 1 && kerasMajorVersion != 2)
            throw new UnsupportedKerasConfigurationException(
                    "Keras major version has to be either 1 or 2 (" + kerasMajorVersion + " provided)");
        else if (kerasMajorVersion == 1)
            return new Keras1LayerConfiguration();
        else
            return new Keras2LayerConfiguration();
    }
}
