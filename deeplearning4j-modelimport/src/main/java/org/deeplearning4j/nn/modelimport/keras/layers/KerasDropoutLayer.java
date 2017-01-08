package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

import java.util.Map;

/**
 * Created by davekale on 1/5/17.
 */
@Slf4j
public class KerasDropoutLayer extends KerasLayer {

    public KerasDropoutLayer(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    public KerasDropoutLayer(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        this.dl4jLayer = new DropoutLayer.Builder()
            .name(this.layerName)
            .dropOut(getDropoutFromConfig(layerConfig))
            .build();
    }

    public DropoutLayer getDropoutLayer() {
        return (DropoutLayer)this.dl4jLayer;
    }
}
