package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

import java.util.Map;

/**
 * Created by davekale on 1/5/17.
 */
@Slf4j
public class KerasDenseLayer extends KerasLayer {

    public KerasDenseLayer(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    public KerasDenseLayer(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        this.dl4jLayer = new DenseLayer.Builder()
            .name(this.layerName)
            .nOut(getNOutFromConfig(layerConfig))
            .dropOut(getDropoutFromConfig(layerConfig))
            .activation(getActivationFromConfig(layerConfig))
            .weightInit(getWeightInitFromConfig(layerConfig, enforceTrainingConfig))
            .biasInit(0.0)
            .l1(getWeightL1RegularizationFromConfig(layerConfig, enforceTrainingConfig))
            .l2(getWeightL2RegularizationFromConfig(layerConfig, enforceTrainingConfig))
            .build();
    }

    public DenseLayer getDenseLayer() {
        return (DenseLayer)this.dl4jLayer;
    }
}
