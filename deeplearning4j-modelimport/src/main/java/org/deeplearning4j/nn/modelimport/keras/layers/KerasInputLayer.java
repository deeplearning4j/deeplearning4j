package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;

/**
 * Created by davekale on 1/5/17.
 */
@Slf4j
public class KerasInputLayer extends KerasLayer {

    public KerasInputLayer(String layerName, int[] inputShape)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerName, inputShape, true);
    }

    public KerasInputLayer(String layerName, int[] inputShape, boolean enforceTrainingConfig)
            throws UnsupportedKerasConfigurationException {
        this.className = LAYER_CLASS_NAME_INPUT;
        this.layerName = layerName;
        this.inputShape = inputShape;
        this.dl4jLayer = null;
    }
}
