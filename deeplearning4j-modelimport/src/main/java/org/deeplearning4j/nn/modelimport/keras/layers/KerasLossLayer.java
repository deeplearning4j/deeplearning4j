package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by davekale on 1/5/17.
 */
@Slf4j
public class KerasLossLayer extends KerasLayer {

    public static final String KERAS_CLASS_NAME_LOSS = "Loss";

    public KerasLossLayer(String layerName, String kerasLoss)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerName, kerasLoss, true);
    }

    public KerasLossLayer(String layerName, String kerasLoss, boolean enforceTrainingConfig)
            throws UnsupportedKerasConfigurationException {
        this.className = KERAS_CLASS_NAME_LOSS;
        LossFunctions.LossFunction loss;
        try {
            loss = mapLossFunction(kerasLoss);
        } catch (UnsupportedKerasConfigurationException e) {
            if (enforceTrainingConfig)
                throw e;
            log.warn("Unsupported Keras loss function. Replacing with MSE.");
            loss = LossFunctions.LossFunction.SQUARED_LOSS;
        }
        this.dl4jLayer = new LossLayer.Builder(loss).name(layerName).build();
    }

    public LossLayer getLossLayer() {
        return (LossLayer)this.dl4jLayer;
    }
}
