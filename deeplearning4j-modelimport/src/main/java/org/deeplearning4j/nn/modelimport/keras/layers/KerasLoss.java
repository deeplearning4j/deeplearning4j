package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Map;

/**
 * Builds a DL4J LossLayer from a Keras training loss function.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasLoss extends KerasLayer {

    public static final String KERAS_CLASS_NAME_LOSS = "Loss";

    /**
     * Constructor from layer name and input shape.
     *
     * @param layerName             layer name
     * @param inboundLayerName      name of inbound layer
     * @param kerasLoss             name of Keras loss function
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasLoss(String layerName, String inboundLayerName, String kerasLoss)
            throws UnsupportedKerasConfigurationException {
        this(layerName, inboundLayerName, kerasLoss, true);
    }

    /**
     * Constructor from layer name and input shape.
     *
     * @param layerName                 layer name
     * @param inboundLayerName          name of inbound layer
     * @param kerasLoss                 name of Keras loss function
     * @param enforceTrainingConfig     whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasLoss(String layerName, String inboundLayerName, String kerasLoss, boolean enforceTrainingConfig)
            throws UnsupportedKerasConfigurationException {
        this.className = KERAS_CLASS_NAME_LOSS;
        this.layerName = layerName;
        this.inputShape = null;
        this.dimOrder = DimOrder.NONE;
        this.inboundLayerNames = new ArrayList<String>();
        this.inboundLayerNames.add(inboundLayerName);
        LossFunctions.LossFunction loss;
        try {
            loss = mapLossFunction(kerasLoss);
        } catch (UnsupportedKerasConfigurationException e) {
            if (enforceTrainingConfig)
                throw e;
            log.warn("Unsupported Keras loss function. Replacing with MSE.");
            loss = LossFunctions.LossFunction.SQUARED_LOSS;
        }
        this.layer = new LossLayer.Builder(loss).name(layerName).build();
    }

    private KerasLoss(Map<String,Object> layerConfig) {}
    private KerasLoss(Map<String,Object> layerConfig, boolean enforceTrainingConfig) {}

    /**
     * Get DL4J LossLayer.
     *
     * @return  LossLayer
     */
    public LossLayer getLossLayer() { return (LossLayer)this.layer; }

    /**
     * Get layer output type.
     *
     * @param  inputType    Array of InputTypes
     * @return              output type as InputType
     * @throws InvalidKerasConfigurationException
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException("Keras Loss layer accepts only one input (received " + inputType.length + ")");
        return this.getLossLayer().getOutputType(-1, inputType[0]);
    }
}
