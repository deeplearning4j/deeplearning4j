package org.deeplearning4j.util;

import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.util.HashSet;
import java.util.Set;

/**
 * Utility methods for output layer configuration/validation
 *
 * @author Alex Black
 */
public class OutputLayerUtil {

    private OutputLayerUtil(){ }

    private static final Set<Class<?>> OUTSIDE_ZERO_ONE_RANGE = new HashSet<>();
    static {
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationCube.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationELU.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationHardTanH.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationIdentity.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationLReLU.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationPReLU.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationRationalTanh.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationReLU.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationReLU6.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationRReLU.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationSELU.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationSoftPlus.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationSoftSign.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationSwish.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationTanH.class);
        OUTSIDE_ZERO_ONE_RANGE.add(ActivationThresholdedReLU.class);
    }

    private static final String COMMON_MSG = "This configuration validation check can be disabled for MultiLayerConfiguration" +
            " and ComputationGraphConfiguration using validateOutputConfig(false), however this is not recommended.";


    /**
     * Validate the output layer (or loss layer) configuration, to detect invalid consfiugrations. A DL4JInvalidConfigException
     * will be thrown for invalid configurations (like softmax + nOut=1).<br>
     *
     * If the specified layer is not an output layer, this is a no-op
     * @param layerName Name of the layer
     * @param layer         Layer
     */
    public static void validateOutputLayer(String layerName, Layer layer){
        IActivation activation;
        ILossFunction loss;
        long nOut;
        boolean isLossLayer = false;
        if (layer instanceof BaseOutputLayer) {
            activation = ((BaseOutputLayer) layer).getActivationFn();
            loss = ((BaseOutputLayer) layer).getLossFn();
            nOut = ((BaseOutputLayer) layer).getNOut();
        } else if (layer instanceof LossLayer) {
            activation = ((LossLayer) layer).getActivationFn();
            loss = ((LossLayer) layer).getLossFn();
            nOut = ((LossLayer) layer).getNOut();
            isLossLayer = true;
        } else if (layer instanceof RnnLossLayer) {
            activation = ((RnnLossLayer) layer).getActivationFn();
            loss = ((RnnLossLayer) layer).getLossFn();
            nOut = ((RnnLossLayer) layer).getNOut();
            isLossLayer = true;
        } else if (layer instanceof CnnLossLayer) {
            activation = ((CnnLossLayer) layer).getActivationFn();
            loss = ((CnnLossLayer) layer).getLossFn();
            nOut = ((CnnLossLayer) layer).getNOut();
            isLossLayer = true;
        } else {
            //Not an output layer
            return;
        }
        OutputLayerUtil.validateOutputLayerConfiguration(layerName, nOut, isLossLayer, activation, loss);
    }

    /**
     * Validate the output layer (or loss layer) configuration, to detect invalid consfiugrations. A DL4JInvalidConfigException
     * will be thrown for invalid configurations (like softmax + nOut=1).<br>
     * <p>
     * If the specified layer is not an output layer, this is a no-op
     *
     * @param layerName    Name of the layer
     * @param nOut         Number of outputs for the layer
     * @param isLossLayer  Should be true for loss layers (no params), false for output layers
     * @param activation   Activation function
     * @param lossFunction Loss function
     */
    public static void validateOutputLayerConfiguration(String layerName, long nOut, boolean isLossLayer, IActivation activation, ILossFunction lossFunction){
        //nOut = 1 + softmax
        if(!isLossLayer && nOut == 1 && activation instanceof ActivationSoftmax){   //May not have valid nOut for LossLayer
            throw new DL4JInvalidConfigException("Invalid output layer configuration for layer \"" + layerName + "\": Softmax + nOut=1 networks " +
                    "are not supported. Softmax cannot be used with nOut=1 as the output will always be exactly 1" +
                    "regardless of the input. " + COMMON_MSG);
        }

        //loss function required probability, but activation is outside 0-1 range
        if(lossFunctionExpectsProbability(lossFunction) && activationExceedsZeroOneRange(activation, isLossLayer)){
            throw new DL4JInvalidConfigException("Invalid output layer configuration for layer \"" + layerName + "\": loss function " + lossFunction +
                    " expects activations to be in the range 0 to 1 (probabilities) but activation function " + activation +
                    " does not bound values to this 0 to 1 range. This indicates a likely invalid network configuration. " + COMMON_MSG);
        }

        //Common mistake: softmax + xent
        if(activation instanceof ActivationSoftmax && lossFunction instanceof LossBinaryXENT){
            throw new DL4JInvalidConfigException("Invalid output layer configuration for layer \"" + layerName + "\": softmax activation function in combination " +
                    "with LossBinaryXENT (binary cross entropy loss function). For multi-class classification, use softmax + " +
                    "MCXENT (multi-class cross entropy); for binary multi-label classification, use sigmoid + XENT. " + COMMON_MSG);
        }

        //Common mistake: sigmoid + mcxent
        if(activation instanceof ActivationSigmoid && lossFunction instanceof LossMCXENT){
            throw new DL4JInvalidConfigException("Invalid output layer configuration for layer \"" + layerName + "\": sigmoid activation function in combination " +
                    "with LossMCXENT (multi-class cross entropy loss function). For multi-class classification, use softmax + " +
                    "MCXENT (multi-class cross entropy); for binary multi-label classification, use sigmoid + XENT. " + COMMON_MSG);
        }
    }

    public static boolean lossFunctionExpectsProbability(ILossFunction lf) {
        return lf instanceof LossMCXENT || lf instanceof LossBinaryXENT;
    }

    public static boolean activationExceedsZeroOneRange(IActivation activation, boolean isLossLayer){

        if(OUTSIDE_ZERO_ONE_RANGE.contains(activation.getClass())){
            if(isLossLayer && activation instanceof ActivationIdentity){
                //Note: we're intentionally excluding identity here, for situations like dense(softmax) -> loss(identity)
                //However, we might miss a few invalid configs like dense(relu) -> loss(identity)
                return false;
            }
            return true;
        }
        return false;
    }
}
