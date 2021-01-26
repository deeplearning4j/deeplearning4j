/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.util;

import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer;
import org.nd4j.evaluation.IEvaluation;
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

    private static final String COMMON_MSG = "\nThis configuration validation check can be disabled for MultiLayerConfiguration" +
            " and ComputationGraphConfiguration using validateOutputLayerConfig(false), however this is not recommended.";


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
        if (layer instanceof BaseOutputLayer && !(layer instanceof OCNNOutputLayer)) {
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
                    "are not supported. Softmax cannot be used with nOut=1 as the output will always be exactly 1.0 " +
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
        //Note LossNegativeLogLikelihood extends LossMCXENT
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

    /**
     * Validates if the output layer configuration is valid for classifier evaluation.
     * This is used to try and catch invalid evaluation - i.e., trying to use classifier evaluation on a regression model.
     * This method won't catch all possible invalid cases, but should catch some common problems.
     *
     * @param outputLayer          Output layer
     * @param classifierEval       Class for the classifier evaluation
     */
    public static void validateOutputLayerForClassifierEvaluation(Layer outputLayer, Class<? extends IEvaluation> classifierEval){
        if(outputLayer instanceof Yolo2OutputLayer){
            throw new IllegalStateException("Classifier evaluation using " + classifierEval.getSimpleName() + " class cannot be applied for object" +
                    " detection evaluation using Yolo2OutputLayer: " + classifierEval.getSimpleName() + "  class is for classifier evaluation only.");
        }

        //Check that the activation function provides probabilities. This can't catch everything, but should catch a few
        // of the common mistakes users make
        if(outputLayer instanceof BaseLayer){
            BaseLayer bl = (BaseLayer)outputLayer;
            boolean isOutputLayer = outputLayer instanceof OutputLayer || outputLayer instanceof RnnOutputLayer || outputLayer instanceof CenterLossOutputLayer;

            if(activationExceedsZeroOneRange(bl.getActivationFn(), !isOutputLayer)){
                throw new IllegalStateException("Classifier evaluation using " + classifierEval.getSimpleName() + " class cannot be applied to output" +
                        " layers with activation functions that are not probabilities (in range 0 to 1). Output layer type: " +
                        outputLayer.getClass().getSimpleName() + " has activation function " + bl.getActivationFn().getClass().getSimpleName() +
                        ". This check can be disabled using MultiLayerNetwork.getLayerWiseConfigurations().setValidateOutputLayerConfig(false)" +
                        " or ComputationGraph.getConfiguration().setValidateOutputLayerConfig(false)");
            }
        }
    }
}
