/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.modelimport.keras.layers.recurrent;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.util.MaskZeroLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.core.KerasMasking;
import org.deeplearning4j.nn.modelimport.keras.layers.embeddings.KerasEmbedding;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasConstraintUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.params.LSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils.getIActivationFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils.mapToIActivation;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getHasBiasFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getNOutFromConfig;

/**
 * Imports a Keras LSTM layer as a DL4J LSTM layer.
 *
 * @author dave@skymind.io, Max Pumperla
 */
@Slf4j
@Data
@EqualsAndHashCode(callSuper = false)
public class KerasLSTM extends KerasLayer {

    private final String LSTM_FORGET_BIAS_INIT_ZERO = "zero";
    private final String LSTM_FORGET_BIAS_INIT_ONE = "one";

    private final int NUM_TRAINABLE_PARAMS_KERAS_2 = 3;
    private final int NUM_TRAINABLE_PARAMS = 12;

    private final String KERAS_PARAM_NAME_W_C = "W_c";
    private final String KERAS_PARAM_NAME_W_F = "W_f";
    private final String KERAS_PARAM_NAME_W_I = "W_i";
    private final String KERAS_PARAM_NAME_W_O = "W_o";
    private final String KERAS_PARAM_NAME_U_C = "U_c";
    private final String KERAS_PARAM_NAME_U_F = "U_f";
    private final String KERAS_PARAM_NAME_U_I = "U_i";
    private final String KERAS_PARAM_NAME_U_O = "U_o";
    private final String KERAS_PARAM_NAME_B_C = "b_c";
    private final String KERAS_PARAM_NAME_B_F = "b_f";
    private final String KERAS_PARAM_NAME_B_I = "b_i";
    private final String KERAS_PARAM_NAME_B_O = "b_o";
    private final int NUM_WEIGHTS_IN_KERAS_LSTM = 12;

    protected boolean unroll = false;
    protected boolean returnSequences;

    /**
     * Pass-through constructor from KerasLayer
     *
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasLSTM(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasLSTM(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration.
     * @param enforceTrainingConfig whether to load Keras training configuration
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasLSTM(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, enforceTrainingConfig, Collections.<String, KerasLayer>emptyMap());
    }


    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig    dictionary containing Keras layer configuration.
     * @param previousLayers dictionary containing the previous layers in the topology
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasLSTM(Map<String, Object> layerConfig, Map<String, ? extends KerasLayer> previousLayers)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true, previousLayers);
    }


    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @param previousLayers        - dictionary containing the previous layers in the topology
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasLSTM(Map<String, Object> layerConfig, boolean enforceTrainingConfig,
                     Map<String, ? extends KerasLayer> previousLayers)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        Pair<WeightInit, Distribution> init = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
                enforceTrainingConfig, conf, kerasMajorVersion);
        WeightInit weightInit = init.getFirst();
        Distribution distribution = init.getSecond();

        Pair<WeightInit, Distribution> recurrentInit = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INNER_INIT(),
                enforceTrainingConfig, conf, kerasMajorVersion);
        WeightInit recurrentWeightInit = recurrentInit.getFirst();
        Distribution recurrentDistribution = recurrentInit.getSecond();

        boolean hasBias = getHasBiasFromConfig(layerConfig, conf);

        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        this.returnSequences = (Boolean) innerConfig.get(conf.getLAYER_FIELD_RETURN_SEQUENCES());

        // TODO: support recurrent dropout
        // double recurrentDropout = KerasRnnUtils.getRecurrentDropout(conf, layerConfig);
        this.unroll = KerasRnnUtils.getUnrollRecurrentLayer(conf, layerConfig);

        LayerConstraint biasConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_B_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint weightConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_W_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint recurrentConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_RECURRENT_CONSTRAINT(), conf, kerasMajorVersion);

        Pair<Boolean, Double> maskingConfig = KerasLayerUtils.getMaskingConfiguration(inboundLayerNames, previousLayers);

        LSTM.Builder builder = new LSTM.Builder()
                .gateActivationFunction(getGateActivationFromConfig(layerConfig))
                .forgetGateBiasInit(getForgetBiasInitFromConfig(layerConfig, enforceTrainingConfig))
                .name(this.layerName)
                .nOut(getNOutFromConfig(layerConfig, conf))
                .dropOut(this.dropout)
                .activation(getIActivationFromConfig(layerConfig, conf))
                .weightInit(weightInit)
                .weightInitRecurrent(recurrentWeightInit)
                .biasInit(0.0) // TODO: this is incorrect
                .l1(this.weightL1Regularization)
                .l2(this.weightL2Regularization);
        if (distribution != null)
            builder.dist(distribution);
        if (recurrentDistribution != null)
            builder.weightInitRecurrent(recurrentDistribution);
        if (biasConstraint != null)
            builder.constrainBias(biasConstraint);
        if (weightConstraint != null)
            builder.constrainInputWeights(weightConstraint);
        if (recurrentConstraint != null)
            builder.constrainRecurrent(recurrentConstraint);

        this.layer = builder.build();
        if (!returnSequences) {
            this.layer = new LastTimeStep(this.layer);
        }
        if (maskingConfig.getFirst()) {
            this.layer = new MaskZeroLayer(this.layer, maskingConfig.getSecond());
        }
    }

    /**
     * Get DL4J Layer. If returnSequences is true, this can be casted to an "LSTM" layer, otherwise it can be casted
     * to a "LastTimeStep" layer.
     *
     * @return LSTM Layer
     */
    public Layer getLSTMLayer() {
        return layer;
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1 && inputType.length != 3)
            throw new InvalidKerasConfigurationException("Keras LSTM layer accepts only one single input" +
                            "or three (input to LSTM and two states tensors, but " +
                            "received " + inputType.length + ".");
        InputPreProcessor preProcessor = getInputPreprocessor(inputType);
        if (preProcessor != null) {
            if (returnSequences) {
                return preProcessor.getOutputType(inputType[0]);
            } else {
                return this.getLSTMLayer().getOutputType(-1, preProcessor.getOutputType(inputType[0]));
            }
        } else
            return this.getLSTMLayer().getOutputType(-1, inputType[0]);

    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return number of trainable parameters (12)
     */
    @Override
    public int getNumParams() {
        return kerasMajorVersion == 2 ? NUM_TRAINABLE_PARAMS_KERAS_2 : NUM_TRAINABLE_PARAMS;
    }

    /**
     * Gets appropriate DL4J InputPreProcessor for given InputTypes.
     *
     * @param inputType Array of InputTypes
     * @return DL4J InputPreProcessor
     * @throws InvalidKerasConfigurationException Invalid Keras configuration exception
     * @see org.deeplearning4j.nn.conf.InputPreProcessor
     */
    @Override
    public InputPreProcessor getInputPreprocessor(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1 && inputType.length != 3)
            throw new InvalidKerasConfigurationException("Keras LSTM layer accepts only one single input" +
                    "or three (input to LSTM and two states tensors, but " +
                    "received " + inputType.length + ".");
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType[0], layerName);
    }

    /**
     * Set weights for layer.
     *
     * @param weights LSTM layer weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<>();
        /* Keras stores LSTM parameters in distinct arrays (e.g., the recurrent weights
         * are stored in four matrices: U_c, U_f, U_i, U_o) while DL4J stores them
         * concatenated into one matrix (e.g., U = [ U_c U_f U_o U_i ]). Thus we have
         * to map the Keras weight matrix to its corresponding DL4J weight submatrix.
         */
        INDArray W_i;
        INDArray W_f;
        INDArray W_c;
        INDArray W_o;
        INDArray U_i;
        INDArray U_f;
        INDArray U_c;
        INDArray U_o;
        INDArray b_i;
        INDArray b_f;
        INDArray b_c;
        INDArray b_o;


        if (kerasMajorVersion == 2) {
            INDArray W;
            if (weights.containsKey(conf.getKERAS_PARAM_NAME_W()))
                W = weights.get(conf.getKERAS_PARAM_NAME_W());
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + conf.getKERAS_PARAM_NAME_W());
            INDArray U;
            if (weights.containsKey(conf.getKERAS_PARAM_NAME_RW()))
                U = weights.get(conf.getKERAS_PARAM_NAME_RW());
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + conf.getKERAS_PARAM_NAME_RW());
            INDArray b;
            if (weights.containsKey(conf.getKERAS_PARAM_NAME_B()))
                b = weights.get(conf.getKERAS_PARAM_NAME_B());
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + conf.getKERAS_PARAM_NAME_B());

            val sliceInterval = b.length() / 4;
            W_i = W.get(NDArrayIndex.all(), NDArrayIndex.interval(0, sliceInterval));
            W_f = W.get(NDArrayIndex.all(), NDArrayIndex.interval(sliceInterval, 2 * sliceInterval));
            W_c = W.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * sliceInterval, 3 * sliceInterval));
            W_o = W.get(NDArrayIndex.all(), NDArrayIndex.interval(3 * sliceInterval, 4 * sliceInterval));
            U_i = U.get(NDArrayIndex.all(), NDArrayIndex.interval(0, sliceInterval));
            U_f = U.get(NDArrayIndex.all(), NDArrayIndex.interval(sliceInterval, 2 * sliceInterval));
            U_c = U.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * sliceInterval, 3 * sliceInterval));
            U_o = U.get(NDArrayIndex.all(), NDArrayIndex.interval(3 * sliceInterval, 4 * sliceInterval));
            b_i = b.get(NDArrayIndex.all(), NDArrayIndex.interval(0, sliceInterval));
            b_f = b.get(NDArrayIndex.all(), NDArrayIndex.interval(sliceInterval, 2 * sliceInterval));
            b_c = b.get(NDArrayIndex.all(), NDArrayIndex.interval(2 * sliceInterval, 3 * sliceInterval));
            b_o = b.get(NDArrayIndex.all(), NDArrayIndex.interval(3 * sliceInterval, 4 * sliceInterval));
        } else {
            if (weights.containsKey(KERAS_PARAM_NAME_W_C))
                W_c = weights.get(KERAS_PARAM_NAME_W_C);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_W_C);
            if (weights.containsKey(KERAS_PARAM_NAME_W_F))
                W_f = weights.get(KERAS_PARAM_NAME_W_F);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_W_F);
            if (weights.containsKey(KERAS_PARAM_NAME_W_O))
                W_o = weights.get(KERAS_PARAM_NAME_W_O);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_W_O);
            if (weights.containsKey(KERAS_PARAM_NAME_W_I))
                W_i = weights.get(KERAS_PARAM_NAME_W_I);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_W_I);
            if (weights.containsKey(KERAS_PARAM_NAME_U_C))
                U_c = weights.get(KERAS_PARAM_NAME_U_C);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_U_C);
            if (weights.containsKey(KERAS_PARAM_NAME_U_F))
                U_f = weights.get(KERAS_PARAM_NAME_U_F);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_U_F);
            if (weights.containsKey(KERAS_PARAM_NAME_U_O))
                U_o = weights.get(KERAS_PARAM_NAME_U_O);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_U_O);
            if (weights.containsKey(KERAS_PARAM_NAME_U_I))
                U_i = weights.get(KERAS_PARAM_NAME_U_I);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_U_I);
            if (weights.containsKey(KERAS_PARAM_NAME_B_C))
                b_c = weights.get(KERAS_PARAM_NAME_B_C);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_B_C);
            if (weights.containsKey(KERAS_PARAM_NAME_B_F))
                b_f = weights.get(KERAS_PARAM_NAME_B_F);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_B_F);
            if (weights.containsKey(KERAS_PARAM_NAME_B_O))
                b_o = weights.get(KERAS_PARAM_NAME_B_O);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_B_O);
            if (weights.containsKey(KERAS_PARAM_NAME_B_I))
                b_i = weights.get(KERAS_PARAM_NAME_B_I);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_B_I);

        }

        // Need to convert from IFCO to CFOI order
        int wCols = W_c.columns();
        int wRows = W_c.rows();

        INDArray W = Nd4j.zeros(wRows, 4 * wCols);
        W.put(new INDArrayIndex[]{NDArrayIndex.interval(0, wRows), NDArrayIndex.interval(0, wCols)}, W_c);
        W.put(new INDArrayIndex[]{NDArrayIndex.interval(0, wRows), NDArrayIndex.interval(wCols, 2 * wCols)}, W_f);
        W.put(new INDArrayIndex[]{NDArrayIndex.interval(0, wRows), NDArrayIndex.interval(2 * wCols, 3 * wCols)}, W_o);
        W.put(new INDArrayIndex[]{NDArrayIndex.interval(0, wRows), NDArrayIndex.interval(3 * wCols, 4 * wCols)}, W_i);
        this.weights.put(LSTMParamInitializer.INPUT_WEIGHT_KEY, W);

        int uCols = U_c.columns();
        int uRows = U_c.rows();
        INDArray U = Nd4j.zeros(uRows, 4 * uCols);
        U.put(new INDArrayIndex[]{NDArrayIndex.interval(0, U.rows()), NDArrayIndex.interval(0, uCols)}, U_c);
        U.put(new INDArrayIndex[]{NDArrayIndex.interval(0, U.rows()), NDArrayIndex.interval(uCols, 2 * uCols)}, U_f);
        U.put(new INDArrayIndex[]{NDArrayIndex.interval(0, U.rows()), NDArrayIndex.interval(2 * uCols, 3 * uCols)}, U_o);
        U.put(new INDArrayIndex[]{NDArrayIndex.interval(0, U.rows()), NDArrayIndex.interval(3 * uCols, 4 * uCols)}, U_i);
        this.weights.put(LSTMParamInitializer.RECURRENT_WEIGHT_KEY, U);


        int bCols = b_c.columns();
        int bRows = b_c.rows();
        INDArray b = Nd4j.zeros(bRows, 4 * bCols);
        b.put(new INDArrayIndex[]{NDArrayIndex.interval(0, b.rows()), NDArrayIndex.interval(0, bCols)}, b_c);
        b.put(new INDArrayIndex[]{NDArrayIndex.interval(0, b.rows()), NDArrayIndex.interval(bCols, 2 * bCols)}, b_f);
        b.put(new INDArrayIndex[]{NDArrayIndex.interval(0, b.rows()), NDArrayIndex.interval(2 * bCols, 3 * bCols)}, b_o);
        b.put(new INDArrayIndex[]{NDArrayIndex.interval(0, b.rows()), NDArrayIndex.interval(3 * bCols, 4 * bCols)}, b_i);
        this.weights.put(LSTMParamInitializer.BIAS_KEY, b);

        if (weights.size() > NUM_WEIGHTS_IN_KERAS_LSTM) {
            Set<String> paramNames = weights.keySet();
            paramNames.remove(KERAS_PARAM_NAME_W_C);
            paramNames.remove(KERAS_PARAM_NAME_W_F);
            paramNames.remove(KERAS_PARAM_NAME_W_I);
            paramNames.remove(KERAS_PARAM_NAME_W_O);
            paramNames.remove(KERAS_PARAM_NAME_U_C);
            paramNames.remove(KERAS_PARAM_NAME_U_F);
            paramNames.remove(KERAS_PARAM_NAME_U_I);
            paramNames.remove(KERAS_PARAM_NAME_U_O);
            paramNames.remove(KERAS_PARAM_NAME_B_C);
            paramNames.remove(KERAS_PARAM_NAME_B_F);
            paramNames.remove(KERAS_PARAM_NAME_B_I);
            paramNames.remove(KERAS_PARAM_NAME_B_O);
            String unknownParamNames = paramNames.toString();
            log.warn("Attemping to set weights for unknown parameters: "
                    + unknownParamNames.substring(1, unknownParamNames.length() - 1));
        }
    }

    /**
     * Get whether LSTM layer should be unrolled (for truncated BPTT).
     *
     * @return whether to unroll the LSTM
     */
    public boolean getUnroll() {
        return this.unroll;
    }


    /**
     * Get LSTM gate activation function from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return LSTM inner activation function
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    public IActivation getGateActivationFromConfig(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(conf.getLAYER_FIELD_INNER_ACTIVATION()))
            throw new InvalidKerasConfigurationException(
                    "Keras LSTM layer config missing " + conf.getLAYER_FIELD_INNER_ACTIVATION() + " field");
        return mapToIActivation((String) innerConfig.get(conf.getLAYER_FIELD_INNER_ACTIVATION()), conf);
    }

    /**
     * Get LSTM forget gate bias initialization from Keras layer configuration.
     *
     * @param layerConfig dictionary containing Keras layer configuration
     * @return LSTM forget gate bias init
     * @throws InvalidKerasConfigurationException Unsupported Keras config
     */
    public double getForgetBiasInitFromConfig(Map<String, Object> layerConfig, boolean train)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        String kerasForgetBiasInit;
        if (innerConfig.containsKey(conf.getLAYER_FIELD_UNIT_FORGET_BIAS())) {
            kerasForgetBiasInit = LSTM_FORGET_BIAS_INIT_ONE;
        } else if (!innerConfig.containsKey(conf.getLAYER_FIELD_FORGET_BIAS_INIT())) {
            throw new InvalidKerasConfigurationException(
                    "Keras LSTM layer config missing " + conf.getLAYER_FIELD_FORGET_BIAS_INIT() + " field");
        } else {
            kerasForgetBiasInit = (String) innerConfig.get(conf.getLAYER_FIELD_FORGET_BIAS_INIT());
        }
        double init;
        switch (kerasForgetBiasInit) {
            case LSTM_FORGET_BIAS_INIT_ZERO:
                init = 0.0;
                break;
            case LSTM_FORGET_BIAS_INIT_ONE:
                init = 1.0;
                break;
            default:
                if (train)
                    throw new UnsupportedKerasConfigurationException(
                            "Unsupported LSTM forget gate bias initialization: " + kerasForgetBiasInit);
                else {
                    init = 1.0;
                    log.warn("Unsupported LSTM forget gate bias initialization: " + kerasForgetBiasInit
                            + " (using 1 instead)");
                }
                break;
        }
        return init;
    }
}
