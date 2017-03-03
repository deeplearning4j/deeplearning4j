package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Imports a Keras LSTM layer as a DL4J GravesLSTM layer.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasLstm extends KerasLayer {

    /* Keras layer configuration fields. */
    public static final String LAYER_FIELD_INNER_INIT = "inner_init";
    public static final String LAYER_FIELD_INNER_ACTIVATION = "inner_activation";
    public static final String LAYER_FIELD_FORGET_BIAS_INIT = "forget_bias_init";
    public static final String LAYER_FIELD_DROPOUT_U = "dropout_U";
    public static final String LAYER_FIELD_UNROLL = "unroll";
    public static final String LSTM_FORGET_BIAS_INIT_ZERO = "zero";
    public static final String LSTM_FORGET_BIAS_INIT_ONE = "one";

    /* Keras layer parameter names. */
    public static final int NUM_TRAINABLE_PARAMS = 12;
    public static final String KERAS_PARAM_NAME_W_C = "W_c";
    public static final String KERAS_PARAM_NAME_W_F = "W_f";
    public static final String KERAS_PARAM_NAME_W_I = "W_i";
    public static final String KERAS_PARAM_NAME_W_O = "W_o";
    public static final String KERAS_PARAM_NAME_U_C = "U_c";
    public static final String KERAS_PARAM_NAME_U_F = "U_f";
    public static final String KERAS_PARAM_NAME_U_I = "U_i";
    public static final String KERAS_PARAM_NAME_U_O = "U_o";
    public static final String KERAS_PARAM_NAME_B_C = "b_c";
    public static final String KERAS_PARAM_NAME_B_F = "b_f";
    public static final String KERAS_PARAM_NAME_B_I = "b_i";
    public static final String KERAS_PARAM_NAME_B_O = "b_o";

    public static final int NUM_WEIGHTS_IN_KERAS_LSTM = 12;

    protected boolean unroll = false; // whether to unroll LSTM

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig   dictionary containing Keras layer configuration.
     *
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasLstm(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig               dictionary containing Keras layer configuration
     * @param enforceTrainingConfig     whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasLstm(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);
        WeightInit weightInit = getWeightInitFromConfig(layerConfig, enforceTrainingConfig);
        WeightInit recurrentWeightInit = getRecurrentWeightInitFromConfig(layerConfig, enforceTrainingConfig);
        if (weightInit != recurrentWeightInit)
            if (enforceTrainingConfig)
                throw new UnsupportedKerasConfigurationException(
                                "Specifying different initialization for recurrent weights not supported.");
            else
                log.warn("Specifying different initialization for recurrent weights not supported.");
        getRecurrentDropout(layerConfig);
        this.unroll = getUnrollRecurrentLayer(layerConfig);
        this.layer = new GravesLSTM.Builder().gateActivationFunction(getGateActivationFromConfig(layerConfig))
                        .forgetGateBiasInit(getForgetBiasInitFromConfig(layerConfig, enforceTrainingConfig))
                        .name(this.layerName).nOut(getNOutFromConfig(layerConfig)).dropOut(this.dropout)
                        .activation(getActivationFromConfig(layerConfig)).weightInit(weightInit).biasInit(0.0)
                        .l1(this.weightL1Regularization).l2(this.weightL2Regularization).build();
    }

    /**
     * Get DL4J GravesLSTMLayer.
     *
     * @return  GravesLSTMLayer
     */
    public GravesLSTM getGravesLSTMLayer() {
        return (GravesLSTM) this.layer;
    }

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
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer accepts only one input (received " + inputType.length + ")");
        return this.getGravesLSTMLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return          number of trainable parameters (12)
     */
    @Override
    public int getNumParams() {
        return NUM_TRAINABLE_PARAMS;
    }

    /**
     * Set weights for layer.
     *
     * @param weights
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<String, INDArray>();
        /* Keras stores LSTM parameters in distinct arrays (e.g., the recurrent weights
         * are stored in four matrices: U_c, U_f, U_i, U_o) while DL4J stores them
         * concatenated into one matrix (e.g., U = [ U_c U_f U_o U_i ]). Thus we have
         * to map the Keras weight matrix to its corresponding DL4J weight submatrix.
         */
        INDArray W_c;
        if (weights.containsKey(KERAS_PARAM_NAME_W_C))
            W_c = weights.get(KERAS_PARAM_NAME_W_C);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_W_C);
        INDArray W_f;
        if (weights.containsKey(KERAS_PARAM_NAME_W_F))
            W_f = weights.get(KERAS_PARAM_NAME_W_F);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_W_F);
        INDArray W_o;
        if (weights.containsKey(KERAS_PARAM_NAME_W_O))
            W_o = weights.get(KERAS_PARAM_NAME_W_O);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_W_O);
        INDArray W_i;
        if (weights.containsKey(KERAS_PARAM_NAME_W_I))
            W_i = weights.get(KERAS_PARAM_NAME_W_I);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_W_I);
        INDArray W = Nd4j.zeros(W_c.rows(), W_c.columns() + W_f.columns() + W_o.columns() + W_i.columns());
        W.put(new INDArrayIndex[] {NDArrayIndex.interval(0, W.rows()), NDArrayIndex.interval(0, W_c.columns())}, W_c);
        W.put(new INDArrayIndex[] {NDArrayIndex.interval(0, W.rows()),
                        NDArrayIndex.interval(W_c.columns(), W_c.columns() + W_f.columns())}, W_f);
        W.put(new INDArrayIndex[] {NDArrayIndex.interval(0, W.rows()), NDArrayIndex
                        .interval(W_c.columns() + W_f.columns(), W_c.columns() + W_f.columns() + W_o.columns())}, W_o);
        W.put(new INDArrayIndex[] {NDArrayIndex.interval(0, W.rows()),
                        NDArrayIndex.interval(W_c.columns() + W_f.columns() + W_o.columns(),
                                        W_c.columns() + W_f.columns() + W_o.columns() + W_i.columns())},
                        W_i);
        this.weights.put(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY, W);

        INDArray U_c;
        if (weights.containsKey(KERAS_PARAM_NAME_U_C))
            U_c = weights.get(KERAS_PARAM_NAME_U_C);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_U_C);
        INDArray U_f;
        if (weights.containsKey(KERAS_PARAM_NAME_U_F))
            U_f = weights.get(KERAS_PARAM_NAME_U_F);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_U_F);
        INDArray U_o;
        if (weights.containsKey(KERAS_PARAM_NAME_U_O))
            U_o = weights.get(KERAS_PARAM_NAME_U_O);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_U_O);
        INDArray U_i;
        if (weights.containsKey(KERAS_PARAM_NAME_U_I))
            U_i = weights.get(KERAS_PARAM_NAME_U_I);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_U_I);
        /* DL4J has three additional columns in its recurrent weights matrix that don't appear in Keras LSTMs.
         * These are for peephole connections. Since Keras doesn't use them, we zero them out.
         */
        INDArray U = Nd4j.zeros(U_c.rows(), U_c.columns() + U_f.columns() + U_o.columns() + U_i.columns() + 3);
        U.put(new INDArrayIndex[] {NDArrayIndex.interval(0, U.rows()), NDArrayIndex.interval(0, U_c.columns())}, U_c);
        U.put(new INDArrayIndex[] {NDArrayIndex.interval(0, U.rows()),
                        NDArrayIndex.interval(U_c.columns(), U_c.columns() + U_f.columns())}, U_f);
        U.put(new INDArrayIndex[] {NDArrayIndex.interval(0, U.rows()), NDArrayIndex
                        .interval(U_c.columns() + U_f.columns(), U_c.columns() + U_f.columns() + U_o.columns())}, U_o);
        U.put(new INDArrayIndex[] {NDArrayIndex.interval(0, U.rows()),
                        NDArrayIndex.interval(U_c.columns() + U_f.columns() + U_o.columns(),
                                        U_c.columns() + U_f.columns() + U_o.columns() + U_i.columns())},
                        U_i);
        this.weights.put(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY, U);

        INDArray b_c;
        if (weights.containsKey(KERAS_PARAM_NAME_B_C))
            b_c = weights.get(KERAS_PARAM_NAME_B_C);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_B_C);
        INDArray b_f;
        if (weights.containsKey(KERAS_PARAM_NAME_B_F))
            b_f = weights.get(KERAS_PARAM_NAME_B_F);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_B_F);
        INDArray b_o;
        if (weights.containsKey(KERAS_PARAM_NAME_B_O))
            b_o = weights.get(KERAS_PARAM_NAME_B_O);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_B_O);
        INDArray b_i;
        if (weights.containsKey(KERAS_PARAM_NAME_B_I))
            b_i = weights.get(KERAS_PARAM_NAME_B_I);
        else
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_B_I);
        INDArray b = Nd4j.zeros(b_c.rows(), b_c.columns() + b_f.columns() + b_o.columns() + b_i.columns());
        b.put(new INDArrayIndex[] {NDArrayIndex.interval(0, b.rows()), NDArrayIndex.interval(0, b_c.columns())}, b_c);
        b.put(new INDArrayIndex[] {NDArrayIndex.interval(0, b.rows()),
                        NDArrayIndex.interval(b_c.columns(), b_c.columns() + b_f.columns())}, b_f);
        b.put(new INDArrayIndex[] {NDArrayIndex.interval(0, b.rows()), NDArrayIndex
                        .interval(b_c.columns() + b_f.columns(), b_c.columns() + b_f.columns() + b_o.columns())}, b_o);
        b.put(new INDArrayIndex[] {NDArrayIndex.interval(0, b.rows()),
                        NDArrayIndex.interval(b_c.columns() + b_f.columns() + b_o.columns(),
                                        b_c.columns() + b_f.columns() + b_o.columns() + b_i.columns())},
                        b_i);
        this.weights.put(GravesLSTMParamInitializer.BIAS_KEY, b);

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
     * @return
     */
    public boolean getUnroll() {
        return this.unroll;
    }

    public static boolean getUnrollRecurrentLayer(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_UNROLL))
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer config missing " + LAYER_FIELD_UNROLL + " field");
        return (boolean) innerConfig.get(LAYER_FIELD_UNROLL);
    }

    /**
     * Get LSTM recurrent weight initialization from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return                  epsilon
     * @throws InvalidKerasConfigurationException
     */
    public static WeightInit getRecurrentWeightInitFromConfig(Map<String, Object> layerConfig, boolean train)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_INNER_INIT))
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer config missing " + LAYER_FIELD_INNER_INIT + " field");
        String kerasInit = (String) innerConfig.get(LAYER_FIELD_INNER_INIT);
        WeightInit init;
        try {
            init = mapWeightInitialization(kerasInit);
        } catch (UnsupportedKerasConfigurationException e) {
            if (train)
                throw e;
            else {
                init = WeightInit.XAVIER;
                log.warn("Unknown weight initializer " + kerasInit + " (Using XAVIER instead).");
            }
        }
        return init;
    }

    /**
     * Get LSTM recurrent weight dropout from Keras layer configuration. Currently unsupported.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return                  epsilon
     * @throws InvalidKerasConfigurationException
     */
    public static double getRecurrentDropout(Map<String, Object> layerConfig)
                    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        /* NOTE: Keras "dropout" parameter determines dropout probability,
         * while DL4J "dropout" parameter determines retention probability.
         */
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        double dropout = 1.0;
        if (innerConfig.containsKey(LAYER_FIELD_DROPOUT_U))
            dropout = 1.0 - (double) innerConfig.get(LAYER_FIELD_DROPOUT_U);
        if (dropout < 1.0)
            throw new UnsupportedKerasConfigurationException(
                            "Dropout > 0 on LSTM recurrent connections not supported.");
        return dropout;
    }

    /**
     * Get LSTM gate activation function from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return                  epsilon
     * @throws InvalidKerasConfigurationException
     */
    public static IActivation getGateActivationFromConfig(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_INNER_ACTIVATION))
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer config missing " + LAYER_FIELD_INNER_ACTIVATION + " field");
        return mapActivation((String) innerConfig.get(LAYER_FIELD_INNER_ACTIVATION));
    }

    /**
     * Get LSTM forget gate bias initialization from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return                  epsilon
     * @throws InvalidKerasConfigurationException
     */
    public static double getForgetBiasInitFromConfig(Map<String, Object> layerConfig, boolean train)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_FORGET_BIAS_INIT))
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer config missing " + LAYER_FIELD_FORGET_BIAS_INIT + " field");
        String kerasForgetBiasInit = (String) innerConfig.get(LAYER_FIELD_FORGET_BIAS_INIT);
        double init = 0;
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
