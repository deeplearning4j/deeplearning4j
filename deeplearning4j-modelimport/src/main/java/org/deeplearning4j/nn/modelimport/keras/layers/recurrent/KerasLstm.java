package org.deeplearning4j.nn.modelimport.keras.layers.recurrent;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
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

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils.getActivationFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasActivationUtils.mapActivation;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasInitilizationUtils.getWeightInitFromConfig;
import static org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils.getNOutFromConfig;

/**
 * Imports a Keras LSTM layer as a DL4J LSTM layer.
 *
 * @author dave@skymind.io
 */
@Slf4j
@Data
public class KerasLstm extends KerasLayer {

    private final String LAYER_FIELD_UNROLL = "unroll";
    private final String LSTM_FORGET_BIAS_INIT_ZERO = "zero";
    private final String LSTM_FORGET_BIAS_INIT_ONE = "one";

    /* Keras layer parameter names. */
    private final String KERAS_PARAM_NAME_KERNEL = "kernel";
    private final String KERAS_PARAM_NAME_BIAS = "bias";
    private final String KERAS_PARAM_NAME_RECURRENT_KERNEL = "recurrent_kernel";

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

    protected boolean unroll = false; // whether to unroll LSTM

    /**
     * Pass-through constructor from KerasLayer
     * @param kerasVersion major keras version
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasLstm(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
        super(kerasVersion);
    }

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

        Pair<WeightInit, Distribution> init = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INIT(),
                enforceTrainingConfig, conf, kerasMajorVersion);
        WeightInit weightInit = init.getFirst();
        Distribution distribution = init.getSecond();

        Pair<WeightInit, Distribution> recurrentInit = getWeightInitFromConfig(layerConfig, conf.getLAYER_FIELD_INNER_INIT(),
                enforceTrainingConfig, conf, kerasMajorVersion);
        WeightInit recurrentWeightInit = recurrentInit.getFirst();
        Distribution recurrentDistribution = recurrentInit.getSecond();

        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        Boolean returnSequences = (Boolean) innerConfig.get(conf.getLAYER_FIELD_RETURN_SEQUENCES());
        if (!returnSequences) {
            log.warn("Keras setting 'return_sequences = False' is not properly supported," +
                    "DL4J's LSTM layer returns sequences by default");
        }
        if (weightInit != recurrentWeightInit || distribution != recurrentDistribution)
            if (enforceTrainingConfig)
                throw new UnsupportedKerasConfigurationException(
                                "Specifying different initialization for recurrent weights not supported.");
            else
                log.warn("Specifying different initialization for recurrent weights not supported.");
        getRecurrentDropout(layerConfig);
        this.unroll = getUnrollRecurrentLayer(layerConfig);

        LayerConstraint biasConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_B_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint weightConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_W_CONSTRAINT(), conf, kerasMajorVersion);
        LayerConstraint recurrentConstraint = KerasConstraintUtils.getConstraintsFromConfig(
                layerConfig, conf.getLAYER_FIELD_RECURRENT_CONSTRAINT(), conf, kerasMajorVersion);

        LSTM.Builder builder = new LSTM.Builder()
                        .gateActivationFunction(getGateActivationFromConfig(layerConfig))
                        .forgetGateBiasInit(getForgetBiasInitFromConfig(layerConfig, enforceTrainingConfig))
                        .name(this.layerName)
                        .nOut(getNOutFromConfig(layerConfig, conf))
                        .dropOut(this.dropout)
                        .activation(getActivationFromConfig(layerConfig, conf))
                        .weightInit(weightInit)
                        .biasInit(0.0)
                        .l1(this.weightL1Regularization)
                        .l2(this.weightL2Regularization);
        if (distribution != null)
            builder.dist(distribution);
        if (biasConstraint != null)
            builder.constrainBias(biasConstraint);
        if (weightConstraint != null)
            builder.constrainInputWeights(weightConstraint);
        if (recurrentConstraint != null)
            builder.constrainRecurrent(recurrentConstraint);
        this.layer = builder.build();
    }

    /**
     * Get DL4J LSTM layer.
     *
     * @return  LSTM Layer
     */
    public LSTM getLSTMLayer() {
        return (LSTM) this.layer;
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
        InputPreProcessor preProcessor = getInputPreprocessor(inputType);
        if (preProcessor != null)
            return  preProcessor.getOutputType(inputType[0]);
        else
            return this.getLSTMLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return          number of trainable parameters (12)
     */
    @Override
    public int getNumParams() {
        return kerasMajorVersion == 2 ? NUM_TRAINABLE_PARAMS_KERAS_2 : NUM_TRAINABLE_PARAMS;
    }

    /**
     * Gets appropriate DL4J InputPreProcessor for given InputTypes.
     *
     * @param  inputType    Array of InputTypes
     * @return              DL4J InputPreProcessor
     * @throws InvalidKerasConfigurationException Invalid Keras configuration exception
     * @see org.deeplearning4j.nn.conf.InputPreProcessor
     */
    @Override
    public InputPreProcessor getInputPreprocessor(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras LSTM layer accepts only one input (received " + inputType.length + ")");
        InputPreProcessor preprocessor = null;
        if (inputType[0] instanceof InputType.InputTypeFeedForward) {
            preprocessor = new FeedForwardToRnnPreProcessor();
        }
        return preprocessor;
    }



    /**
     * Set weights for layer.
     *
     * @param weights
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
            if (weights.containsKey(KERAS_PARAM_NAME_KERNEL))
                W = weights.get(KERAS_PARAM_NAME_KERNEL);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_RECURRENT_KERNEL);
            INDArray U;
            if (weights.containsKey(KERAS_PARAM_NAME_RECURRENT_KERNEL))
                U = weights.get(KERAS_PARAM_NAME_RECURRENT_KERNEL);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_RECURRENT_KERNEL);
            INDArray b;
            if (weights.containsKey(KERAS_PARAM_NAME_BIAS))
                b = weights.get(KERAS_PARAM_NAME_BIAS);
            else
                throw new InvalidKerasConfigurationException(
                        "Keras LSTM layer does not contain parameter " + KERAS_PARAM_NAME_BIAS);

            int sliceInterval = b.length() / 4;
            W_i = W.get(NDArrayIndex.all(), NDArrayIndex.interval(0, sliceInterval));
            W_f = W.get(NDArrayIndex.all(), NDArrayIndex.interval(sliceInterval, 2 * sliceInterval ));
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
        }
        else {
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
        this.weights.put(LSTMParamInitializer.INPUT_WEIGHT_KEY, W);

        INDArray U = Nd4j.zeros(U_c.rows(), U_c.columns() + U_f.columns() + U_o.columns() + U_i.columns());
        U.put(new INDArrayIndex[] {NDArrayIndex.interval(0, U.rows()), NDArrayIndex.interval(0, U_c.columns())}, U_c);
        U.put(new INDArrayIndex[] {NDArrayIndex.interval(0, U.rows()),
                        NDArrayIndex.interval(U_c.columns(), U_c.columns() + U_f.columns())}, U_f);
        U.put(new INDArrayIndex[] {NDArrayIndex.interval(0, U.rows()), NDArrayIndex
                        .interval(U_c.columns() + U_f.columns(), U_c.columns() + U_f.columns() + U_o.columns())}, U_o);
        U.put(new INDArrayIndex[] {NDArrayIndex.interval(0, U.rows()),
                        NDArrayIndex.interval(U_c.columns() + U_f.columns() + U_o.columns(),
                                        U_c.columns() + U_f.columns() + U_o.columns() + U_i.columns())},
                        U_i);
        this.weights.put(LSTMParamInitializer.RECURRENT_WEIGHT_KEY, U);

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
     * @return
     */
    public boolean getUnroll() {
        return this.unroll;
    }

    public boolean getUnrollRecurrentLayer(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(LAYER_FIELD_UNROLL))
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer config missing " + LAYER_FIELD_UNROLL + " field");
        return (boolean) innerConfig.get(LAYER_FIELD_UNROLL);
    }

    /**
     * Get LSTM recurrent weight dropout from Keras layer configuration. Currently unsupported.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return                  epsilon
     * @throws InvalidKerasConfigurationException
     */
    public double getRecurrentDropout(Map<String, Object> layerConfig)
                    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        /* NOTE: Keras "dropout" parameter determines dropout probability,
         * while DL4J "dropout" parameter determines retention probability.
         */
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        double dropout = 1.0;
        if (innerConfig.containsKey(conf.getLAYER_FIELD_DROPOUT_U()))
            dropout = 1.0 - (double) innerConfig.get(conf.getLAYER_FIELD_DROPOUT_U());
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
    public IActivation getGateActivationFromConfig(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        if (!innerConfig.containsKey(conf.getLAYER_FIELD_INNER_ACTIVATION()))
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer config missing " + conf.getLAYER_FIELD_INNER_ACTIVATION() + " field");
        return mapActivation((String) innerConfig.get(conf.getLAYER_FIELD_INNER_ACTIVATION()), conf);
    }

    /**
     * Get LSTM forget gate bias initialization from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return                  epsilon
     * @throws InvalidKerasConfigurationException
     */
    public double getForgetBiasInitFromConfig(Map<String, Object> layerConfig, boolean train)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        String kerasForgetBiasInit;
        if (innerConfig.containsKey(conf.getLAYER_FIELD_UNIT_FORGET_BIAS())) {
            kerasForgetBiasInit = LSTM_FORGET_BIAS_INIT_ONE;
        } else if (!innerConfig.containsKey(conf.getLAYER_FIELD_FORGET_BIAS_INIT()))
            throw new InvalidKerasConfigurationException(
                            "Keras LSTM layer config missing " + conf.getLAYER_FIELD_FORGET_BIAS_INIT() + " field");
        else kerasForgetBiasInit = (String) innerConfig.get(conf.getLAYER_FIELD_FORGET_BIAS_INIT());
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
