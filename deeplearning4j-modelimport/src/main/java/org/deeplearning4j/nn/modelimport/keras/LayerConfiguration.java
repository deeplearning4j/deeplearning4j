package org.deeplearning4j.nn.modelimport.keras;

import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Routines for importing saved Keras layer configurations.
 *
 * @author davekale
 */
public class LayerConfiguration {
    public static final String KERAS_REGULARIZATION_TYPE_L1 = "l1";
    public static final String KERAS_REGULARIZATION_TYPE_L2 = "l2";
    public static final String KERAS_LAYER_PROPERTY_NAME = "name";
    public static final String KERAS_LAYER_PROPERTY_DROPOUT = "dropout";
    public static final String KERAS_LAYER_PROPERTY_ACTIVATION = "activation";
    public static final String KERAS_LAYER_PROPERTY_INIT = "init";
    public static final String KERAS_LAYER_PROPERTY_W_REGULARIZER = "W_regularizer";
    public static final String KERAS_LAYER_PROPERTY_B_REGULARIZER = "b_regularizer";
    public static final String KERAS_LAYER_PROPERTY_OUTPUT_DIM = "output_dim";
    public static final String KERAS_LAYER_PROPERTY_SUBSAMPLE = "subsample";
    public static final String KERAS_LAYER_PROPERTY_NB_ROW = "nb_row";
    public static final String KERAS_LAYER_PROPERTY_NB_COL = "nb_col";
    public static final String KERAS_LAYER_PROPERTY_NB_FILTER = "nb_filter";
    public static final String KERAS_LAYER_PROPERTY_STRIDES = "strides";
    public static final String KERAS_LAYER_PROPERTY_POOL_SIZE = "pool_size";
    public static final String KERAS_MODEL_PROPERTY_CLASS = "keras_class";
    public static final String KERAS_LAYER_PROPERTY_INNER_ACTIVATION = "inner_activation";
    public static final String KERAS_LAYER_PROPERTY_INNER_INIT = "inner_init";
    public static final String KERAS_LAYER_PROPERTY_DROPOUT_U = "dropout_U";
    public static final String KERAS_LAYER_PROPERTY_FORGET_BIAS_INIT = "forget_bias_init";
    public static final String KERAS_LAYER_PROPERTY_DROPOUT_W = "dropout_W";
    public static final String KERAS_ACTIVATION_LINEAR = "linear";
    public static final String DL4J_ACTIVATION_IDENTITY = "identity";
    public static final String KERAS_LAYER_DENSE = "Dense";
    public static final String KERAS_LAYER_TIME_DISTRIBUTED_DENSE = "TimeDistributedDense";
    public static final String KERAS_LAYER_LSTM = "LSTM";
    public static final String KERAS_LAYER_CONVOLUTION_2D = "Convolution2D";
    public static final String KERAS_LAYER_MAX_POOLING_2D = "MaxPooling2D";
    public static final String KERAS_LAYER_FLATTEN = "Flatten";
    public static final String KERAS_INIT_UNIFORM = "uniform";
    public static final String KERAS_INIT_ZERO = "zero";
    public static final String KERAS_INIT_GLOROT_NORMAL = "glorot_normal";
    public static final String KERAS_INIT_GLOROT_UNIFORM = "glorot_uniform";
    public static final String KERAS_INIT_HE_NORMAL = "he_normal";
    public static final String KERAS_INIT_HE_UNIFORM = "he_uniform";
    public static final String KERAS_INIT_LECUN_UNIFORM = "lecun_uniform";
    public static final String KERAS_INIT_NORMAL = "normal";
    public static final String KERAS_INIT_ORTHOGONAL = "orthogonal";
    public static final String KERAS_INIT_IDENTITY = "identity";
    public static final String KERAS_FORGET_BIAS_ZERO = "zero";
    public static final String KERAS_FORGET_BIAS_ONE = "one";
    private static Logger log = LoggerFactory.getLogger(LayerConfiguration.class);

    private LayerConfiguration() {}

    /**
     * Configure DL4J Layer from a Keras layer configuration.
     *
     * @param kerasLayerClass  String containing the Keras layer class type
     * @param kerasConfig      Map containing Keras layer properties
     * @return                 DL4J Layer configuration
     * @see Layer
     */
    public static Layer buildLayer(String kerasLayerClass, Map<String,Object> kerasConfig) {
        return buildLayer(kerasLayerClass, kerasConfig, false);
    }

    /**
     * Configure DL4J Layer from a Keras layer configuration.
     *
     * @param kerasLayerClass  String containing the Keras layer class type
     * @param kerasConfig      Map containing Keras layer properties
     * @param isOutput         Whether this is an output layer
     * @return                 DL4J Layer configuration
     * @see Layer
     */
    public static Layer buildLayer(String kerasLayerClass, Map<String,Object> kerasConfig, boolean isOutput) {
        Layer layer = null;
        switch (kerasLayerClass) {
            case KERAS_LAYER_DENSE:
            case KERAS_LAYER_TIME_DISTRIBUTED_DENSE:
                layer = buildDenseLayer(kerasConfig);
                break;
            case KERAS_LAYER_LSTM:
                layer = buildGravesLstmLayer(kerasConfig);
                break;
            case KERAS_LAYER_CONVOLUTION_2D:
                layer = buildConvolutionLayer(kerasConfig);
                break;
            case KERAS_LAYER_MAX_POOLING_2D:
                layer = buildSubsamplingLayer(kerasConfig);
                break;
            case KERAS_LAYER_FLATTEN:
                log.warn("DL4J adds reshaping layers during model compilation: https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/MultiLayerConfiguration.java#L429");
                break;
            default:
                throw new IncompatibleKerasConfigurationException("Unsupported keras layer type " + kerasLayerClass);
        }
        return layer;
    }

    /**
     * Map Keras to DL4J activation functions.
     *
     * @param kerasActivation   String containing Keras activation function name
     * @return                  String containing DL4J activation function name
     */
    public static String mapActivation(String kerasActivation) {
        if (kerasActivation.equals(KERAS_ACTIVATION_LINEAR))
            return DL4J_ACTIVATION_IDENTITY;
        return kerasActivation;
    }

    /**
     * Map Keras to DL4J weight initialization functions.
     *
     * @param kerasInit     String containing Keras initialization function name
     * @return              DL4J weight initialization enum
     * @see WeightInit
     */
    public static WeightInit mapWeightInitialization(String kerasInit) {
        /* WEIGHT INITIALIZATION
         * TODO: finish mapping keras-to-dl4j weight distributions.
         * Low priority since our focus is on loading trained models.
         *
         * Remaining dl4j distributions: DISTRIBUTION, SIZE, NORMALIZED,
         * VI, RELU, XAVIER
         */
        WeightInit init = WeightInit.XAVIER;
        if (kerasInit != null) {
            switch (kerasInit) {
                case KERAS_INIT_UNIFORM:
                    init = WeightInit.UNIFORM;
                    break;
                case KERAS_INIT_ZERO:
                    init = WeightInit.ZERO;
                    break;
                case KERAS_INIT_GLOROT_NORMAL:
                    init = WeightInit.XAVIER;
                    break;
                case KERAS_INIT_GLOROT_UNIFORM:
                    init = WeightInit.XAVIER_UNIFORM;
                    break;
                case KERAS_INIT_HE_NORMAL:
                    init = WeightInit.RELU;
                    break;
                case KERAS_INIT_HE_UNIFORM:
                    init = WeightInit.RELU_UNIFORM;
                    break;
                case KERAS_INIT_LECUN_UNIFORM:
                case KERAS_INIT_NORMAL:
                case KERAS_INIT_IDENTITY:
                case KERAS_INIT_ORTHOGONAL:
                default:
                    log.warn("Unknown keras weight distribution " + init);
                    break;
            }
        }
        return init;
    }

    /**
     * Get L1 regularization (if any) from Keras weight regularization configuration.
     *
     * @param regularizerConfig     Map containing Keras weight reguarlization configuration
     * @return                      L1 regularization strength (0.0 if none)
     */
    public static double getL1Regularization(Map<String,Object> regularizerConfig) {
        if (regularizerConfig != null && regularizerConfig.containsKey(KERAS_REGULARIZATION_TYPE_L1))
            return (double)regularizerConfig.get(KERAS_REGULARIZATION_TYPE_L1);
        return 0.0;
    }

    /**
     * Get L2 regularization (if any) from Keras weight regularization configuration.
     *
     * @param regularizerConfig     Map containing Keras weight reguarlization configuration
     * @return                      L2 regularization strength (0.0 if none)
     */
    public static double getL2Regularization(Map<String,Object> regularizerConfig) {
        if (regularizerConfig != null && regularizerConfig.containsKey(KERAS_REGULARIZATION_TYPE_L2))
            return (double)regularizerConfig.get(KERAS_REGULARIZATION_TYPE_L2);
        return 0.0;
    }

    /**
     * Check whether Keras weight regularization is of unknown type. Currently prints a warning
     * since main use case for model import is inference, not further training. Unlikely since
     * standard Keras weight regularizers are L1 and L2.
     *
     * @param regularizerConfig     Map containing Keras weight reguarlization configuration
     * @return                      L1 regularization strength (0.0 if none)
     *
     * TODO: should this throw an error instead?
     */
    public static void checkForUnknownRegularizer(Map<String, Object> regularizerConfig) {
        if (regularizerConfig != null) {
            Set<String> regularizerFields = regularizerConfig.keySet();
            regularizerFields.remove(KERAS_REGULARIZATION_TYPE_L1);
            regularizerFields.remove(KERAS_REGULARIZATION_TYPE_L2);
            regularizerFields.remove(KERAS_LAYER_PROPERTY_NAME);
            if (regularizerFields.size() > 0) {
                String unknownField = (String) regularizerFields.toArray()[0];
                log.warn("Unknown regularization field: " + unknownField);
            }
        }
    }

    /**
     * Perform layer configuration steps that are common across all Keras and DL4J layer types.
     *
     * @param builder       DL4J Layer builder object
     * @param kerasConfig   Map containing Keras layer properties
     * @return              DL4J Layer builder object
     * @throws NotImplementedException
     * @see Layer.Builder
     */
    public static Layer.Builder finishLayerConfig(Layer.Builder builder, Map<String,Object> kerasConfig)
        throws NotImplementedException {
        if (kerasConfig.containsKey(KERAS_LAYER_PROPERTY_DROPOUT))
            builder.dropOut((double)kerasConfig.get(KERAS_LAYER_PROPERTY_DROPOUT));
        if (kerasConfig.containsKey(KERAS_LAYER_PROPERTY_ACTIVATION))
            builder.activation(mapActivation((String)kerasConfig.get(KERAS_LAYER_PROPERTY_ACTIVATION)));
        builder.name((String)kerasConfig.get(KERAS_LAYER_PROPERTY_NAME));
        if (kerasConfig.containsKey(KERAS_LAYER_PROPERTY_INIT)) {
            WeightInit init = mapWeightInitialization((String) kerasConfig.get(KERAS_LAYER_PROPERTY_INIT));
            builder.weightInit(init);
            if (init == WeightInit.ZERO)
                builder.biasInit(0.0);
        }
        if (kerasConfig.containsKey(KERAS_LAYER_PROPERTY_W_REGULARIZER)) {
            Map<String,Object> regularizerConfig = (Map<String,Object>)kerasConfig.get(KERAS_LAYER_PROPERTY_W_REGULARIZER);
            double l1 = getL1Regularization(regularizerConfig);
            if (l1 > 0)
                builder.l1(l1);
            double l2 = getL2Regularization(regularizerConfig);
            if (l2 > 0)
                builder.l2(l2);
            checkForUnknownRegularizer(regularizerConfig);
        }
        if (kerasConfig.containsKey(KERAS_LAYER_PROPERTY_B_REGULARIZER)) {
            Map<String,Object> regularizerConfig = (Map<String,Object>)kerasConfig.get(KERAS_LAYER_PROPERTY_B_REGULARIZER);
            double l1 = getL1Regularization(regularizerConfig);
            double l2 = getL2Regularization(regularizerConfig);
            if (l1 > 0 || l2 > 0)
                throw new NotImplementedException("Bias regularization not implemented");
        }
        return builder;
    }

    /**
     * Configure DL4J DenseLayer from a Keras Dense configuration.
     *
     * @param kerasConfig      Map containing Keras Dense layer properties
     * @return                 DL4J DenseLayer configuration
     * @throws NotImplementedException
     * @see DenseLayer
     */
    public static DenseLayer buildDenseLayer(Map<String,Object> kerasConfig)
        throws NotImplementedException {
        DenseLayer.Builder builder = new DenseLayer.Builder()
                .nOut((int)kerasConfig.get(KERAS_LAYER_PROPERTY_OUTPUT_DIM));
        finishLayerConfig(builder, kerasConfig);
        return builder.build();
    }

    /**
     * Configure DL4J ConvolutionLayer from a Keras *Convolution configuration.
     *
     * @param kerasConfig      Map containing Keras *Convolution layer properties
     * @return                 DL4J ConvolutionLayer configuration
     * @throws NotImplementedException
     * @see ConvolutionLayer
     *
     * TODO: verify whether works for 1D convolutions.
     */
    public static ConvolutionLayer buildConvolutionLayer(Map<String,Object> kerasConfig)
        throws NotImplementedException {
        List<Integer> stride = (List<Integer>)kerasConfig.get(KERAS_LAYER_PROPERTY_SUBSAMPLE);
        int nb_row = (Integer)kerasConfig.get(KERAS_LAYER_PROPERTY_NB_ROW);
        int nb_col = (Integer)kerasConfig.get(KERAS_LAYER_PROPERTY_NB_COL);
        ConvolutionLayer.Builder builder = new ConvolutionLayer.Builder()
                .stride(stride.get(0), stride.get(1))
                .kernelSize(nb_row, nb_col)
                .nOut((int)kerasConfig.get(KERAS_LAYER_PROPERTY_NB_FILTER));
        finishLayerConfig(builder, kerasConfig);
        return builder.build();
    }

    /**
     * Configure DL4J SubsamplingLayer from a Keras *Pooling* configuration.
     *
     * @param kerasConfig      Map containing Keras *Pooling* layer properties
     * @return                 DL4J SubsamplingLayer configuration
     * @throws NotImplementedException
     * @see SubsamplingLayer
     *
     * TODO: add other pooling layer types and shapes.
     */
    public static SubsamplingLayer buildSubsamplingLayer(Map<String,Object> kerasConfig)
        throws NotImplementedException {
        List<Integer> stride = (List<Integer>)kerasConfig.get(KERAS_LAYER_PROPERTY_STRIDES);
        List<Integer> pool = (List<Integer>)kerasConfig.get(KERAS_LAYER_PROPERTY_POOL_SIZE);
        SubsamplingLayer.Builder builder = new SubsamplingLayer.Builder()
                                                .stride(stride.get(0), stride.get(1))
                                                .kernelSize(pool.get(0), pool.get(1));
        switch ((String)kerasConfig.get(KERAS_MODEL_PROPERTY_CLASS)) {
            case KERAS_LAYER_MAX_POOLING_2D:
                builder.poolingType(SubsamplingLayer.PoolingType.MAX);
                break;
            /* TODO: add other pooling layer types and shapes. */
            default:
                throw new NotImplementedException("Other pooling types and shapes not supported.");
        }
        finishLayerConfig(builder, kerasConfig);
        return builder.build();
    }

    /**
     * Configure DL4J GravesLSTM layer from a Keras LSTM configuration.
     *
     * @param kerasConfig      Map containing Keras LSTM layer properties
     * @return                 DL4J GravesLSTM configuration
     * @throws IncompatibleKerasConfigurationException
     * @throws NotImplementedException
     * @see GravesLSTM
     */
    public static GravesLSTM buildGravesLstmLayer(Map<String,Object> kerasConfig)
        throws IncompatibleKerasConfigurationException, NotImplementedException {
        if (!kerasConfig.get(KERAS_LAYER_PROPERTY_ACTIVATION).equals(kerasConfig.get(KERAS_LAYER_PROPERTY_INNER_ACTIVATION)))
            throw new IncompatibleKerasConfigurationException("Specifying different activation for LSTM inner cells not supported.");
        if (!kerasConfig.get(KERAS_LAYER_PROPERTY_INIT).equals(kerasConfig.get(KERAS_LAYER_PROPERTY_INNER_INIT)))
            log.warn("Specifying different initialization for inner cells not supported.");
        if ((double)kerasConfig.get(KERAS_LAYER_PROPERTY_DROPOUT_U) > 0.0)
            throw new IncompatibleKerasConfigurationException("Dropout > 0 on LSTM recurrent connections not supported.");

        GravesLSTM.Builder builder = new GravesLSTM.Builder();
        builder.nOut((int)kerasConfig.get(KERAS_LAYER_PROPERTY_OUTPUT_DIM));
        String forgetBiasInit = (String)kerasConfig.get(KERAS_LAYER_PROPERTY_FORGET_BIAS_INIT);
        switch (forgetBiasInit) {
            case KERAS_FORGET_BIAS_ZERO:
                builder.forgetGateBiasInit(0.0);
                break;
            case KERAS_FORGET_BIAS_ONE:
                builder.forgetGateBiasInit(1.0);
                break;
            default:
                log.warn("Unsupported bias initialization: " + forgetBiasInit + ".");
                break;
        }
        kerasConfig.put(KERAS_LAYER_PROPERTY_DROPOUT, (double)kerasConfig.get(KERAS_LAYER_PROPERTY_DROPOUT_W));
        finishLayerConfig(builder, kerasConfig);
        return builder.build();
    }
}
