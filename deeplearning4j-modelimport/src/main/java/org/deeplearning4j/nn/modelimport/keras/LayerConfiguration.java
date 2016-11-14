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
 * @author davekale
 */
public class LayerConfiguration {
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
            case "Dense":
            case "TimeDistributedDense":
                layer = buildDenseLayer(kerasConfig);
                break;
            case "LSTM":
                layer = buildGravesLstmLayer(kerasConfig);
                break;
            case "Convolution2D":
                layer = buildConvolutionLayer(kerasConfig);
                break;
            case "MaxPooling2D":
                layer = buildSubsamplingLayer(kerasConfig);
                break;
            case "Flatten":
                log.warn("DL4J adds reshaping layers during model compilation");
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
        if (kerasActivation.equals("linear"))
            return "identity";
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
                case "uniform":
                    init = WeightInit.UNIFORM;
                    break;
                case "zero":
                    init = WeightInit.ZERO;
                    break;
                case "lecun_uniform":
                case "normal":
                case "identity":
                case "orthogonal":
                case "glorot_normal":
                case "glorot_uniform":
                case "he_normal":
                case "he_uniform":
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
        if (regularizerConfig != null && regularizerConfig.containsKey("l1"))
            return (double)regularizerConfig.get("l1");
        return 0.0;
    }

    /**
     * Get L2 regularization (if any) from Keras weight regularization configuration.
     *
     * @param regularizerConfig     Map containing Keras weight reguarlization configuration
     * @return                      L2 regularization strength (0.0 if none)
     */
    public static double getL2Regularization(Map<String,Object> regularizerConfig) {
        if (regularizerConfig != null && regularizerConfig.containsKey("l2"))
            return (double)regularizerConfig.get("l2");
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
            regularizerFields.remove("l1");
            regularizerFields.remove("l2");
            regularizerFields.remove("name");
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
        if (kerasConfig.containsKey("dropout"))
            builder.dropOut((double)kerasConfig.get("dropout"));
        if (kerasConfig.containsKey("activation"))
            builder.activation(mapActivation((String)kerasConfig.get("activation")));
        builder.name((String)kerasConfig.get("name"));
        if (kerasConfig.containsKey("init")) {
            WeightInit init = mapWeightInitialization((String) kerasConfig.get("init"));
            builder.weightInit(init);
            if (init == WeightInit.ZERO)
                builder.biasInit(0.0);
        }
        if (kerasConfig.containsKey("W_regularizer")) {
            Map<String,Object> regularizerConfig = (Map<String,Object>)kerasConfig.get("W_regularizer");
            double l1 = getL1Regularization(regularizerConfig);
            if (l1 > 0)
                builder.l1(l1);
            double l2 = getL2Regularization(regularizerConfig);
            if (l2 > 0)
                builder.l2(l2);
            checkForUnknownRegularizer(regularizerConfig);
        }
        if (kerasConfig.containsKey("b_regularizer")) {
            Map<String,Object> regularizerConfig = (Map<String,Object>)kerasConfig.get("b_regularizer");
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
                .nOut((int)kerasConfig.get("output_dim"));
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
        List<Integer> stride = (List<Integer>)kerasConfig.get("subsample");
        int nb_row = (Integer)kerasConfig.get("nb_row");
        int nb_col = (Integer)kerasConfig.get("nb_col");
        ConvolutionLayer.Builder builder = new ConvolutionLayer.Builder()
                .stride(stride.get(0), stride.get(1))
                .kernelSize(nb_row, nb_col)
                .nOut((int)kerasConfig.get("nb_filter"));
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
        List<Integer> stride = (List<Integer>)kerasConfig.get("strides");
        List<Integer> pool = (List<Integer>)kerasConfig.get("pool_size");
        SubsamplingLayer.Builder builder = new SubsamplingLayer.Builder()
                                                .stride(stride.get(0), stride.get(1))
                                                .kernelSize(pool.get(0), pool.get(1));
        switch ((String)kerasConfig.get("keras_class")) {
            case "MaxPooling2D":
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
        if (!kerasConfig.get("activation").equals(kerasConfig.get("inner_activation")))
            throw new IncompatibleKerasConfigurationException("Specifying different activation for LSTM inner cells not supported.");
        if (!kerasConfig.get("init").equals(kerasConfig.get("inner_init")))
            log.warn("Specifying different initialization for inner cells not supported.");
        if ((double)kerasConfig.get("dropout_U") > 0.0)
            throw new IncompatibleKerasConfigurationException("Dropout > 0 on LSTM recurrent connections not supported.");

        GravesLSTM.Builder builder = new GravesLSTM.Builder();
        builder.nOut((int)kerasConfig.get("output_dim"));
        String forgetBiasInit = (String)kerasConfig.get("forget_bias_init");
        switch (forgetBiasInit) {
            case "zero":
                builder.forgetGateBiasInit(0.0);
                break;
            case "one":
                builder.forgetGateBiasInit(1.0);
                break;
            default:
                log.warn("Unsupported bias initialization: " + forgetBiasInit + ".");
                break;
        }
        kerasConfig.put("dropout", (double)kerasConfig.get("dropout_W"));
        finishLayerConfig(builder, kerasConfig);
        return builder.build();
    }
}
