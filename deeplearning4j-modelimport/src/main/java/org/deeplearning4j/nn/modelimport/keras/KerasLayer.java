/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.modelimport.keras;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Build Layer from Keras layer configuration.
 *
 * @author dave@skymind.io
 */
public class KerasLayer {
    /* Keras layer types. */
    public static final String LAYER_FIELD_CLASS_NAME = "class_name";
    public static final String LAYER_CLASS_NAME_INPUT = "InputLayer";
    public static final String LAYER_CLASS_NAME_ACTIVATION = "Activation";
    public static final String LAYER_CLASS_NAME_DROPOUT = "Dropout";
    public static final String LAYER_CLASS_NAME_DENSE = "Dense";
    public static final String LAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE = "TimeDistributedDense";
    public static final String LAYER_CLASS_NAME_LSTM = "LSTM";
    public static final String LAYER_CLASS_NAME_CONVOLUTION_2D = "Convolution2D";
    public static final String LAYER_CLASS_NAME_MAX_POOLING_2D = "MaxPooling2D";
    public static final String LAYER_CLASS_NAME_AVERAGE_POOLING_2D = "AveragePooling2D";
    public static final String LAYER_CLASS_NAME_FLATTEN = "Flatten";
    public static final String LAYER_CLASS_NAME_RESHAPE = "Reshape";
    public static final String LAYER_CLASS_NAME_REPEATVECTOR = "RepeatVector";
    public static final String LAYER_CLASS_NAME_MERGE = "Merge";
    public static final String LAYER_CLASS_NAME_BATCHNORMALIZATION = "BatchNormalization";

    /* Keras layer configurations. */
    public static final String LAYER_FIELD_CONFIG = "config";
    public static final String LAYER_FIELD_NAME = "name";
    public static final String LAYER_FIELD_DROPOUT = "dropout";
    public static final String LAYER_FIELD_OUTPUT_DIM = "output_dim";
    public static final String LAYER_FIELD_SUBSAMPLE = "subsample";
    public static final String LAYER_FIELD_NB_ROW = "nb_row";
    public static final String LAYER_FIELD_NB_COL = "nb_col";
    public static final String LAYER_FIELD_NB_FILTER = "nb_filter";
    public static final String LAYER_FIELD_STRIDES = "strides";
    public static final String LAYER_FIELD_POOL_SIZE = "pool_size";
    public static final String LAYER_FIELD_DROPOUT_U = "dropout_U";
    public static final String LAYER_FIELD_DROPOUT_W = "dropout_W";
    public static final String LAYER_FIELD_BATCH_INPUT_SHAPE = "batch_input_shape";
    public static final String LAYER_FIELD_INBOUND_NODES = "inbound_nodes";
    public static final String LAYER_FIELD_BORDER_MODE = "border_mode";
    public static final String LAYER_FIELD_GAMMA_REGULARIZER = "gamma_regularizer";
    public static final String LAYER_FIELD_BETA_REGULARIZER = "beta_regularizer";
    public static final String LAYER_FIELD_MODE = "mode";
    public static final String LAYER_FIELD_AXIS = "axis";
    public static final String LAYER_FIELD_EPSILON = "epsilon";
    public static final String LAYER_FIELD_MOMENTUM = "momentum";

    /* Keras convolution border modes. */
    public static final String LAYER_BORDER_MODE_SAME = "same";
    public static final String LAYER_BORDER_MODE_VALID = "valid";
    public static final String LAYER_BORDER_MODE_FULL = "full";

    /* Keras batch norm modes. */
    public static final int LAYER_BATCHNORM_MODE_1 = 1;
    public static final int LAYER_BATCHNORM_MODE_2 = 2;

    /* Keras weight regularizers. */
    public static final String LAYER_FIELD_W_REGULARIZER = "W_regularizer";
    public static final String LAYER_FIELD_B_REGULARIZER = "b_regularizer";
    public static final String REGULARIZATION_TYPE_L1 = "l1";
    public static final String REGULARIZATION_TYPE_L2 = "l2";

    /* Keras weight initializers. */
    public static final String LAYER_FIELD_INIT = "init";
    public static final String LAYER_FIELD_INNER_INIT = "inner_init";
    public static final String INIT_UNIFORM = "uniform";
    public static final String INIT_ZERO = "zero";
    public static final String INIT_GLOROT_NORMAL = "glorot_normal";
    public static final String INIT_GLOROT_UNIFORM = "glorot_uniform";
    public static final String INIT_HE_NORMAL = "he_normal";
    public static final String INIT_HE_UNIFORM = "he_uniform";
    public static final String INIT_LECUN_UNIFORM = "lecun_uniform";
    public static final String INIT_NORMAL = "normal";
    public static final String INIT_ORTHOGONAL = "orthogonal";
    public static final String INIT_IDENTITY = "identity";

    /* Keras and DL4J activation types. */
    public static final String LAYER_FIELD_ACTIVATION = "activation";
    public static final String LAYER_FIELD_INNER_ACTIVATION = "inner_activation";
    public static final String KERAS_ACTIVATION_LINEAR = "linear";
    public static final String DL4J_ACTIVATION_IDENTITY = "identity";
    public static final String KERAS_ACTIVATION_HARD_SIGMOID = "hard_sigmoid";
    public static final String DL4J_ACTIVATION_HARDSIGMOID = "hardsigmoid";

    /* Keras LSTM forget gate bias initializations. */
    public static final String LAYER_FIELD_FORGET_BIAS_INIT = "forget_bias_init";
    public static final String LSTM_FORGET_BIAS_INIT_ZERO = "zero";
    public static final String LSTM_FORGET_BIAS_INIT_ONE = "one";

    /* Keras dimension ordering for, e.g., convolutional layerNamesOrdered. */
    public static final String LAYER_FIELD_DIM_ORDERING = "dim_ordering";
    public static final String DIM_ORDERING_THEANO = "th";
    public static final String DIM_ORDERING_TENSORFLOW = "tf";

    /* Keras loss functions. */
    public static final String LAYER_CLASS_NAME_LOSS = "Loss"; // Not a Keras layer
    public static final String LAYER_FIELD_LOSS = "loss"; // Not a Keras layer field
    public static final String LOSS_SQUARED_LOSS_1 = "mean_squared_error";
    public static final String KERAS_LOSS_SQUARED_LOSS_2 = "mse";
    public static final String KERAS_LOSS_MEAN_ABSOLUTE_ERROR_1 = "mean_absolute_error";
    public static final String KERAS_LOSS_MEAN_ABSOLUTE_ERROR_2 = "mae";
    public static final String KERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR_1 = "mean_absolute_percentage_error";
    public static final String KERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR_2 = "mape";
    public static final String KERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR_1 = "mean_squared_logarithmic_error";
    public static final String KERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR_2 = "msle";
    public static final String KERAS_LOSS_SQUARED_HINGE = "squared_hinge";
    public static final String KERAS_LOSS_HINGE    = "hinge";
    public static final String KERAS_LOSS_XENT = "binary_crossentropy";
    public static final String KERAS_LOSS_MCXENT = "categorical_crossentropy";
    public static final String KERAS_LOSS_SP_XE    = "sparse_categorical_crossentropy";
    public static final String KERAS_LOSS_KL_DIVERGENCE_1 = "kullback_leibler_divergence";
    public static final String KERAS_LOSS_KL_DIVERGENCE_2 = "kld";
    public static final String KERAS_LOSS_POISSON  = "poisson";
    public static final String KERAS_LOSS_COSINE_PROXIMITY = "cosine_proximity";

    /* Logging. */
    private static Logger log = LoggerFactory.getLogger(KerasLayer.class);

    /* Keras backends store convolutional inputs and weights
     * in tensors with different dimension orders.
     */
    public enum DimOrder {
        NONE, THEANO, TENSORFLOW, UNKNOWN;
    }

    private Map<String,Object> layerConfig;       // Keras layer configuration
    private String className;                     // Keras layer class name
    private String layerName;                     // Keras layer name
    private DimOrder dimOrder = DimOrder.NONE; // Keras layer backend dimension order
    private int[] inputShape;                     // Keras layer input shape
    private List<String> inboundLayerNames = new ArrayList<>(); // List of inbound layers
    private Layer dl4jLayer;                      // Resulting DL4J layer
    private boolean train;                        // Whether layer is meant for training

    /**
     * Constructor. Sets train to false (i.e., layer is meant for inference only).
     *
     * @param layerConfig       nested map containing Keras layer configuration
     */
    public KerasLayer(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, false);
    }

    /**
     * Constructor. "train" parameter controls whether layer is built for training. This
     * controls behavior of certain exceptions. In training mode, passing an unsupported
     * regularizer will generate an error. In non-training mode, it generates only a
     * warning.
     *
     * @param layerConfig       nested map containing Keras layer configuration
     * @param train             whether layer should be built for training (controls certain exceptions)
     */
    public KerasLayer(Map<String,Object> layerConfig, boolean train)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String,Object> outerConfig = layerConfig;
        this.className = (String)checkAndGetField(outerConfig, LAYER_FIELD_CLASS_NAME);
        Map<String,Object> innerConfig = (Map<String,Object>)checkAndGetField(outerConfig, LAYER_FIELD_CONFIG);
        for (String field : outerConfig.keySet())
            if (!field.equals(LAYER_FIELD_CONFIG))
                innerConfig.put(field, outerConfig.get(field));
        this.layerConfig = innerConfig;

        this.train = train;
        this.layerName = (String)checkAndGetField(this.layerConfig, LAYER_FIELD_NAME);
        this.dl4jLayer = buildLayerFromConfig(this.layerConfig, this.train);
        this.dimOrder = getDimOrderFromConfig(this.layerConfig);
        this.inputShape = getInputShapeFromConfig(this.layerConfig, this.dimOrder);
        this.inboundLayerNames = getInboundLayerNamesFromConfig(this.layerConfig);
    }

    /**
     * Get Keras layer configuration.
     *
     * @return      Keras layer configuration
     */
    public Map<String,Object> getConfiguration() {
        return this.layerConfig;
    }

    /**
     * Get Keras layer class name.
     *
     * @return      Keras layer class name
     */
    public String getClassName() {
        return this.className;
    }

    /**
     * Get Keras layer name.
     *
     * @return      layer name
     */
    public String getName() {
        return this.layerName;
    }

    /**
     * Get Keras layer backend dimension order.
     *
     * @return      Keras layer (backend) dimension order
     */
    public DimOrder getDimOrder() {
        return this.dimOrder;
    }

    /**
     * Get Keras layer input shape.
     *
     * @return      input shape
     */
    public int[] getInputShape() {
        return this.inputShape;
    }

    /**
     * Get list of inbound layers.
     *
     * @return      list of inbound layer names
     */
    public List<String> getInboundLayerNames() {
        return this.inboundLayerNames;
    }

    /**
     * Set list of inbound layers.
     *
     * @param inboundLayerNames
     */
    public void setInboundLayerNames(List<String> inboundLayerNames) {
        this.inboundLayerNames = new ArrayList<String>(inboundLayerNames);
    }

    /**
     * Add layer to list of inbound layers.
     *
     * @param layer
     */
    public void addInboundLayer(String layer) {
        this.inboundLayerNames.add(layer);
    }

    /**
     * Gets "training" mode status.
     *
     * @return      training mode
     */
    public boolean getTrain() {
        return this.train;
    }

    /**
     * Indicates whether this layer a valid inbound layer. Currently, only
     * (known) DL4J Layers and inputs are valid inbound layers. "Preprocessor"
     * layers (reshaping, merging, etc.) are replaced by their own inbound layers.
     *
     * TODO: revisit this once "preprocessor" layers are handled explicitly
     *
     * @return      boolean indicating whether layer is valid inbound layer
     * @see org.deeplearning4j.nn.api.Layer
     */
    public boolean isValidInboundLayer() {
        return this.dl4jLayer != null || this.className.equals(LAYER_CLASS_NAME_INPUT);
    }

    /**
     * Indicates whether this layer is a DL4J Layer.
     *
     * @return      boolean indicating whether layer is DL4J Layer
     * @see org.deeplearning4j.nn.api.Layer
     */
    public boolean isDl4jLayer() {
        return this.dl4jLayer != null;
    }

    /**
     * Gets corresponding DL4J Layer, if any.
     *
     * @return      DL4J Layer
     * @see org.deeplearning4j.nn.api.Layer
     */
    public Layer getDl4jLayer() {
        return this.dl4jLayer;
    }

    /**
     *
     * @return      boolean indicating whether layer is DL4J Preprocessor
     * @see org.deeplearning4j.nn.graph.vertex.impl.PreprocessorVertex
     */
    public boolean isDl4jPreprocessor() throws UnsupportedKerasConfigurationException {
        throw new UnsupportedKerasConfigurationException("Conversion from Keras layer to DL4J preprocessor not impemented.");
    }

    /**
     * Gets corresponding DL4J PreprocessorVertex, if any.
     *
     * @return      DL4J PreprocessorVertex
     * @see org.deeplearning4j.nn.graph.vertex.impl.PreprocessorVertex
     */
    public PreprocessorVertex getDl4jPreprocessor() throws UnsupportedKerasConfigurationException {
        throw new UnsupportedKerasConfigurationException("Conversion from Keras layer to DL4J preprocessor not impemented.");
    }

    /**
     * Create a (placeholder) input layer.
     *
     * @param layerName     name of input layer
     * @param inputShape    input shape for input layer
     * @return              KerasLayer for input layer
     */
    public static KerasLayer createInputLayer(String layerName, int[] inputShape)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put(LAYER_FIELD_NAME, layerName);
        List<Integer> batchInputShape = new ArrayList<Integer>();
        batchInputShape.add(null);
        for (int i = 0; i < inputShape.length; i++)
            batchInputShape.add(inputShape[i]);
        config.put(LAYER_FIELD_BATCH_INPUT_SHAPE, batchInputShape);
        Map<String,Object> layerConfig = new HashMap<String,Object>();
        layerConfig.put(LAYER_FIELD_CONFIG, config);
        layerConfig.put(LAYER_FIELD_CLASS_NAME, LAYER_CLASS_NAME_INPUT);
        return new KerasLayer(layerConfig, false);
    }

    /**
     * Create a LossLayer in train mode to enforce requirement that we support
     * the Keras loss function.
     *
     * @param layerName     name of loss layer
     * @param kerasLoss     Keras loss function
     * @return              KerasLayer for loss layer
     */
    public static KerasLayer createLossLayer(String layerName, String kerasLoss)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return createLossLayer(layerName, kerasLoss, true);
    }

    /**
     * Create a LossLayer. Building layer with train=true will throw an exception for
     * unsupported Keras loss functions.
     *
     * @param layerName     name of loss layer
     * @param kerasLoss     Keras loss function
     * @param train         whether to build layer for training
     * @return              KerasLayer for loss layer
     */
    public static KerasLayer createLossLayer(String layerName, String kerasLoss, boolean train)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String,Object> config = new HashMap<String,Object>();
        config.put(LAYER_FIELD_NAME, layerName);
        config.put(LAYER_FIELD_LOSS, kerasLoss);
        Map<String,Object> layerConfig = new HashMap<String,Object>();
        layerConfig.put(LAYER_FIELD_CONFIG, config);
        layerConfig.put(LAYER_FIELD_CLASS_NAME, LAYER_CLASS_NAME_LOSS);
        return new KerasLayer(layerConfig, train);
    }

    /**
     * Build DL4J Layer from a Keras layer configuration.
     *
     * @param layerConfig      map containing Keras layer properties
     * @return                 DL4J Layer configuration
     * @see Layer
     */
    public static Layer buildLayerFromConfig(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return buildLayerFromConfig(layerConfig, false);
    }

    /**
     * Build DL4J Layer from a Keras layer configuration. Building layer with train=true will
     * throw exceptions for unsupported Keras options (e.g., certain types of regularizers).
     * Building layer with train=false will generate only warnings.
     *
     * @param layerConfig      map containing Keras layer properties
     * @param train            build layer in training mode
     * @return                 DL4J Layer configuration
     * @see Layer
     */
    public static Layer buildLayerFromConfig(Map<String,Object> layerConfig, boolean train)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        if (!layerConfig.containsKey(LAYER_FIELD_CLASS_NAME))
            throw new InvalidKerasConfigurationException("Missing " + LAYER_FIELD_CLASS_NAME + " field.");
        String layerClassName = (String)layerConfig.get(LAYER_FIELD_CLASS_NAME);
        Layer layer = null;
        switch (layerClassName) {
            case LAYER_CLASS_NAME_ACTIVATION:
                layer = buildActivationLayer(layerConfig, train);
                break;
            case LAYER_CLASS_NAME_DROPOUT:
                layer = buildDropoutLayer(layerConfig, train);
                break;
            case LAYER_CLASS_NAME_DENSE:
            case LAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE:
            /* TODO: test to make sure that mapping TimeDistributedDense to DenseLayer works.
             * Also, Keras recently added support for TimeDistributed layer wrapper so may
             * need to look into how that changes things.
             * */
                layer = buildDenseLayer(layerConfig, train);
                break;
            case LAYER_CLASS_NAME_LSTM:
                layer = buildGravesLstmLayer(layerConfig, train);
                break;
            case LAYER_CLASS_NAME_CONVOLUTION_2D:
            /* TODO: Add support for 1D, 3D convolutional layerNamesOrdered? */
                layer = buildConvolutionLayer(layerConfig, train);
                break;
            case LAYER_CLASS_NAME_MAX_POOLING_2D:
            case LAYER_CLASS_NAME_AVERAGE_POOLING_2D:
            /* TODO: Add support for 1D, 3D pooling layerNamesOrdered? */
                layer = buildSubsamplingLayer(layerConfig, train);
                break;
            case LAYER_CLASS_NAME_BATCHNORMALIZATION:
                layer = buildBatchNormalizationLayer(layerConfig, train);
                break;
            case LAYER_CLASS_NAME_LOSS:
                layer = buildLossLayer(layerConfig, train);
                break;
            case LAYER_CLASS_NAME_FLATTEN:
            case LAYER_CLASS_NAME_RESHAPE:
            case LAYER_CLASS_NAME_REPEATVECTOR:
            case LAYER_CLASS_NAME_MERGE:
            case LAYER_CLASS_NAME_INPUT:
                log.warn("Found Keras " + layerClassName + ". DL4J adds \"preprocessor\" layers during model compilation: https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/MultiLayerConfiguration.java#L429");
                break;
            default:
                throw new InvalidKerasConfigurationException("Unsupported keras layer type " + layerClassName);
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
        /* Keras and DL4J use the same name for most activations:
         * - softmax
         * - softplus
         * - softsign
         * - relu
         * - tanh
         * - sigmoid
         */
        String dl4jActivation = null;
        switch (kerasActivation) {
            case KERAS_ACTIVATION_LINEAR:
                dl4jActivation = DL4J_ACTIVATION_IDENTITY;
                break;
            case KERAS_ACTIVATION_HARD_SIGMOID:
                dl4jActivation = DL4J_ACTIVATION_HARDSIGMOID;
                break;
            default:
                dl4jActivation = kerasActivation;
                break;
        }
        return dl4jActivation;
    }

    /**
     * Map Keras to DL4J weight initialization functions.
     *
     * @param kerasInit     String containing Keras initialization function name
     * @return              DL4J weight initialization enum
     * @see WeightInit
     */
    public static WeightInit mapWeightInitialization(String kerasInit) throws UnsupportedKerasConfigurationException {
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
                case INIT_GLOROT_NORMAL:
                    init = WeightInit.XAVIER;
                    break;
                case INIT_GLOROT_UNIFORM:
                    init = WeightInit.XAVIER_UNIFORM;
                    break;
                case INIT_HE_NORMAL:
                    init = WeightInit.RELU;
                    break;
                case INIT_HE_UNIFORM:
                    init = WeightInit.RELU_UNIFORM;
                    break;
                case INIT_ZERO:
                    init = WeightInit.ZERO;
                    break;
                case INIT_UNIFORM:
                    /* TODO: map to DL4J dist with scale taken from config. */
                case INIT_NORMAL:
                    /* TODO: map to DL4J normal with params taken from config. */
                case INIT_IDENTITY: // does not map to existing Dl4J initializer
                case INIT_ORTHOGONAL: // does not map to existing Dl4J initializer
                case INIT_LECUN_UNIFORM: // does not map to existing Dl4J initializer
                default:
                    throw new UnsupportedKerasConfigurationException("Unknown keras weight initializer " + init);
            }
        }
        return init;
    }

    /**
     * Map Keras to DL4J loss functions.
     *
     * @param kerasLoss    String containing Keras activation function name
     * @return             String containing DL4J activation function name
     */
    public static LossFunctions.LossFunction mapLossFunction(String kerasLoss)
            throws UnsupportedKerasConfigurationException {
        LossFunctions.LossFunction dl4jLoss = LossFunctions.LossFunction.SQUARED_LOSS;
        switch (kerasLoss) {
            case LOSS_SQUARED_LOSS_1:
            case KERAS_LOSS_SQUARED_LOSS_2:
                dl4jLoss = LossFunctions.LossFunction.SQUARED_LOSS;
                break;
            case KERAS_LOSS_MEAN_ABSOLUTE_ERROR_1:
            case KERAS_LOSS_MEAN_ABSOLUTE_ERROR_2:
                dl4jLoss = LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR;
                break;
            case KERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR_1:
            case KERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR_2:
                dl4jLoss = LossFunctions.LossFunction.MEAN_ABSOLUTE_PERCENTAGE_ERROR;
                break;
            case KERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR_1:
            case KERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR_2:
                dl4jLoss = LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR;
                break;
            case KERAS_LOSS_SQUARED_HINGE:
                dl4jLoss = LossFunctions.LossFunction.SQUARED_HINGE;
                break;
            case KERAS_LOSS_HINGE:
                dl4jLoss = LossFunctions.LossFunction.HINGE;
                break;
            case KERAS_LOSS_XENT:
                dl4jLoss = LossFunctions.LossFunction.XENT;
                break;
            case KERAS_LOSS_SP_XE:
                /* TODO: should this be an error instead? */
                log.warn("Sparse cross entropy not implemented, using multiclass cross entropy instead.");
            case KERAS_LOSS_MCXENT:
                dl4jLoss = LossFunctions.LossFunction.MCXENT;
                break;
            case KERAS_LOSS_KL_DIVERGENCE_1:
            case KERAS_LOSS_KL_DIVERGENCE_2:
                dl4jLoss = LossFunctions.LossFunction.KL_DIVERGENCE;
                break;
            case KERAS_LOSS_POISSON:
                dl4jLoss = LossFunctions.LossFunction.POISSON;
                break;
            case KERAS_LOSS_COSINE_PROXIMITY:
                dl4jLoss = LossFunctions.LossFunction.COSINE_PROXIMITY;
                break;
            default:
                throw new UnsupportedKerasConfigurationException("Unknown Keras loss function " + kerasLoss);
        }
        return dl4jLoss;
    }

    /**
     * Get Keras (backend) dimension order from Keras layer configuration.
     *
     * @param layerConfig       Keras layer configuration
     * @return                  Dimension order
     */
    private DimOrder getDimOrderFromConfig(Map<String,Object> layerConfig) {
        DimOrder dimOrder = DimOrder.NONE;
        if (layerConfig.containsKey(LAYER_FIELD_DIM_ORDERING)) {
            String dimOrderStr = (String)layerConfig.get(LAYER_FIELD_DIM_ORDERING);
            switch(dimOrderStr) {
                case DIM_ORDERING_TENSORFLOW:
                    dimOrder = DimOrder.TENSORFLOW;
                    break;
                case DIM_ORDERING_THEANO:
                    dimOrder = DimOrder.THEANO;
                    break;
                default:
                    log.warn("Keras layer has unknown Keras dimension order: " + dimOrder);
                    break;
            }
        }
        return dimOrder;
    }

    /**
     * Get input shape from Keras layer configuration.
     *
     * @param layerConfig       Keras layer configuration
     * @return                  input shape array
     */
    private int[] getInputShapeFromConfig(Map<String,Object> layerConfig, DimOrder dimOrder) {
        if (!layerConfig.containsKey(LAYER_FIELD_BATCH_INPUT_SHAPE))
            return null;
        List<Integer> batchInputShape = (List<Integer>)layerConfig.get(LAYER_FIELD_BATCH_INPUT_SHAPE);
        int[] inputShape = new int[batchInputShape.size()-1];
        for (int i = 1; i < batchInputShape.size(); i++) {
            inputShape[i - 1] = batchInputShape.get(i) != null ? batchInputShape.get(i) : 0;
        }
        /* DL4J convolutional input:       # rows, # cols, # channels
         * TensorFlow convolutional input: # rows, # cols, # channels
         * Theano convolutional input:     # channels, # rows, # cols
         */
        if (dimOrder == DimOrder.THEANO && inputShape.length == 3 && this.dl4jLayer instanceof ConvolutionLayer) {
            int numChannels = inputShape[0];
            inputShape[0] = inputShape[1];
            inputShape[1] = inputShape[2];
            inputShape[2] = numChannels;
        }
        return inputShape;
    }

    /**
     * Get list of inbound layers from Keras layer configuration.
     *
     * @param layerConfig       Keras layer configuration
     * @return                  List of inbound layer names
     */
    private static List<String> getInboundLayerNamesFromConfig(Map<String,Object> layerConfig) {
        List<String> inboundNodeNames = new ArrayList<>();
        if (layerConfig.containsKey(LAYER_FIELD_INBOUND_NODES)) {
            List<Object> inboundNodes = (List<Object>)layerConfig.get(LAYER_FIELD_INBOUND_NODES);
            if (inboundNodes.size() > 0) {
                inboundNodes = (List<Object>)inboundNodes.get(0);
                for (Object o : inboundNodes) {
                    String nodeName = (String)((List<Object>)o).get(0);
                    inboundNodeNames.add(nodeName);
                }
            }
        }
        return inboundNodeNames;
    }

    /**
     * Get L1 regularization (if any) from Keras weight regularization configuration.
     *
     * @param regularizerConfig     Map containing Keras weight reguarlization configuration
     * @return                      L1 regularization strength (0.0 if none)
     */
    private static double getL1Regularization(Map<String,Object> regularizerConfig) {
        if (regularizerConfig != null && regularizerConfig.containsKey(REGULARIZATION_TYPE_L1))
            return (double)regularizerConfig.get(REGULARIZATION_TYPE_L1);
        return 0.0;
    }

    /**
     * Get L2 regularization (if any) from Keras weight regularization configuration.
     *
     * @param regularizerConfig     Map containing Keras weight reguarlization configuration
     * @return                      L2 regularization strength (0.0 if none)
     */
    private static double getL2Regularization(Map<String,Object> regularizerConfig) {
        if (regularizerConfig != null && regularizerConfig.containsKey(REGULARIZATION_TYPE_L2))
            return (double)regularizerConfig.get(REGULARIZATION_TYPE_L2);
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
    private static void checkForUnknownRegularizer(Map<String, Object> regularizerConfig, boolean train)
        throws UnsupportedKerasConfigurationException {
        if (regularizerConfig != null) {
            Set<String> regularizerFields = regularizerConfig.keySet();
            regularizerFields.remove(REGULARIZATION_TYPE_L1);
            regularizerFields.remove(REGULARIZATION_TYPE_L2);
            regularizerFields.remove(LAYER_FIELD_NAME);
            if (regularizerFields.size() > 0) {
                String unknownField = (String) regularizerFields.toArray()[0];
                if (train)
                    throw new UnsupportedKerasConfigurationException("Unknown regularization field " + unknownField);
                else
                    log.warn("Ignoring unknown regularization field " + unknownField);
            }
        }
    }

    /**
     * Build DL4J ActivationLayer from a Keras Activation configuration.
     *
     * @param layerConfig      map containing Keras Activation layer properties
     * @param train            whether to build layer in training mode
     * @return                 DL4J ActivationLayer configuration
     * @throws UnsupportedOperationException
     * @see ActivationLayer
     */
    private static ActivationLayer buildActivationLayer(Map<String, Object> layerConfig, boolean train)
            throws UnsupportedKerasConfigurationException {
        ActivationLayer.Builder builder = new ActivationLayer.Builder();
        finishLayerConfig(builder, layerConfig, train);
        return builder.build();
    }

    /**
     * Build DL4J DropoutLayer from a Keras Dropout configuration.
     *
     * @param layerConfig      map containing Keras Dropout layer properties
     * @param train            whether to build layer in training mode
     * @return                 DL4J DropoutLayer configuration
     * @throws UnsupportedOperationException
     * @see DropoutLayer
     */
    private static DropoutLayer buildDropoutLayer(Map<String, Object> layerConfig, boolean train)
            throws UnsupportedKerasConfigurationException {
        DropoutLayer.Builder builder = new DropoutLayer.Builder();
        finishLayerConfig(builder, layerConfig, train);
        return builder.build();
    }

    /**
     * Build DL4J DenseLayer from a Keras Dense configuration.
     *
     * @param layerConfig      map containing Keras Dense layer properties
     * @param train            whether to build layer in training mode
     * @return                 DL4J DenseLayer configuration
     * @throws UnsupportedOperationException
     * @see DenseLayer
     */
    private static DenseLayer buildDenseLayer(Map<String,Object> layerConfig, boolean train)
            throws UnsupportedKerasConfigurationException {
        DenseLayer.Builder builder = new DenseLayer.Builder()
                .nOut((int)layerConfig.get(LAYER_FIELD_OUTPUT_DIM));
        finishLayerConfig(builder, layerConfig, train);
        return builder.build();
    }

    /**
     * Build DL4J ConvolutionLayer from a Keras *Convolution configuration.
     *
     * @param layerConfig      map containing Keras *Convolution layer properties
     * @param train            whether to build layer in training mode
     * @return                 DL4J ConvolutionLayer configuration
     * @throws UnsupportedOperationException
     * @see ConvolutionLayer
     *
     * TODO: verify whether works for 1D convolutions. What about 3D convolutions?
     */
    private static ConvolutionLayer buildConvolutionLayer(Map<String,Object> layerConfig, boolean train)
            throws UnsupportedKerasConfigurationException {
        List<Integer> stride = (List<Integer>)layerConfig.get(LAYER_FIELD_SUBSAMPLE);
        int nb_row = (Integer)layerConfig.get(LAYER_FIELD_NB_ROW);
        int nb_col = (Integer)layerConfig.get(LAYER_FIELD_NB_COL);
        String borderMode = (String)layerConfig.get(LAYER_FIELD_BORDER_MODE);
        ConvolutionLayer.Builder builder = new ConvolutionLayer.Builder()
                .stride(stride.get(0), stride.get(1))
                .kernelSize(nb_row, nb_col)
                .nOut((int)layerConfig.get(LAYER_FIELD_NB_FILTER));
        switch (borderMode) {
            /* Keras relies upon the Theano and TensorFlow border mode definitions
             * and operations:
             * - Theano: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
             * - TensorFlow: https://www.tensorflow.org/api_docs/python/nn/convolution#conv2d
             */
            case LAYER_BORDER_MODE_SAME:
                /* TensorFlow-only "same" mode is equivalent to DL4J Same mode. */
                builder.convolutionMode(ConvolutionMode.Same);
                break;
            case LAYER_BORDER_MODE_VALID:
                /* TensorFlow and Theano "valid" modes apply filter only
                 * to complete patches within the image borders with no
                 * padding. That is equivalent to DL4J Truncate mode
                 * with no padding.
                 */
                builder.convolutionMode(ConvolutionMode.Truncate);
                break;
            case LAYER_BORDER_MODE_FULL:
                /* Theano-only "full" mode zero pads the image so that
                 * outputs = (inputs + filters + 1) / stride. This should
                 * be equivalent to DL4J Truncate mode with padding
                 * equal to filters-1.
                 * TODO: verify this is correct.
                 */
                int[] padding = new int[]{nb_row-1, nb_col-1};
                builder.convolutionMode(ConvolutionMode.Truncate).padding(padding);
                break;
        }
        finishLayerConfig(builder, layerConfig, train);
        return builder.build();
    }

    /**
     * Build DL4J SubsamplingLayer from a Keras *Pooling* configuration.
     *
     * @param layerConfig      map containing Keras *Pooling* layer properties
     * @param train            whether to build layer in training mode
     * @return                 DL4J SubsamplingLayer configuration
     * @throws UnsupportedOperationException
     * @see SubsamplingLayer
     *
     * TODO: add other pooling layer types and shapes.
     */
    private static SubsamplingLayer buildSubsamplingLayer(Map<String,Object> layerConfig, boolean train)
            throws UnsupportedKerasConfigurationException {
        List<Integer> stride = (List<Integer>)layerConfig.get(LAYER_FIELD_STRIDES);
        List<Integer> pool = (List<Integer>)layerConfig.get(LAYER_FIELD_POOL_SIZE);
        SubsamplingLayer.Builder builder = new SubsamplingLayer.Builder()
                                                .stride(stride.get(0), stride.get(1))
                                                .kernelSize(pool.get(0), pool.get(1));
        String layerClassName = (String)layerConfig.get(LAYER_FIELD_CLASS_NAME);
        switch (layerClassName) {
            case LAYER_CLASS_NAME_MAX_POOLING_2D:
                builder.poolingType(SubsamplingLayer.PoolingType.MAX);
                break;
            case LAYER_CLASS_NAME_AVERAGE_POOLING_2D:
                builder.poolingType(SubsamplingLayer.PoolingType.AVG);
                break;
            /* TODO: 1D (and 3D?) shaped pooling layers. */
            default:
                throw new UnsupportedKerasConfigurationException("Unsupported Keras pooling layer " + layerClassName);
        }
        String borderMode = (String)layerConfig.get(LAYER_FIELD_BORDER_MODE);
        switch (borderMode) {
            /* See notes for "border_mode" in buildConvolutionLayer above. */
            case LAYER_BORDER_MODE_SAME:
                /* TensorFlow-only "same" mode is equivalent to DL4J Same mode. */
                builder.convolutionMode(ConvolutionMode.Same);
                break;
            case LAYER_BORDER_MODE_VALID:
                /* "valid" mode is equivalent to DL4J Truncate mode with no padding. */
                builder.convolutionMode(ConvolutionMode.Truncate);
                break;
            case LAYER_BORDER_MODE_FULL:
                /* Theano-only "full" mode should be equivalent to DL4J Truncate mode
                 * with padding equal to filters-1.
                 * TODO: verify this is correct.
                 */
                int[] padding = new int[]{pool.get(0)-1, pool.get(1)-1};
                builder.convolutionMode(ConvolutionMode.Truncate).padding(padding);
                break;
        }
        finishLayerConfig(builder, layerConfig, train);
        return builder.build();
    }

    /**
     * Build DL4J GravesLSTM layer from a Keras LSTM configuration.
     *
     * @param layerConfig      map containing Keras LSTM layer properties
     * @param train            whether to build layer in training mode
     * @return                 DL4J GravesLSTM configuration
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedOperationException
     * @see GravesLSTM
     */
    private static GravesLSTM buildGravesLstmLayer(Map<String,Object> layerConfig, boolean train)
            throws UnsupportedKerasConfigurationException {
        if (!layerConfig.get(LAYER_FIELD_INIT).equals(layerConfig.get(LAYER_FIELD_INNER_INIT)))
            if (train)
                throw new UnsupportedKerasConfigurationException("Specifying different initialization for LSTM inner cells not supported.");
            else
                log.warn("Specifying different initialization for LSTM inner cells not supported.");
        if ((double)layerConfig.get(LAYER_FIELD_DROPOUT_U) > 0.0)
            throw new UnsupportedKerasConfigurationException("Dropout > 0 on LSTM recurrent connections not supported.");

        GravesLSTM.Builder builder = new GravesLSTM.Builder();
        builder.nOut((int)layerConfig.get(LAYER_FIELD_OUTPUT_DIM));
        builder.gateActivationFunction(mapActivation((String)layerConfig.get(LAYER_FIELD_INNER_ACTIVATION)));
        String forgetBiasInit = (String)layerConfig.get(LAYER_FIELD_FORGET_BIAS_INIT);
        switch (forgetBiasInit) {
            case LSTM_FORGET_BIAS_INIT_ZERO:
                builder.forgetGateBiasInit(0.0);
                break;
            case LSTM_FORGET_BIAS_INIT_ONE:
                builder.forgetGateBiasInit(1.0);
                break;
            default:
                if (train)
                    throw new UnsupportedKerasConfigurationException("Unsupported bias initialization: " + forgetBiasInit);
                else {
                    builder.forgetGateBiasInit(1.0);
                    log.warn("Unsupported bias initialization: " + forgetBiasInit + ". Using ONE instead");
                }
                break;
        }
        layerConfig.put(LAYER_FIELD_DROPOUT, (double)layerConfig.get(LAYER_FIELD_DROPOUT_W));
        finishLayerConfig(builder, layerConfig, train);
        return builder.build();
    }

    /**
     * Build DL4J BatchNormalization layer from a Keras LSTM configuration.
     *
     * @param layerConfig      map containing Keras LSTM layer properties
     * @param train            whether to build layer in training mode
     * @return                 DL4J GravesLSTM configuration
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedOperationException
     * @see BatchNormalization
     */
    private static BatchNormalization buildBatchNormalizationLayer(Map<String,Object> layerConfig, boolean train)
            throws UnsupportedKerasConfigurationException {
        if (train)
            if (layerConfig.get(LAYER_FIELD_GAMMA_REGULARIZER) != null)
                throw new UnsupportedKerasConfigurationException("Regularization for BatchNormalization gamma parameter not supported");
        else
            log.warn("Regularization for BatchNormalization gamma parameter not supported...ignoring.");
        if (train)
            if (layerConfig.get(LAYER_FIELD_BETA_REGULARIZER) != null)
                throw new UnsupportedKerasConfigurationException("Regularization for BatchNormalization beta parameter not supported");
            else
                log.warn("Regularization for BatchNormalization beta parameter not supported...ignoring.");
        int batchNormMode = (int)layerConfig.get(LAYER_FIELD_MODE);
        switch(batchNormMode) {
            case LAYER_BATCHNORM_MODE_1:
                throw new UnsupportedKerasConfigurationException("Keras BatchNormalization mode " + LAYER_BATCHNORM_MODE_1 + " (sample-wise) not supported");
            case LAYER_BATCHNORM_MODE_2:
                throw new UnsupportedKerasConfigurationException("Keras BatchNormalization (per-batch statistics during testing) " + LAYER_BATCHNORM_MODE_2 + " not supported");
        }
        int axis = (int)layerConfig.get(LAYER_FIELD_AXIS);
        log.warn("Ignoring BatchNormalization " + LAYER_FIELD_AXIS + "=" + axis + " config. DL4J BatchNormalization defaults to the \"channels\" axis");
        BatchNormalization.Builder builder = new BatchNormalization.Builder();
        builder.eps((double)layerConfig.get(LAYER_FIELD_EPSILON))
                .momentum((double)layerConfig.get(LAYER_FIELD_MOMENTUM));

        finishLayerConfig(builder, layerConfig, train);
        return builder.build();
    }

    /**
     * Build DL4J LossLayer from a (contrived) Keras layer configuration including a
     * Keras loss function.
     *
     * @param layerConfig   map containing Keras LSTM layer properties
     * @param train         whether to build layer in training mode
     * @return
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedOperationException
     */
    private static LossLayer buildLossLayer(Map<String,Object> layerConfig, boolean train)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        String kerasLoss = (String)checkAndGetField(layerConfig, LAYER_FIELD_LOSS);
        LossFunctions.LossFunction loss;
        try {
            loss = mapLossFunction(kerasLoss);
        } catch (UnsupportedKerasConfigurationException e) {
            if (train)
                throw e;
            log.warn("Unsupported Keras loss function. Replacing with MSE.");
            loss = LossFunctions.LossFunction.SQUARED_LOSS;
        }
        LossLayer.Builder builder = new LossLayer.Builder(loss);
        finishLayerConfig(builder, layerConfig, train);
        return builder.build();
    }

    /**
     * Perform layer configuration steps that are common across all Keras and DL4J layer types.
     *
     * @param builder       DL4J Layer builder object
     * @param layerConfig   map containing Keras layer properties
     * @param train         whether to build layer in training mode
     * @return              DL4J Layer builder object
     * @throws UnsupportedOperationException
     * @see Layer.Builder
     */
    private static Layer.Builder finishLayerConfig(Layer.Builder builder, Map<String,Object> layerConfig, boolean train)
            throws UnsupportedKerasConfigurationException {
        if (layerConfig.containsKey(LAYER_FIELD_DROPOUT)) {
            /* NOTE: Keras "dropout" parameter determines dropout probability,
             * while DL4J "dropout" parameter determines retention probability.
             */
            builder.dropOut(1.0-(double)layerConfig.get(LAYER_FIELD_DROPOUT));
        }
        if (layerConfig.containsKey(LAYER_FIELD_ACTIVATION))
            builder.activation(mapActivation((String)layerConfig.get(LAYER_FIELD_ACTIVATION)));
        builder.name((String)layerConfig.get(LAYER_FIELD_NAME));
        if (layerConfig.containsKey(LAYER_FIELD_INIT)) {
            String kerasInit = (String)layerConfig.get(LAYER_FIELD_INIT);
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
            builder.weightInit(init);
            if (init == WeightInit.ZERO)
                builder.biasInit(0.0);
        }
        if (layerConfig.containsKey(LAYER_FIELD_W_REGULARIZER)) {
            Map<String,Object> regularizerConfig = (Map<String,Object>)layerConfig.get(LAYER_FIELD_W_REGULARIZER);
            double l1 = getL1Regularization(regularizerConfig);
            if (l1 > 0)
                builder.l1(l1);
            double l2 = getL2Regularization(regularizerConfig);
            if (l2 > 0)
                builder.l2(l2);
            checkForUnknownRegularizer(regularizerConfig, train);
        }
        if (layerConfig.containsKey(LAYER_FIELD_B_REGULARIZER)) {
            Map<String,Object> regularizerConfig = (Map<String,Object>)layerConfig.get(LAYER_FIELD_B_REGULARIZER);
            double l1 = getL1Regularization(regularizerConfig);
            double l2 = getL2Regularization(regularizerConfig);
            if (l1 > 0 || l2 > 0)
                if (train)
                    throw new UnsupportedKerasConfigurationException("Bias regularization not implemented");
                else
                    log.warn("Bias regularization not supported. Ignoring.");
        }
        return builder;
    }

    /**
     * Convenience function for checking whether a map contains a key,
     * throwing an error if it does not, and returning the corresponding
     * value if it does. We do this over and over again with the maps
     * created by parsing Keras model configuration JSON strings.
     *
     * @param map   nested (key,value) map of arbitrary depth representing JSON
     * @param key   key to check for in map
     * @return
     */
    private static Object checkAndGetField(Map<String,Object> map, String key) throws InvalidKerasConfigurationException {
        if (!map.containsKey(key))
            throw new InvalidKerasConfigurationException("Field " + key + " missing from layer config");
        return map.get(key);
    }


}
