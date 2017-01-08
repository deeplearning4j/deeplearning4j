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

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.modelimport.keras.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

/**
 * Build Layer from Keras layer configuration.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasLayer {
    /* Keras layer types. */
    public static final String LAYER_FIELD_CLASS_NAME = "class_name";
    public static final String LAYER_CLASS_NAME_ACTIVATION = "Activation";
    public static final String LAYER_CLASS_NAME_INPUT = "Input";
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
    public static final String LAYER_FIELD_BATCH_INPUT_SHAPE = "batch_input_shape";
    public static final String LAYER_FIELD_INBOUND_NODES = "inbound_nodes";
    public static final String LAYER_FIELD_DROPOUT = "dropout";
    public static final String LAYER_FIELD_DROPOUT_W = "dropout_W";
    public static final String LAYER_FIELD_OUTPUT_DIM = "output_dim";
    public static final String LAYER_FIELD_NB_FILTER = "nb_filter";
    public static final String LAYER_FIELD_NB_ROW = "nb_row";
    public static final String LAYER_FIELD_NB_COL = "nb_col";
    public static final String LAYER_FIELD_POOL_SIZE = "pool_size";
    public static final String LAYER_FIELD_SUBSAMPLE = "subsample";
    public static final String LAYER_FIELD_STRIDES = "strides";
    public static final String LAYER_FIELD_BORDER_MODE = "border_mode";

    /* Keras convolution border modes. */
    public static final String LAYER_BORDER_MODE_SAME = "same";
    public static final String LAYER_BORDER_MODE_VALID = "valid";
    public static final String LAYER_BORDER_MODE_FULL = "full";

    /* Keras weight regularizers. */
    public static final String LAYER_FIELD_W_REGULARIZER = "W_regularizer";
    public static final String LAYER_FIELD_B_REGULARIZER = "b_regularizer";
    public static final String REGULARIZATION_TYPE_L1 = "l1";
    public static final String REGULARIZATION_TYPE_L2 = "l2";

    /* Keras weight initializers. */
    public static final String LAYER_FIELD_INIT = "init";
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
    public static final String KERAS_ACTIVATION_SOFTMAX = "softmax";
    public static final String KERAS_ACTIVATION_SOFTPLUS = "softplus";
    public static final String KERAS_ACTIVATION_SOFTSIGN = "softsign";
    public static final String KERAS_ACTIVATION_RELU = "relu";
    public static final String KERAS_ACTIVATION_TANH = "tanh";
    public static final String KERAS_ACTIVATION_SIGMOID = "sigmoid";
    public static final String KERAS_ACTIVATION_HARD_SIGMOID = "hard_sigmoid";
    public static final String KERAS_ACTIVATION_LINEAR = "linear";

    /* Keras dimension ordering for, e.g., convolutional layerNamesOrdered. */
    public static final String LAYER_FIELD_DIM_ORDERING = "dim_ordering";
    public static final String DIM_ORDERING_THEANO = "th";
    public static final String DIM_ORDERING_TENSORFLOW = "tf";

    /* Keras loss functions. */
    public static final String LAYER_CLASS_NAME_LOSS = "Loss"; // Not a Keras layer
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

    /* Keras backends store convolutional inputs and weights
     * in tensors with different dimension orders.
     */
    public enum DimOrder {
        NONE, THEANO, TENSORFLOW;
    }

    protected String className;                     // Keras layer class name
    protected String layerName;                     // Keras layer name
    protected int[] inputShape;                     // Keras layer input shape
    protected DimOrder dimOrder;                    // Keras layer backend dimension order
    protected List<String> inboundLayerNames;       // List of inbound layers
    protected Layer dl4jLayer;                      // Resulting DL4J layer

    /**
     * Build KerasLayer from a Keras layer configuration.
     *
     * @param layerConfig      map containing Keras layer properties
     * @return                 KerasLayer
     * @see Layer
     */
    public static KerasLayer getKerasLayerFromConfig(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return getKerasLayerFromConfig(layerConfig, false);
    }

    /**
     * Build KerasLayer from a Keras layer configuration. Building layer with
     * enforceTrainingConfig=true will throw exceptions for unsupported Keras
     * options related to training (e.g., unknown regularizers). Otherwise
     * we only generate warnings.
     *
     * @param layerConfig      map containing Keras layer properties
     * @param enforceTrainingConfig            build layer in training mode
     * @return                 KerasLayer
     * @see Layer
     */
    public static KerasLayer getKerasLayerFromConfig(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        String layerClassName = getClassNameFromConfig(layerConfig);
        KerasLayer layer = null;
        switch (layerClassName) {
            case LAYER_CLASS_NAME_ACTIVATION:
                layer = new KerasActivationLayer(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_DROPOUT:
                layer = new KerasDropoutLayer(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_DENSE:
            case LAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE:
            /* TODO: test to make sure that mapping TimeDistributedDense to DenseLayer works.
             * Also, Keras recently added support for TimeDistributed layer wrapper so may
             * need to look into how that changes things.
             * */
                layer = new KerasDenseLayer(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_LSTM:
                layer = new KerasLstmLayer(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_CONVOLUTION_2D:
            /* TODO: Add support for 1D, 3D convolutional layerNamesOrdered? */
                layer = new KerasConvolutionLayer(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_MAX_POOLING_2D:
            case LAYER_CLASS_NAME_AVERAGE_POOLING_2D:
            /* TODO: Add support for 1D, 3D pooling layerNamesOrdered? */
                layer = new KerasSubsamplingLayer(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_BATCHNORMALIZATION:
                layer = new KerasBatchNormalizationLayer(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_FLATTEN:
            case LAYER_CLASS_NAME_RESHAPE:
            case LAYER_CLASS_NAME_REPEATVECTOR:
            case LAYER_CLASS_NAME_MERGE:
            case LAYER_CLASS_NAME_INPUT:
            case LAYER_CLASS_NAME_LOSS:
                log.warn("Found Keras " + layerClassName + ". DL4J adds \"preprocessor\" layers during model compilation: https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/MultiLayerConfiguration.java#L429");
                break;
            default:
                throw new InvalidKerasConfigurationException("Unsupported keras layer type " + layerClassName);
        }
        return layer;
    }

    protected KerasLayer() {}

    /**
     * Constructor.
     *
     * @param layerConfig           nested map containing Keras layer configuration
     */
    protected KerasLayer(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor. "enforceTrainingConfig" parameter controls whether layer is built for
     * training. This controls behavior of certain exceptions. In training mode, passing
     * an unsupported regularizer will generate an error. In non-training mode, it
     * generates only a warning.
     *
     * @param layerConfig               nested map containing Keras layer configuration
     * @param enforceTrainingConfig     whether layer should be built for training (controls certain exceptions)
     */
    protected KerasLayer(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this.className = getClassNameFromConfig(layerConfig);
        this.layerName = getLayerNameFromConfig(layerConfig);
        this.inputShape = getInputShapeFromConfig(layerConfig);
        this.dimOrder = getDimOrderFromConfig(layerConfig);
        this.inboundLayerNames = getInboundLayerNamesFromConfig(layerConfig);
        this.dl4jLayer = null;
        checkForUnsupportedConfigurations(layerConfig, enforceTrainingConfig);
    }

    /**
     * Checks whether layer config contains unsupported options.
     *
     * @param layerConfig
     * @param enforceTrainingConfig
     * @throws UnsupportedKerasConfigurationException
     * @throws InvalidKerasConfigurationException
     */
    public static void checkForUnsupportedConfigurations(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        getBiasL1RegularizationFromConfig(layerConfig, enforceTrainingConfig);
        getBiasL2RegularizationFromConfig(layerConfig, enforceTrainingConfig);
        Map<String,Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.containsKey(LAYER_FIELD_W_REGULARIZER))
            checkForUnknownRegularizer((Map<String,Object>)innerConfig.get(LAYER_FIELD_W_REGULARIZER), enforceTrainingConfig);
        if (innerConfig.containsKey(LAYER_FIELD_B_REGULARIZER))
            checkForUnknownRegularizer((Map<String,Object>)innerConfig.get(LAYER_FIELD_B_REGULARIZER), enforceTrainingConfig);
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
    public String getLayerName() {
        return this.layerName;
    }

    /**
     * Get layer input shape.
     *
     * @return      input shape
     */
    public int[] getInputShape() {
        return this.inputShape;
    }

    /**
     * Get Keras layer backend dimension order.
     *
     * @return      Keras layer (backend) dimension order
     */
    public DimOrder getDimOrderFromConfig() {
        return this.dimOrder;
    }

    /**
     * Set Keras layer backend dimension order.
     *
     * @return      Keras layer (backend) dimension order
     */
    public void setDimOrder(DimOrder dimOrder) {
        this.dimOrder = dimOrder;
    }

    /**
     * Get list of inbound layers.
     *
     * @return      list of inbound layer names
     */
    public List<String> getInboundLayerNamesFromConfig() {
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
     * Gets corresponding DL4J Layer, if any.
     *
     * @return      DL4J Layer
     * @see org.deeplearning4j.nn.api.Layer
     */
    public Layer getLayer() {
        return this.dl4jLayer;
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

    public static InputPreProcessor getInputPreProcessor(KerasLayerOld layer, KerasLayerOld prevLayer) {
        InputPreProcessor preprocessor = null;
//        Layer dl4jLayer = layer.getLayer();
//        Layer prevDl4jLayer = prevLayer.getLayer();
//        if (dl4jLayer instanceof FeedForwardLayer)
//            if (prevDl4jLayer instanceof ConvolutionLayer) {
//                int[] inputShape =
//            }

        return preprocessor;
    }

    /**
     * Map Keras to DL4J activation functions.
     *
     * @param kerasActivation   String containing Keras activation function name
     * @return                  String containing DL4J activation function name
     */
    public static IActivation mapActivation(String kerasActivation) throws UnsupportedKerasConfigurationException {
        IActivation dl4jActivation = null;
        /* Keras and DL4J use the same name for most activations. */
        switch (kerasActivation) {
            case KERAS_ACTIVATION_SOFTMAX:
                dl4jActivation = new ActivationSoftmax();
                break;
            case KERAS_ACTIVATION_SOFTPLUS:
                dl4jActivation = new ActivationSoftPlus();
                break;
            case KERAS_ACTIVATION_SOFTSIGN:
                dl4jActivation = new ActivationSoftSign();
                break;
            case KERAS_ACTIVATION_RELU:
                dl4jActivation = new ActivationReLU();
                break;
            case KERAS_ACTIVATION_TANH:
                dl4jActivation = new ActivationTanH();
                break;
            case KERAS_ACTIVATION_SIGMOID:
                dl4jActivation = new ActivationSigmoid();
                break;
            case KERAS_ACTIVATION_HARD_SIGMOID:
                dl4jActivation = new ActivationHardSigmoid();
                break;
            case KERAS_ACTIVATION_LINEAR:
                dl4jActivation = new ActivationIdentity();
                break;
            default:
                throw new UnsupportedKerasConfigurationException("TODO");
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
     * @param kerasLoss    String containing Keras loss function name
     * @return             String containing DL4J loss function
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
     * Map Keras pooling layers to DL4J pooling types.
     *
     * @param className
     * @return
     * @throws UnsupportedKerasConfigurationException
     */
    public static SubsamplingLayer.PoolingType mapPoolingType(String className)
            throws UnsupportedKerasConfigurationException {
        SubsamplingLayer.PoolingType poolingType;
        switch (className) {
            case LAYER_CLASS_NAME_MAX_POOLING_2D:
                poolingType = SubsamplingLayer.PoolingType.MAX;
                break;
            case LAYER_CLASS_NAME_AVERAGE_POOLING_2D:
                poolingType = SubsamplingLayer.PoolingType.AVG;
                break;
            /* TODO: 1D (and 3D?) shaped pooling layers. */
            default:
                throw new UnsupportedKerasConfigurationException("Unsupported Keras pooling layer " + className);
        }
        return poolingType;
    }

    /**
     * Get Keras layer class name from layer config.
     *
     * @param layerConfig
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static String getClassNameFromConfig(Map<String,Object> layerConfig) throws InvalidKerasConfigurationException {
        if (!layerConfig.containsKey(LAYER_FIELD_CLASS_NAME))
            throw new InvalidKerasConfigurationException("Field " + LAYER_FIELD_CLASS_NAME + " missing from layer config");
        return (String)layerConfig.get(LAYER_FIELD_CLASS_NAME);
    }

    /**
     * Get inner layer config from layer config.
     *
     * @param layerConfig
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static Map<String,Object> getInnerLayerConfigFromConfig(Map<String,Object> layerConfig) throws InvalidKerasConfigurationException {
        if (!layerConfig.containsKey(LAYER_FIELD_CONFIG))
            throw new InvalidKerasConfigurationException("Field " + LAYER_FIELD_CONFIG + " missing from layer config");
        return (Map<String,Object>)layerConfig.get(LAYER_FIELD_CONFIG);
    }

    /**
     * Get layer name from layer config.
     *
     * @param layerConfig
     * @return
     * @throws InvalidKerasConfigurationException
     */
    protected String getLayerNameFromConfig(Map<String,Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_NAME))
            throw new InvalidKerasConfigurationException("Field " + LAYER_FIELD_NAME + " missing from layer config");
        return (String)innerConfig.get(LAYER_FIELD_NAME);
    }

    /**
     * Get Keras input shape from Keras layer configuration.
     *
     * @param layerConfig       Keras layer configuration
     * @return                  input shape array
     */
    private int[] getInputShapeFromConfig(Map<String,Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_BATCH_INPUT_SHAPE))
            return null;
        List<Integer> batchInputShape = (List<Integer>)innerConfig.get(LAYER_FIELD_BATCH_INPUT_SHAPE);
        int[] inputShape = new int[batchInputShape.size()-1];
        for (int i = 1; i < batchInputShape.size(); i++) {
            inputShape[i - 1] = batchInputShape.get(i) != null ? batchInputShape.get(i) : 0;
        }
        return inputShape;
    }

    /**
     * Get Keras (backend) dimension order from Keras layer configuration.
     *
     * @param layerConfig       Keras layer configuration
     * @return                  Dimension order
     */
    private DimOrder getDimOrderFromConfig(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        DimOrder dimOrder = DimOrder.NONE;
        if (innerConfig.containsKey(LAYER_FIELD_DIM_ORDERING)) {
            String dimOrderStr = (String)innerConfig.get(LAYER_FIELD_DIM_ORDERING);
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
     * Get list of inbound layers from Keras layer configuration.
     *
     * @param layerConfig       Keras layer configuration
     * @return                  List of inbound layer names
     */
    public static List<String> getInboundLayerNamesFromConfig(Map<String,Object> layerConfig) {
        List<String> inboundLayerNames = new ArrayList<>();
        if (layerConfig.containsKey(LAYER_FIELD_INBOUND_NODES)) {
            List<Object> inboundNodes = (List<Object>)layerConfig.get(LAYER_FIELD_INBOUND_NODES);
            if (inboundNodes.size() > 0) {
                inboundNodes = (List<Object>)inboundNodes.get(0);
                for (Object o : inboundNodes) {
                    String nodeName = (String)((List<Object>)o).get(0);
                    inboundLayerNames.add(nodeName);
                }
            }
        }
        return inboundLayerNames;
    }

    /**
     * Get number of outputs from layer config.
     *
     * @param layerConfig
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static int getNOutFromConfig(Map<String,Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        int nOut;
        if (innerConfig.containsKey(LAYER_FIELD_OUTPUT_DIM))
            /* Most feedforward layers: Dense, RNN, etc. */
            nOut = (int)innerConfig.get(LAYER_FIELD_OUTPUT_DIM);
        else if (innerConfig.containsKey(LAYER_FIELD_NB_FILTER))
            /* Convolutional layers. */
            nOut = (int)innerConfig.get(LAYER_FIELD_NB_FILTER);
        else
            throw new InvalidKerasConfigurationException("TODO");
        return nOut;
    }

    /**
     * Get dropout from layer config.
     *
     * @param layerConfig
     * @return
     * @throws InvalidKerasConfigurationException
     */
    protected double getDropoutFromConfig(Map<String,Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        /* NOTE: Keras "dropout" parameter determines dropout probability,
         * while DL4J "dropout" parameter determines retention probability.
         */
        double dropout = 1.0;
        if (innerConfig.containsKey(LAYER_FIELD_DROPOUT)) {
            /* For most feedforward layers. */
            dropout = 1.0 - (double) innerConfig.get(LAYER_FIELD_DROPOUT);
        } else if (layerConfig.containsKey(LAYER_FIELD_DROPOUT_W)) {
            /* For LSTMs. */
            dropout = 1.0 - (double) layerConfig.get(LAYER_FIELD_DROPOUT_W);
        }
        return dropout;
    }

    /**
     * Get activation function from layer config.
     *
     * @param layerConfig
     * @return
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    protected IActivation getActivationFromConfig(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_ACTIVATION))
            throw new InvalidKerasConfigurationException("TODO");
        return mapActivation((String)innerConfig.get(LAYER_FIELD_ACTIVATION));
    }

    /**
     * Get weight initialization from layer config.
     *
     * @param layerConfig
     * @param enforceTrainingConfig
     * @return
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    protected WeightInit getWeightInitFromConfig(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_INIT))
            throw new InvalidKerasConfigurationException("TODO");
        String kerasInit = (String) innerConfig.get(LAYER_FIELD_INIT);
        WeightInit init;
        try {
            init = mapWeightInitialization(kerasInit);
        } catch (UnsupportedKerasConfigurationException e) {
            if (enforceTrainingConfig)
                throw e;
            else {
                init = WeightInit.XAVIER;
                log.warn("Unknown weight initializer " + kerasInit + " (Using XAVIER instead).");
            }
        }
        return init;
    }

    /**
     * Get L1 weight regularization (if any) from Keras weight regularization configuration.
     *
     * @param layerConfig     Map containing Keras weight reguarlization configuration
     * @return                L1 regularization strength (0.0 if none)
     */
    public static double getWeightL1RegularizationFromConfig(Map<String,Object> layerConfig, boolean willBeTrained)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.containsKey(LAYER_FIELD_W_REGULARIZER)) {
            Map<String, Object> regularizerConfig = (Map<String, Object>)innerConfig.get(LAYER_FIELD_W_REGULARIZER);
            if (regularizerConfig.containsKey(REGULARIZATION_TYPE_L1))
                return (double) regularizerConfig.get(REGULARIZATION_TYPE_L1);
        }
        return 0.0;
    }

    /**
     * Get L2 weight regularization (if any) from Keras weight regularization configuration.
     *
     * @param layerConfig     Map containing Keras weight reguarlization configuration
     * @return                L1 regularization strength (0.0 if none)
     */
    public static double getWeightL2RegularizationFromConfig(Map<String,Object> layerConfig, boolean willBeTrained)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.containsKey(LAYER_FIELD_W_REGULARIZER)) {
            Map<String, Object> regularizerConfig = (Map<String, Object>)innerConfig.get(LAYER_FIELD_W_REGULARIZER);
            if (regularizerConfig.containsKey(REGULARIZATION_TYPE_L2))
                return (double) regularizerConfig.get(REGULARIZATION_TYPE_L2);
        }
        return 0.0;
    }

    /**
     * Get L1 bias regularization (if any) from Keras bias regularization configuration.
     *
     * @param layerConfig     Map containing Keras bias reguarlization configuration
     * @return                L1 regularization strength (0.0 if none)
     */
    public static double getBiasL1RegularizationFromConfig(Map<String,Object> layerConfig, boolean willBeTrained)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.containsKey(LAYER_FIELD_B_REGULARIZER)) {
            Map<String, Object> regularizerConfig = (Map<String, Object>)innerConfig.get(LAYER_FIELD_B_REGULARIZER);
            if (regularizerConfig.containsKey(REGULARIZATION_TYPE_L1))
                throw new UnsupportedKerasConfigurationException("TODO");
        }
        return 0.0;
    }

    /**
     * Get L2 bias regularization (if any) from Keras bias regularization configuration.
     *
     * @param layerConfig     Map containing Keras weight reguarlization configuration
     * @return                L1 regularization strength (0.0 if none)
     */
    private static double getBiasL2RegularizationFromConfig(Map<String,Object> layerConfig, boolean willBeTrained)
            throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.containsKey(LAYER_FIELD_B_REGULARIZER)) {
            Map<String, Object> regularizerConfig = (Map<String, Object>)innerConfig.get(LAYER_FIELD_B_REGULARIZER);
            if (regularizerConfig.containsKey(REGULARIZATION_TYPE_L2))
                throw new UnsupportedKerasConfigurationException("TODO");
        }
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
    private static void checkForUnknownRegularizer(Map<String, Object> regularizerConfig, boolean enforceTrainingConfig)
        throws UnsupportedKerasConfigurationException {
        if (regularizerConfig != null) {
            for (String field : regularizerConfig.keySet()) {
                if (!field.equals(REGULARIZATION_TYPE_L1) && !field.equals(REGULARIZATION_TYPE_L2) && !field.equals(LAYER_FIELD_NAME)) {
                    if (enforceTrainingConfig)
                        throw new UnsupportedKerasConfigurationException("Unknown regularization field " + field);
                    else
                        log.warn("Ignoring unknown regularization field " + field);
                }
            }
        }
    }

    public static int[] getStrideFromConfig(Map<String,Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        int[] strides = null;
        if (innerConfig.containsKey(LAYER_FIELD_SUBSAMPLE)) {
            /* Convolutional layers. */
            List<Integer> stridesList = (List<Integer>)innerConfig.get(LAYER_FIELD_SUBSAMPLE);
            strides = ArrayUtil.toArray(stridesList);
        } else if (innerConfig.containsKey(LAYER_FIELD_STRIDES)) {
            /* Pooling layers. */
            List<Integer> stridesList = (List<Integer>)innerConfig.get(LAYER_FIELD_STRIDES);
            strides = ArrayUtil.toArray(stridesList);
        } else
            throw new InvalidKerasConfigurationException("TODO");
        return strides;
    }

    public static int[] getKernelSizeFromConfig(Map<String,Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        int[] kernelSize = null;
        if (innerConfig.containsKey(LAYER_FIELD_NB_ROW) && innerConfig.containsKey(LAYER_FIELD_NB_COL)) {
            /* Convolutional layers. */
            List<Integer> kernelSizeList = new ArrayList<Integer>();
            kernelSizeList.add((Integer)innerConfig.get(LAYER_FIELD_NB_ROW));
            kernelSizeList.add((Integer)innerConfig.get(LAYER_FIELD_NB_COL));
            kernelSize = ArrayUtil.toArray(kernelSizeList);
        } else if (innerConfig.containsKey(LAYER_FIELD_POOL_SIZE)) {
            /* Pooling layers. */
            List<Integer> kernelSizeList = (List<Integer>)innerConfig.get(LAYER_FIELD_POOL_SIZE);
            kernelSize = ArrayUtil.toArray(kernelSizeList);
        } else
            throw new InvalidKerasConfigurationException("TODO");
        return kernelSize;
    }

    public static ConvolutionMode getConvolutionModeFromConfig(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_BORDER_MODE))
            throw new InvalidKerasConfigurationException("TODO");
        String borderMode = (String)innerConfig.get(LAYER_FIELD_BORDER_MODE);
        ConvolutionMode convolutionMode = null;
        switch (borderMode) {
            /* Keras relies upon the Theano and TensorFlow border mode definitions
             * and operations:
             * - Theano: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
             * - TensorFlow: https://www.tensorflow.org/api_docs/python/nn/convolution#conv2d
             */
            case LAYER_BORDER_MODE_SAME:
                /* TensorFlow-only "same" mode is equivalent to DL4J Same mode. */
                convolutionMode = ConvolutionMode.Same;
                break;
            case LAYER_BORDER_MODE_VALID:
                /* TensorFlow and Theano "valid" modes apply filter only
                 * to complete patches within the image borders with no
                 * padding. That is equivalent to DL4J Truncate mode
                 * with no padding.
                 */
            case LAYER_BORDER_MODE_FULL:
                /* Theano-only "full" mode zero pads the image so that
                 * outputs = (inputs + filters + 1) / stride. This should
                 * be equivalent to DL4J Truncate mode with padding
                 * equal to filters-1.
                 * TODO: verify this is correct.
                 */
                convolutionMode = ConvolutionMode.Truncate;
                break;
            default:
                throw new UnsupportedKerasConfigurationException("TODO");
        }
        return convolutionMode;
    }

    public int[] getPaddingFromConfig(Map<String,Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        int[] padding = null;
        if (!innerConfig.containsKey(LAYER_FIELD_BORDER_MODE))
            throw new InvalidKerasConfigurationException("TODO");
        String borderMode = (String)innerConfig.get(LAYER_FIELD_BORDER_MODE);
        if (borderMode == LAYER_FIELD_BORDER_MODE) {
            padding = getKernelSizeFromConfig(layerConfig);
            for (int i = 0; i < padding.length; i++)
                padding[i]--;
        }
        return padding;
    }
}
