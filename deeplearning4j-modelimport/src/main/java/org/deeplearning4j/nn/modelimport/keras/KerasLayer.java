/*-
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
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasMerge;
import org.deeplearning4j.nn.modelimport.keras.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.ArrayUtil;

import java.lang.reflect.Constructor;
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
    public static final String LAYER_CLASS_NAME_INPUT = "InputLayer";
    public static final String LAYER_CLASS_NAME_DROPOUT = "Dropout";
    public static final String LAYER_CLASS_NAME_DENSE = "Dense";
    public static final String LAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE = "TimeDistributedDense";
    public static final String LAYER_CLASS_NAME_LSTM = "LSTM";
    public static final String LAYER_CLASS_NAME_CONVOLUTION_1D = "Convolution1D";
    public static final String LAYER_CLASS_NAME_CONVOLUTION_2D = "Convolution2D";
    public static final String LAYER_CLASS_NAME_MAX_POOLING_1D = "MaxPooling1D";
    public static final String LAYER_CLASS_NAME_MAX_POOLING_2D = "MaxPooling2D";
    public static final String LAYER_CLASS_NAME_AVERAGE_POOLING_1D = "AveragePooling1D";
    public static final String LAYER_CLASS_NAME_AVERAGE_POOLING_2D = "AveragePooling2D";
    public static final String LAYER_CLASS_NAME_ZERO_PADDING_1D = "ZeroPadding1D";
    public static final String LAYER_CLASS_NAME_ZERO_PADDING_2D = "ZeroPadding2D";
    public static final String LAYER_CLASS_NAME_FLATTEN = "Flatten";
    public static final String LAYER_CLASS_NAME_MERGE = "Merge";
    public static final String LAYER_CLASS_NAME_BATCHNORMALIZATION = "BatchNormalization";
    public static final String LAYER_CLASS_NAME_TIME_DISTRIBUTED = "TimeDistributed";
    public static final String LAYER_CLASS_NAME_EMBEDDING = "Embedding";
    public static final String LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_1D = "GlobalMaxPooling1D";
    public static final String LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_2D = "GlobalMaxPooling2D";
    public static final String LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_1D = "GlobalAveragePooling1D";
    public static final String LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_2D = "GlobalAveragePooling2D";

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

    /* Keras dimension ordering for, e.g., convolutional layersOrdered. */
    public static final String LAYER_FIELD_DIM_ORDERING = "dim_ordering";
    public static final String DIM_ORDERING_THEANO = "th";
    public static final String DIM_ORDERING_TENSORFLOW = "tf";

    /* Keras loss functions. */
    public static final String KERAS_LOSS_MEAN_SQUARED_ERROR = "mean_squared_error";
    public static final String KERAS_LOSS_MSE = "mse";
    public static final String KERAS_LOSS_MEAN_ABSOLUTE_ERROR = "mean_absolute_error";
    public static final String KERAS_LOSS_MAE = "mae";
    public static final String KERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mean_absolute_percentage_error";
    public static final String KERAS_LOSS_MAPE = "mape";
    public static final String KERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR = "mean_squared_logarithmic_error";
    public static final String KERAS_LOSS_MSLE = "msle";
    public static final String KERAS_LOSS_SQUARED_HINGE = "squared_hinge";
    public static final String KERAS_LOSS_HINGE = "hinge";
    public static final String KERAS_LOSS_BINARY_CROSSENTROPY = "binary_crossentropy";
    public static final String KERAS_LOSS_CATEGORICAL_CROSSENTROPY = "categorical_crossentropy";
    public static final String KERAS_LOSS_SPARSE_CATEGORICAL_CROSSENTROPY = "sparse_categorical_crossentropy";
    public static final String KERAS_LOSS_KULLBACK_LEIBLER_DIVERGENCE = "kullback_leibler_divergence";
    public static final String KERAS_LOSS_KLD = "kld";
    public static final String KERAS_LOSS_POISSON = "poisson";
    public static final String KERAS_LOSS_COSINE_PROXIMITY = "cosine_proximity";
    public static final String LAYER_FIELD_LAYER = "layer";

    public static final Map<String, Class<? extends KerasLayer>> customLayers = new HashMap<>();

    /* Keras backends store convolutional inputs and weights
     * in tensors with different dimension orders.
     */
    public enum DimOrder {
        NONE, THEANO, TENSORFLOW;
    }

    protected String className; // Keras layer class name
    protected String layerName; // Keras layer name
    protected int[] inputShape; // Keras layer input shape
    protected DimOrder dimOrder; // Keras layer backend dimension order
    protected List<String> inboundLayerNames; // List of inbound layers
    protected Layer layer; // Resulting DL4J layer
    protected GraphVertex vertex; // Resulting DL4J vertex
    protected Map<String, INDArray> weights; // Weights
    protected double weightL1Regularization = 0.0; // L1 regularization
    protected double weightL2Regularization = 0.0; // L2 regularization
    protected double dropout = 1.0; // Dropout

    /**
     * Build KerasLayer from a Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration      map containing Keras layer properties
     * @return                 KerasLayer
     * @see Layer
     */
    public static KerasLayer getKerasLayerFromConfig(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return getKerasLayerFromConfig(layerConfig, false);
    }

    /**
     * Build KerasLayer from a Keras layer configuration. Building layer with
     * enforceTrainingConfig=true will throw exceptions for unsupported Keras
     * options related to training (e.g., unknown regularizers). Otherwise
     * we only generate warnings.
     *
     * @param layerConfig       dictionary containing Keras layer configuration               map containing Keras layer properties
     * @param enforceTrainingConfig     whether to enforce training-only configurations
     * @return                 KerasLayer
     * @see Layer
     */
    public static KerasLayer getKerasLayerFromConfig(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        String layerClassName = getClassNameFromConfig(layerConfig);
        if (layerClassName.equals(LAYER_CLASS_NAME_TIME_DISTRIBUTED)) {
            layerConfig = getTimeDistributedLayerConfig(layerConfig);
            layerClassName = getClassNameFromConfig(layerConfig);
        }

        KerasLayer layer = null;
        switch (layerClassName) {
            case LAYER_CLASS_NAME_ACTIVATION:
                layer = new KerasActivation(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_DROPOUT:
                layer = new KerasDropout(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_DENSE:
            case LAYER_CLASS_NAME_TIME_DISTRIBUTED_DENSE:
                /* TODO: test to make sure that mapping TimeDistributedDense to DenseLayer works.
                 * Also, Keras recently added support for TimeDistributed layer wrapper so may
                 * need to look into how that changes things.
                 * */
                layer = new KerasDense(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_LSTM:
                layer = new KerasLstm(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_CONVOLUTION_2D:
                /* TODO: Add support for 1D, 3D convolutional layersOrdered? */
                layer = new KerasConvolution(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_MAX_POOLING_2D:
            case LAYER_CLASS_NAME_AVERAGE_POOLING_2D:
                /* TODO: Add support for 1D, 3D pooling layersOrdered? */
                layer = new KerasPooling(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_1D:
            case LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_2D:
            case LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_1D:
            case LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_2D:
                layer = new KerasGlobalPooling(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_BATCHNORMALIZATION:
                layer = new KerasBatchNormalization(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_EMBEDDING:
                layer = new KerasEmbedding(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_INPUT:
                layer = new KerasInput(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_MERGE:
                layer = new KerasMerge(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_FLATTEN:
                layer = new KerasFlatten(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_ZERO_PADDING_2D:
                layer = new KerasZeroPadding(layerConfig, enforceTrainingConfig);
                break;
            case LAYER_CLASS_NAME_CONVOLUTION_1D:
            case LAYER_CLASS_NAME_MAX_POOLING_1D:
            case LAYER_CLASS_NAME_AVERAGE_POOLING_1D:
            case LAYER_CLASS_NAME_ZERO_PADDING_1D:
            default:
                // check if user registered a custom config
                Class<? extends KerasLayer> customConfig = customLayers.get(layerClassName);

                if(customConfig == null)
                    throw new UnsupportedKerasConfigurationException("Unsupported keras layer type " + layerClassName);
                try {
                    Constructor constructor = customConfig.getConstructor(Map.class);
                    layer = (KerasLayer) constructor.newInstance(layerConfig);
                } catch(Exception e) {
                    throw new RuntimeException(e);
                }
                break;
        }
        return layer;
    }

    public static void registerCustomLayer(String layerName, Class<? extends KerasLayer> configClass) {
        customLayers.put(layerName, configClass);
    }

    protected KerasLayer() {
        this.className = null;
        this.layerName = null;
        this.inputShape = null;
        this.dimOrder = DimOrder.NONE;
        this.inboundLayerNames = new ArrayList<String>();
        this.layer = null;
        this.vertex = null;
        this.weights = null;
    }

    /**
     * Constructor.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     */
    protected KerasLayer(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor. "enforceTrainingConfig" parameter controls whether layer is built for
     * training. This controls behavior of certain exceptions. In training mode, passing
     * an unsupported regularizer will generate an error. In non-training mode, it
     * generates only a warning.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @param enforceTrainingConfig     whether layer should be built for training (controls certain exceptions)
     */
    protected KerasLayer(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this.className = getClassNameFromConfig(layerConfig);
        if (this.className == null)
            throw new InvalidKerasConfigurationException("Keras layer class name is missing");
        this.layerName = getLayerNameFromConfig(layerConfig);
        if (this.layerName == null)
            throw new InvalidKerasConfigurationException("Keras layer class name is missing");
        this.inputShape = getInputShapeFromConfig(layerConfig);
        this.dimOrder = getDimOrderFromConfig(layerConfig);
        this.inboundLayerNames = getInboundLayerNamesFromConfig(layerConfig);
        this.layer = null;
        this.vertex = null;
        this.weights = null;
        this.weightL1Regularization = getWeightL1RegularizationFromConfig(layerConfig, enforceTrainingConfig);
        this.weightL2Regularization = getWeightL2RegularizationFromConfig(layerConfig, enforceTrainingConfig);
        this.dropout = getDropoutFromConfig(layerConfig);
        checkForUnsupportedConfigurations(layerConfig, enforceTrainingConfig);
    }

    /**
     * Checks whether layer config contains unsupported options.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @param enforceTrainingConfig
     * @throws UnsupportedKerasConfigurationException
     * @throws InvalidKerasConfigurationException
     */
    public static void checkForUnsupportedConfigurations(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        getBiasL1RegularizationFromConfig(layerConfig, enforceTrainingConfig);
        getBiasL2RegularizationFromConfig(layerConfig, enforceTrainingConfig);
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.containsKey(LAYER_FIELD_W_REGULARIZER))
            checkForUnknownRegularizer((Map<String, Object>) innerConfig.get(LAYER_FIELD_W_REGULARIZER),
                            enforceTrainingConfig);
        if (innerConfig.containsKey(LAYER_FIELD_B_REGULARIZER))
            checkForUnknownRegularizer((Map<String, Object>) innerConfig.get(LAYER_FIELD_B_REGULARIZER),
                            enforceTrainingConfig);
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
        if (this.inputShape == null)
            return null;
        return this.inputShape.clone();
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
    public List<String> getInboundLayerNames() {
        if (this.inboundLayerNames == null)
            this.inboundLayerNames = new ArrayList<String>();
        return this.inboundLayerNames;
    }

    /**
     * Set list of inbound layers.
     *
     * @param   inboundLayerNames   list of inbound layer naems
     * @return
     */
    public void setInboundLayerNames(List<String> inboundLayerNames) {
        this.inboundLayerNames = new ArrayList<String>(inboundLayerNames);
    }

    /**
     * Returns number of trainable parameters in layer.
     *
     * @return          number of trainable parameters
     */
    public int getNumParams() {
        return 0;
    }

    /**
     * Indicates whether layer uses regularization.
     *
     * @return  boolean
     */
    public boolean usesRegularization() {
        return (this.weightL1Regularization > 0.0 || this.weightL2Regularization > 0.0 || this.dropout < 1.0);
    }

    /**
     * Set weights for Keras layer.
     *
     * @param weights
     */
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        //no op
    }

    /**
     * Copy Keras layer weights to DL4J Layer.
     *
     * @param layer
     * @throws InvalidKerasConfigurationException
     */
    public void copyWeightsToLayer(org.deeplearning4j.nn.api.Layer layer) throws InvalidKerasConfigurationException {
        if (this.getNumParams() > 0) {
            String dl4jLayerName = layer.conf().getLayer().getLayerName();
            String kerasLayerName = this.getLayerName();
            String msg = "Error when attempting to copy weights from Keras layer " + kerasLayerName + " to DL4J layer "
                            + dl4jLayerName;

            if (this.weights == null)
                throw new InvalidKerasConfigurationException(msg + "(weights is null)");

            Set<String> paramsInLayer = new HashSet<String>(layer.paramTable().keySet());
            Set<String> paramsInKerasLayer = new HashSet<String>(this.weights.keySet());

            /* Check for parameters in layer for which we don't have weights. */
            paramsInLayer.removeAll(paramsInKerasLayer);
            for (String paramName : paramsInLayer)
                throw new InvalidKerasConfigurationException(
                                msg + "(no stored weights for parameter " + paramName + ")");

            /* Check for parameters NOT in layer for which we DO have weights. */
            paramsInKerasLayer.removeAll(layer.paramTable().keySet());
            for (String paramName : paramsInKerasLayer)
                throw new InvalidKerasConfigurationException(msg + "(found no parameter named " + paramName + ")");

            /* Copy weights. */
            for (String paramName : layer.paramTable().keySet())
                layer.setParam(paramName, this.weights.get(paramName));
        }
    }

    /**
     * Whether this Keras layer maps to a DL4J Layer.
     *
     * @return      true or false
     */
    public boolean isLayer() {
        return this.layer != null;
    }

    /**
     * Gets corresponding DL4J Layer, if any.
     *
     * @return      DL4J Layer
     * @see org.deeplearning4j.nn.api.Layer
     */
    public Layer getLayer() {
        return this.layer;
    }

    /**
     * Whether this Keras layer maps to a DL4J Vertex.
     *
     * @return      true or false
     */
    public boolean isVertex() {
        return this.vertex != null;
    }

    /**
     * Gets corresponding DL4J Vertex, if any.
     *
     * @return      DL4J Vertex
     * @see org.deeplearning4j.nn.conf.graph.GraphVertex
     */
    public GraphVertex getVertex() {
        return this.vertex;
    }

    /**
     * Whether this Keras layer maps to a DL4J InputPreProcessor.
     *
     * @return      true or false
     */
    public boolean isInputPreProcessor() {
        return false;
    }

    /**
     * Gets appropriate DL4J InputPreProcessor for given InputTypes.
     *
     * @param  inputType    Array of InputTypes
     * @return              DL4J InputPreProcessor
     * @throws InvalidKerasConfigurationException
     * @see org.deeplearning4j.nn.conf.InputPreProcessor
     */
    public InputPreProcessor getInputPreprocessor(InputType... inputType) throws InvalidKerasConfigurationException {
        InputPreProcessor preprocessor = null;
        if (this.layer != null) {
            if (inputType.length > 1)
                throw new InvalidKerasConfigurationException(
                                "Keras layer of type \"" + this.className + "\" accepts only one input");
            preprocessor = this.layer.getPreProcessorForInputType(inputType[0]);
        }
        return preprocessor;
    }

    /**
     * Get layer output type.
     *
     * @param  inputType    Array of InputTypes
     * @return              output type as InputType
     * @throws InvalidKerasConfigurationException
     */
    public InputType getOutputType(InputType... inputType)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        throw new UnsupportedOperationException(
                        "Cannot determine output type for Keras layer of type " + this.className);
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
    public boolean isValidInboundLayer() throws InvalidKerasConfigurationException {
        return (getLayer() != null || getVertex() != null || getInputPreprocessor() != null
                        || this.className.equals(LAYER_CLASS_NAME_INPUT));
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
                throw new UnsupportedKerasConfigurationException(
                                "Unknown Keras activation function " + kerasActivation);
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
                    throw new UnsupportedKerasConfigurationException("Unknown keras weight initializer " + kerasInit);
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
            case KERAS_LOSS_MEAN_SQUARED_ERROR:
            case KERAS_LOSS_MSE:
                dl4jLoss = LossFunctions.LossFunction.SQUARED_LOSS;
                break;
            case KERAS_LOSS_MEAN_ABSOLUTE_ERROR:
            case KERAS_LOSS_MAE:
                dl4jLoss = LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR;
                break;
            case KERAS_LOSS_MEAN_ABSOLUTE_PERCENTAGE_ERROR:
            case KERAS_LOSS_MAPE:
                dl4jLoss = LossFunctions.LossFunction.MEAN_ABSOLUTE_PERCENTAGE_ERROR;
                break;
            case KERAS_LOSS_MEAN_SQUARED_LOGARITHMIC_ERROR:
            case KERAS_LOSS_MSLE:
                dl4jLoss = LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR;
                break;
            case KERAS_LOSS_SQUARED_HINGE:
                dl4jLoss = LossFunctions.LossFunction.SQUARED_HINGE;
                break;
            case KERAS_LOSS_HINGE:
                dl4jLoss = LossFunctions.LossFunction.HINGE;
                break;
            case KERAS_LOSS_BINARY_CROSSENTROPY:
                dl4jLoss = LossFunctions.LossFunction.XENT;
                break;
            case KERAS_LOSS_SPARSE_CATEGORICAL_CROSSENTROPY:
                /* TODO: should this be an error instead? */
                log.warn("Sparse cross entropy not implemented, using multiclass cross entropy instead.");
            case KERAS_LOSS_CATEGORICAL_CROSSENTROPY:
                dl4jLoss = LossFunctions.LossFunction.MCXENT;
                break;
            case KERAS_LOSS_KULLBACK_LEIBLER_DIVERGENCE:
            case KERAS_LOSS_KLD:
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
    public static PoolingType mapPoolingType(String className) throws UnsupportedKerasConfigurationException {
        PoolingType poolingType;
        switch (className) {
            case LAYER_CLASS_NAME_MAX_POOLING_2D:
            case LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_1D:
            case LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_2D:
                poolingType = PoolingType.MAX;
                break;
            case LAYER_CLASS_NAME_AVERAGE_POOLING_2D:
            case LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_1D:
            case LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_2D:
                poolingType = PoolingType.AVG;
                break;
            /* TODO: 1D (and 3D?) shaped pooling layers. */
            default:
                throw new UnsupportedKerasConfigurationException("Unsupported Keras pooling layer " + className);
        }
        return poolingType;
    }

    /**
     * Map Keras pooling layers to DL4J pooling dimensions.
     *
     * @param className
     * @return
     * @throws UnsupportedKerasConfigurationException
     */
    public static int[] mapPoolingDimensions(String className) throws UnsupportedKerasConfigurationException {
        int[] dimensions;
        switch (className) {
            case LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_1D:
            case LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_1D:
                dimensions = new int[] {2};
                break;
            case LAYER_CLASS_NAME_GLOBAL_MAX_POOLING_2D:
            case LAYER_CLASS_NAME_GLOBAL_AVERAGE_POOLING_2D:
                dimensions = new int[] {2, 3};
                break;
            default:
                throw new UnsupportedKerasConfigurationException("Unsupported Keras pooling layer " + className);
        }
        return dimensions;
    }

    /**
     * Get Keras layer class name from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static String getClassNameFromConfig(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException {
        if (!layerConfig.containsKey(LAYER_FIELD_CLASS_NAME))
            throw new InvalidKerasConfigurationException(
                            "Field " + LAYER_FIELD_CLASS_NAME + " missing from layer config");
        return (String) layerConfig.get(LAYER_FIELD_CLASS_NAME);
    }

    /**
     * Extract inner layer config from TimeDistributed configuration and merge
     * it into the outer config.
     *
     * @param layerConfig       dictionary containing Keras TimeDistributed configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static Map<String, Object> getTimeDistributedLayerConfig(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException {
        if (!layerConfig.containsKey(LAYER_FIELD_CLASS_NAME))
            throw new InvalidKerasConfigurationException(
                            "Field " + LAYER_FIELD_CLASS_NAME + " missing from layer config");
        if (!layerConfig.get(LAYER_FIELD_CLASS_NAME).equals(LAYER_CLASS_NAME_TIME_DISTRIBUTED))
            throw new InvalidKerasConfigurationException("Expected " + LAYER_CLASS_NAME_TIME_DISTRIBUTED
                            + " layer, found " + (String) layerConfig.get(LAYER_FIELD_CLASS_NAME));
        if (!layerConfig.containsKey(LAYER_FIELD_CONFIG))
            throw new InvalidKerasConfigurationException("Field " + LAYER_FIELD_CONFIG + " missing from layer config");
        Map<String, Object> outerConfig = getInnerLayerConfigFromConfig(layerConfig);
        Map<String, Object> innerLayer = (Map<String, Object>) outerConfig.get(LAYER_FIELD_LAYER);
        layerConfig.put(LAYER_FIELD_CLASS_NAME, innerLayer.get(LAYER_FIELD_CLASS_NAME));
        layerConfig.put(LAYER_FIELD_NAME, innerLayer.get(LAYER_FIELD_CLASS_NAME));
        Map<String, Object> innerConfig = (Map<String, Object>) getInnerLayerConfigFromConfig(innerLayer);
        outerConfig.putAll(innerConfig);
        outerConfig.remove(LAYER_FIELD_LAYER);
        return layerConfig;
    }

    /**
     * Get inner layer config from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static Map<String, Object> getInnerLayerConfigFromConfig(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException {
        if (!layerConfig.containsKey(LAYER_FIELD_CONFIG))
            throw new InvalidKerasConfigurationException("Field " + LAYER_FIELD_CONFIG + " missing from layer config");
        return (Map<String, Object>) layerConfig.get(LAYER_FIELD_CONFIG);
    }

    /**
     * Get layer name from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    protected String getLayerNameFromConfig(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_NAME))
            throw new InvalidKerasConfigurationException("Field " + LAYER_FIELD_NAME + " missing from layer config");
        return (String) innerConfig.get(LAYER_FIELD_NAME);
    }

    /**
     * Get Keras input shape from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return                  input shape array
     */
    private int[] getInputShapeFromConfig(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_BATCH_INPUT_SHAPE))
            return null;
        List<Integer> batchInputShape = (List<Integer>) innerConfig.get(LAYER_FIELD_BATCH_INPUT_SHAPE);
        int[] inputShape = new int[batchInputShape.size() - 1];
        for (int i = 1; i < batchInputShape.size(); i++) {
            inputShape[i - 1] = batchInputShape.get(i) != null ? batchInputShape.get(i) : 0;
        }
        return inputShape;
    }

    /**
     * Get Keras (backend) dimension order from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return                  Dimension order
     */
    private DimOrder getDimOrderFromConfig(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        DimOrder dimOrder = DimOrder.NONE;
        if (innerConfig.containsKey(LAYER_FIELD_DIM_ORDERING)) {
            String dimOrderStr = (String) innerConfig.get(LAYER_FIELD_DIM_ORDERING);
            switch (dimOrderStr) {
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
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return                  List of inbound layer names
     */
    public static List<String> getInboundLayerNamesFromConfig(Map<String, Object> layerConfig) {
        List<String> inboundLayerNames = new ArrayList<>();
        if (layerConfig.containsKey(LAYER_FIELD_INBOUND_NODES)) {
            List<Object> inboundNodes = (List<Object>) layerConfig.get(LAYER_FIELD_INBOUND_NODES);
            if (inboundNodes.size() > 0) {
                inboundNodes = (List<Object>) inboundNodes.get(0);
                for (Object o : inboundNodes) {
                    String nodeName = (String) ((List<Object>) o).get(0);
                    inboundLayerNames.add(nodeName);
                }
            }
        }
        return inboundLayerNames;
    }

    /**
     * Get number of outputs from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static int getNOutFromConfig(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        int nOut;
        if (innerConfig.containsKey(LAYER_FIELD_OUTPUT_DIM))
            /* Most feedforward layers: Dense, RNN, etc. */
            nOut = (int) innerConfig.get(LAYER_FIELD_OUTPUT_DIM);
        else if (innerConfig.containsKey(LAYER_FIELD_NB_FILTER))
            /* Convolutional layers. */
            nOut = (int) innerConfig.get(LAYER_FIELD_NB_FILTER);
        else
            throw new InvalidKerasConfigurationException("Could not determine number of outputs for layer: no "
                            + LAYER_FIELD_OUTPUT_DIM + " or " + LAYER_FIELD_NB_FILTER + " field found");
        return nOut;
    }

    /**
     * Get dropout from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    protected double getDropoutFromConfig(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        /* NOTE: Keras "dropout" parameter determines dropout probability,
         * while DL4J "dropout" parameter determines retention probability.
         */
        double dropout = 1.0;
        if (innerConfig.containsKey(LAYER_FIELD_DROPOUT)) {
            /* For most feedforward layers. */
            dropout = 1.0 - (double) innerConfig.get(LAYER_FIELD_DROPOUT);
        } else if (innerConfig.containsKey(LAYER_FIELD_DROPOUT_W)) {
            /* For LSTMs. */
            dropout = 1.0 - (double) innerConfig.get(LAYER_FIELD_DROPOUT_W);
        }
        return dropout;
    }

    /**
     * Get activation function from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    protected IActivation getActivationFromConfig(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_ACTIVATION))
            throw new InvalidKerasConfigurationException("Keras layer is missing " + LAYER_FIELD_ACTIVATION + " field");
        return mapActivation((String) innerConfig.get(LAYER_FIELD_ACTIVATION));
    }

    /**
     * Get weight initialization from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @param enforceTrainingConfig
     * @return
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    protected WeightInit getWeightInitFromConfig(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_INIT))
            throw new InvalidKerasConfigurationException("Keras layer is missing " + LAYER_FIELD_INIT + " field");
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
     * @param layerConfig       dictionary containing Keras layer configuration     Map containing Keras weight reguarlization configuration
     * @return                L1 regularization strength (0.0 if none)
     */
    public static double getWeightL1RegularizationFromConfig(Map<String, Object> layerConfig, boolean willBeTrained)
                    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.containsKey(LAYER_FIELD_W_REGULARIZER)) {
            Map<String, Object> regularizerConfig = (Map<String, Object>) innerConfig.get(LAYER_FIELD_W_REGULARIZER);
            if (regularizerConfig != null && regularizerConfig.containsKey(REGULARIZATION_TYPE_L1))
                return (double) regularizerConfig.get(REGULARIZATION_TYPE_L1);
        }
        return 0.0;
    }

    /**
     * Get L2 weight regularization (if any) from Keras weight regularization configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration     Map containing Keras weight reguarlization configuration
     * @return                L1 regularization strength (0.0 if none)
     */
    public static double getWeightL2RegularizationFromConfig(Map<String, Object> layerConfig, boolean willBeTrained)
                    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.containsKey(LAYER_FIELD_W_REGULARIZER)) {
            Map<String, Object> regularizerConfig = (Map<String, Object>) innerConfig.get(LAYER_FIELD_W_REGULARIZER);
            if (regularizerConfig != null && regularizerConfig.containsKey(REGULARIZATION_TYPE_L2))
                return (double) regularizerConfig.get(REGULARIZATION_TYPE_L2);
        }
        return 0.0;
    }

    /**
     * Get L1 bias regularization (if any) from Keras bias regularization configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration     Map containing Keras bias reguarlization configuration
     * @return                L1 regularization strength (0.0 if none)
     */
    public static double getBiasL1RegularizationFromConfig(Map<String, Object> layerConfig, boolean willBeTrained)
                    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.containsKey(LAYER_FIELD_B_REGULARIZER)) {
            Map<String, Object> regularizerConfig = (Map<String, Object>) innerConfig.get(LAYER_FIELD_B_REGULARIZER);
            if (regularizerConfig != null && regularizerConfig.containsKey(REGULARIZATION_TYPE_L1))
                throw new UnsupportedKerasConfigurationException("L1 regularization for bias parameter not supported");
        }
        return 0.0;
    }

    /**
     * Get L2 bias regularization (if any) from Keras bias regularization configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration     Map containing Keras weight reguarlization configuration
     * @return                L1 regularization strength (0.0 if none)
     */
    private static double getBiasL2RegularizationFromConfig(Map<String, Object> layerConfig, boolean willBeTrained)
                    throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (innerConfig.containsKey(LAYER_FIELD_B_REGULARIZER)) {
            Map<String, Object> regularizerConfig = (Map<String, Object>) innerConfig.get(LAYER_FIELD_B_REGULARIZER);
            if (regularizerConfig != null && regularizerConfig.containsKey(REGULARIZATION_TYPE_L2))
                throw new UnsupportedKerasConfigurationException("L2 regularization for bias parameter not supported");
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
                if (!field.equals(REGULARIZATION_TYPE_L1) && !field.equals(REGULARIZATION_TYPE_L2)
                                && !field.equals(LAYER_FIELD_NAME)) {
                    if (enforceTrainingConfig)
                        throw new UnsupportedKerasConfigurationException("Unknown regularization field " + field);
                    else
                        log.warn("Ignoring unknown regularization field " + field);
                }
            }
        }
    }

    /**
     * Get (convolution) stride from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static int[] getStrideFromConfig(Map<String, Object> layerConfig) throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        int[] strides = null;
        if (innerConfig.containsKey(LAYER_FIELD_SUBSAMPLE)) {
            /* Convolutional layers. */
            List<Integer> stridesList = (List<Integer>) innerConfig.get(LAYER_FIELD_SUBSAMPLE);
            strides = ArrayUtil.toArray(stridesList);
        } else if (innerConfig.containsKey(LAYER_FIELD_STRIDES)) {
            /* Pooling layers. */
            List<Integer> stridesList = (List<Integer>) innerConfig.get(LAYER_FIELD_STRIDES);
            strides = ArrayUtil.toArray(stridesList);
        } else
            throw new InvalidKerasConfigurationException("Could not determine layer stride: no " + LAYER_FIELD_SUBSAMPLE
                            + " or " + LAYER_FIELD_STRIDES + " field found");
        return strides;
    }

    /**
     * Get (convolution) kernel size from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static int[] getKernelSizeFromConfig(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        int[] kernelSize = null;
        if (innerConfig.containsKey(LAYER_FIELD_NB_ROW) && innerConfig.containsKey(LAYER_FIELD_NB_COL)) {
            /* Convolutional layers. */
            List<Integer> kernelSizeList = new ArrayList<Integer>();
            kernelSizeList.add((Integer) innerConfig.get(LAYER_FIELD_NB_ROW));
            kernelSizeList.add((Integer) innerConfig.get(LAYER_FIELD_NB_COL));
            kernelSize = ArrayUtil.toArray(kernelSizeList);
        } else if (innerConfig.containsKey(LAYER_FIELD_POOL_SIZE)) {
            /* Pooling layers. */
            List<Integer> kernelSizeList = (List<Integer>) innerConfig.get(LAYER_FIELD_POOL_SIZE);
            kernelSize = ArrayUtil.toArray(kernelSizeList);
        } else
            throw new InvalidKerasConfigurationException("Could not determine kernel size: no " + LAYER_FIELD_NB_ROW
                            + ", " + LAYER_FIELD_NB_COL + ", or " + LAYER_FIELD_POOL_SIZE + " field found");
        return kernelSize;
    }

    /**
     * Get convolution border mode from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public static ConvolutionMode getConvolutionModeFromConfig(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        if (!innerConfig.containsKey(LAYER_FIELD_BORDER_MODE))
            throw new InvalidKerasConfigurationException("Could not determine convolution border mode: no "
                            + LAYER_FIELD_BORDER_MODE + " field found");
        String borderMode = (String) innerConfig.get(LAYER_FIELD_BORDER_MODE);
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
                throw new UnsupportedKerasConfigurationException("Unsupported convolution border mode: " + borderMode);
        }
        return convolutionMode;
    }

    /**
     * Get (convolution) padding from Keras layer configuration.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @return
     * @throws InvalidKerasConfigurationException
     */
    public int[] getPaddingFromBorderModeConfig(Map<String, Object> layerConfig)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> innerConfig = getInnerLayerConfigFromConfig(layerConfig);
        int[] padding = null;
        if (!innerConfig.containsKey(LAYER_FIELD_BORDER_MODE))
            throw new InvalidKerasConfigurationException("Could not determine convolution border mode: no "
                            + LAYER_FIELD_BORDER_MODE + " field found");
        String borderMode = (String) innerConfig.get(LAYER_FIELD_BORDER_MODE);
        if (borderMode == LAYER_FIELD_BORDER_MODE) {
            padding = getKernelSizeFromConfig(layerConfig);
            for (int i = 0; i < padding.length; i++)
                padding[i]--;
        }
        return padding;
    }
}
