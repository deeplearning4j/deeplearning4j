package org.deeplearning4j.nn.modelimport.keras.layers;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Imports a Convolution layer from Keras.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasConvolution extends KerasLayer {

    /* Keras layer parameter names. */
    public static final String KERAS_PARAM_NAME_W = "W";
    public static final String KERAS_PARAM_NAME_B = "b";

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig       dictionary containing Keras layer configuration
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasConvolution(Map<String,Object> layerConfig)
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
    public KerasConvolution(Map<String,Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        ConvolutionLayer.Builder builder = new ConvolutionLayer.Builder()
            .name(this.layerName)
            .nOut(getNOutFromConfig(layerConfig))
            .dropOut(this.dropout)
            .activation(getActivationFromConfig(layerConfig))
            .weightInit(getWeightInitFromConfig(layerConfig, enforceTrainingConfig))
            .biasInit(0.0)
            .l1(this.weightL1Regularization)
            .l2(this.weightL2Regularization)
            .convolutionMode(getConvolutionModeFromConfig(layerConfig))
            .kernelSize(getKernelSizeFromConfig(layerConfig))
            .stride(getStrideFromConfig(layerConfig));
        int[] padding = getPaddingFromConfig(layerConfig);
        if (padding != null)
            builder.padding(padding);
        this.layer = builder.build();
    }

    /**
     * Get DL4J ConvolutionLayer.
     *
     * @return  ConvolutionLayer
     */
    public ConvolutionLayer getConvolutionLayer() {
        return (ConvolutionLayer)this.layer;
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
            throw new InvalidKerasConfigurationException("Keras Convolution layer accepts only one input (received " + inputType.length + ")");
        return this.getConvolutionLayer().getOutputType(-1, inputType[0]);
    }

    /**
     * Indicates that layer has trainable weights.
     *
     * @return  true
     */
    @Override
    public boolean hasWeights() {
        return true;
    }

    /**
     * Set weights for layer.
     *
     * @param weights   Map from parameter name to INDArray.
     */
    @Override
    public void setWeights(Map<String, INDArray> weights) throws InvalidKerasConfigurationException {
        this.weights = new HashMap<String,INDArray>();
        if (weights.containsKey(KERAS_PARAM_NAME_W)) {
            /* Theano and TensorFlow backends store convolutional weights
             * with a different dimensional ordering than DL4J so we need
             * to permute them to match.
             *
             * DL4J: (# outputs, # inputs, # rows, # cols)
             */
            INDArray kerasParamValue = weights.get(KERAS_PARAM_NAME_W);
            INDArray paramValue;
            switch (this.getDimOrder()) {
                case TENSORFLOW:
                    /* TensorFlow convolutional weights: # rows, # cols, # inputs, # outputs */
                    paramValue = kerasParamValue.permute(3, 2, 0, 1);
                    break;
                case THEANO:
                    /* Theano convolutional weights match DL4J: # outputs, # inputs, # rows, # cols
                     * Theano's default behavior is to rotate filters by 180 degree before application.
                     */
                    paramValue = kerasParamValue.dup();
                    for (int i = 0; i < paramValue.tensorssAlongDimension(2,3); i++) {
                        //dup required since we only want data from the view not the whole array
                        INDArray copyFilter = paramValue.tensorAlongDimension(i,2,3).dup();
                        double [] flattenedFilter = copyFilter.ravel().data().asDouble();
                        ArrayUtils.reverse(flattenedFilter);
                        INDArray newFilter = Nd4j.create(flattenedFilter,copyFilter.shape());
                        //manipulating weights in place to save memory
                        INDArray inPlaceFilter = paramValue.tensorAlongDimension(i,2,3);
                        inPlaceFilter.muli(0).addi(newFilter);
                    }
                    break;
                default:
                    throw new InvalidKerasConfigurationException("Unknown keras backend " + this.getDimOrder());
            }
            this.weights.put(ConvolutionParamInitializer.WEIGHT_KEY, paramValue);
        } else
            throw new InvalidKerasConfigurationException("Parameter " + KERAS_PARAM_NAME_W + " does not exist in weights");
        if (weights.containsKey(KERAS_PARAM_NAME_B))
            this.weights.put(ConvolutionParamInitializer.BIAS_KEY, weights.get(KERAS_PARAM_NAME_B));
        else
            throw new InvalidKerasConfigurationException("Parameter " + KERAS_PARAM_NAME_B + " does not exist in weights");
        if (weights.size() > 2) {
            Set<String> paramNames = weights.keySet();
            paramNames.remove(KERAS_PARAM_NAME_W);
            paramNames.remove(KERAS_PARAM_NAME_B);
            String unknownParamNames = paramNames.toString();
            log.warn("Attemping to set weights for unknown parameters: " + unknownParamNames.substring(1, unknownParamNames.length()-1));
        }
    }
}
