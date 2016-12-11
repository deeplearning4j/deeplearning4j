package org.deeplearning4j.nn.modelimport.keras;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import static org.bytedeco.javacpp.hdf5.H5F_ACC_RDONLY;

/**
 * Routines for importing saved Keras model configurations.
 *
 * @author dave@skymind.io
 */
public class ModelConfiguration {
    public static final String KERAS_MODEL_PROPERTY_CLASS_NAME = "class_name";
    public static final String KERAS_MODEL_PROPERTY_CONFIG = "config";
    public static final String KERAS_MODEL_CLASS_NAME_SEQUENTIAL = "Sequential";
    public static final String KERAS_MODEL_CLASS_NAME_FUNCTIONAL_API = "Model";

    /* Keras loss functions. */
    public static final String KERAS_LOSS_SQUARED_LOSS_1 = "mean_squared_error";
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
    public static final String KERAS_TRAINING_CONFIG_PROPERTY_LOSS = "loss";

    private static Logger log = LoggerFactory.getLogger(Model.class);

    private ModelConfiguration() {}

    /**
     * Imports a Keras Sequential model configuration saved using call to model.to_json().
     *
     * @param modelJsonFilename    Path to text file storing Keras Sequential configuration as valid JSON.
     * @return                     DL4J MultiLayerConfiguration
     * @throws IOException
     */
    public static MultiLayerConfiguration importSequentialModelConfigFromJsonFile(String modelJsonFilename)
            throws IOException {
        String modelJson = new String(Files.readAllBytes(Paths.get(modelJsonFilename)));
        return importSequentialModelConfig(modelJson);
    }

    /**
     * Imports a Keras Functional API model configuration saved using call to model.to_json().
     *
     * @param modelJsonFilename    Path to text file storing Keras Model configuration as valid JSON.
     * @return                     DL4J ComputationGraphConfiguration
     * @throws IOException
     */
    public static ComputationGraphConfiguration importModelConfigFromJsonFile(String modelJsonFilename)
            throws IOException {
        String modelJson = new String(Files.readAllBytes(Paths.get(modelJsonFilename)));
        return importModelConfig(modelJson);
    }

    /**
     * Imports a Keras Sequential model configuration saved using call to model.to_json().
     *
     * @param modelJson    String storing Keras Sequential configuration as valid JSON.
     * @return             DL4J MultiLayerConfiguration
     * @throws IOException
     */
    public static MultiLayerConfiguration importSequentialModelConfig(String modelJson)
            throws IOException {
        return importSequentialModelConfig(modelJson, null);
    }

    /**
     * Imports a Keras Functional API model configuration saved using call to model.to_json().
     *
     * @param modelJson    String storing Keras Model configuration as valid JSON.
     * @return             DL4J ComputationGraphConfiguration
     * @throws IOException
     */
    public static ComputationGraphConfiguration importModelConfig(String modelJson)
            throws IOException {
        return importModelConfig(modelJson, null);
    }

    /**
     * Imports a Keras Sequential model configuration saved using call to model.to_json() and
     * training configuration stored in valid JSON string.
     *
     * @param modelJson       String storing Keras Sequential configuration as valid JSON.
     * @param trainingJson    String storing Keras training configuration as valid JSON.
     * @return                DL4J MultiLayerConfiguration
     * @throws IOException
     * @throws IncompatibleKerasConfigurationException
     */
    public static MultiLayerConfiguration importSequentialModelConfig(String modelJson, String trainingJson)
            throws IOException, IncompatibleKerasConfigurationException {
        Map<String,Object> modelConfig = parseJsonString(modelJson);
        String arch = (String)modelConfig.get(KERAS_MODEL_PROPERTY_CLASS_NAME);
        if (!arch.equals(KERAS_MODEL_CLASS_NAME_SEQUENTIAL))
            throw new IncompatibleKerasConfigurationException("Expected \"Sequential\" model config, found " + arch);

        Map<String,Object> trainingConfig = null;
        if (trainingJson != null)
            trainingConfig = parseJsonString(trainingJson);

        NeuralNetConfiguration.Builder modelBuilder = new NeuralNetConfiguration.Builder();
        NeuralNetConfiguration.ListBuilder listBuilder = modelBuilder.list();

        /* Iterate through layer configs, building each in turn. In addition, determine
         * input type, whether model is recurrent, and truncated BPTT length.
         */
        int[] inputShape = null;
        InputType inputType = null;
        int truncatedBPTT = -1;
        List<Object> layerConfigObjects = (List<Object>)modelConfig.get(KERAS_MODEL_PROPERTY_CONFIG);
        int layerIndex = 0;
        for (Object layerConfigObject : layerConfigObjects) {
            Map<String, Object> layerConfig = LayerConfiguration.processLayerConfigObject(layerConfigObject);

            /* Determine layer input shape, if any specified. */
            int[] layerInputShape = LayerConfiguration.getLayerInputShape(layerConfig);

            /* Set overall input shape based on input shape of first layer (index 0). Layer
             * 0 MUST have a valid input shape, while other layers should not. Throw errors
             * when needed.
             */
            if (layerIndex == 0) {
                if (layerInputShape != null)
                    inputShape = layerInputShape;
                else
                    throw new IncompatibleKerasConfigurationException("Layer " + layerIndex + " must specify \"batch_input_shape\" field");
            } else if (layerInputShape != null)
                throw new IncompatibleKerasConfigurationException("Layer " + layerIndex + " should not specify \"batch_input_shape\" field");

            /* Build layer based on name, config, order. */
            Layer layer = LayerConfiguration.buildLayer(layerConfig);

            if (layer == null) //We want to skip some Keras layers (e.g., Input, Reshape)
                continue;

            /* We determine input type from the first non-Input keras layer. */
            if (inputType == null) {
                if (layer instanceof BaseRecurrentLayer) {
                    inputType = InputType.recurrent(inputShape[1]);
                    truncatedBPTT = inputShape[0];
                } else if (layer instanceof ConvolutionLayer || layer instanceof SubsamplingLayer)
                    inputType = InputType.convolutional(inputShape[0], inputShape[1], inputShape[2]);
                else
                    inputType = InputType.feedForward(inputShape[0]);
            }

            /* Detect whether L1 or L2 regularization is being applied. */
            if (layer.getL1() > 0 || layer.getL2() > 0)
                modelBuilder.regularization(true);

            /* Add layer to list builder. */
            listBuilder.layer(layerIndex, layer);
            layerIndex++;
        }

        /* Set input type.
         * TODO: should we throw an error if inputType is somehow not set (not really possible).
         * */
        if (inputType != null)
            listBuilder.setInputType(inputType);

        /* Handle truncated BPTT:
         * - less than zero if no recurrent layers found
         * - greater than zero if found recurrent layer and truncation length was set
         * - equal to zero if found recurrent layer but no truncation length set (e.g., the
         *   model was built with Theano backend and used scan symbolic loop instead of
         *   unrolling the RNN for a fixed number of steps.
         *
         * TODO: do we need to throw an error for truncatedBPTT==0?
         */
        if (truncatedBPTT == 0)
            throw new IncompatibleKerasConfigurationException("Cannot import recurrent models without fixed length sequence input.");
        else if (truncatedBPTT > 0)
            listBuilder.tBPTTForwardLength(truncatedBPTT).tBPTTBackwardLength(truncatedBPTT);

        /* If received valid trainingConfig, add loss layer and set other params. */
        if (trainingConfig != null) {
            /* Add loss layer. */
            String kerasLoss = (String)trainingConfig.get(KERAS_TRAINING_CONFIG_PROPERTY_LOSS);
            LossFunctions.LossFunction dl4jLoss = mapLossFunction(kerasLoss);
            listBuilder.layer(listBuilder.getLayerwise().size(), new LossLayer.Builder(dl4jLoss).build());

            /* TODO: handle optimizer configuration. */
            /* TODO: handle other configs (loss weights, sample weights). */
        }

        return listBuilder.build();
    }

    /**
     * Imports a Keras Functional API model configuration saved using call to model.to_json().
     *
     * @param modelJson       String storing Keras Model configuration as valid JSON.
     * @param trainingJson    String storing Keras training configuration as valid JSON.
     * @return                DL4J ComputationGraphConfiguration
     * @throws IOException
     * @throws IncompatibleKerasConfigurationException
     * @throws UnsupportedOperationException
     */
    public static ComputationGraphConfiguration importModelConfig(String modelJson, String trainingJson)
            throws IOException, IncompatibleKerasConfigurationException, UnsupportedOperationException {
        Map<String,Object> modelConfig = parseJsonString(modelJson);
        Map<String,Object> trainingConfig = null;
        if (trainingJson != null)
            trainingConfig = parseJsonString(trainingJson);

        String arch = (String)modelConfig.get(KERAS_MODEL_PROPERTY_CLASS_NAME);
        if (!arch.equals(KERAS_MODEL_CLASS_NAME_FUNCTIONAL_API))
            throw new IncompatibleKerasConfigurationException("Expected \"Model\" model config, found " + arch);

        NeuralNetConfiguration.Builder modelBuilder = new NeuralNetConfiguration.Builder();
        ComputationGraphConfiguration.GraphBuilder graphBuilder = modelBuilder.graphBuilder();

        throw new UnsupportedOperationException("BLAH");
//        /* Iterate through layer configs, building each in turn. In addition, determine
//         * input type, whether model is recurrent, and truncated BPTT length.
//         */
//        Map<String,int[]> inputShapes = new HashMap<String,int[]>();
//        List<InputType> inputTypes = new ArrayList<InputType>();
//        List<Integer> truncatedBPTT = new ArrayList<Integer>();
//        List<Object> layerConfigObjects = (List<Object>)modelConfig.get(KERAS_MODEL_PROPERTY_CONFIG);
//        int layerIndex = 0;
//        for (Object layerConfigObject : layerConfigObjects) {
//            Map<String,Object> layerConfig = LayerConfiguration.processLayerConfigObject(layerConfigObject);
//            String layerClassName = (String)layerConfig.get(LayerConfiguration.LAYER_PROPERTY_CLASS_NAME);
//            if (layerClassName.equals(LayerConfiguration.LAYER_TYPE_INPUT)) {
//                String layerName = (String)layerConfig.get(LayerConfiguration.LAYER_PROPERTY_NAME);
//                int[] inputShape = getInputShapeFromInputLayer(layerConfig);
//                inputShapes.put(layerName, inputShape);
//            }
//
//            /* Build layer based on name, config, order. */
//            Layer layer = LayerConfiguration.buildLayer(layerClassName, layerConfig, layerIndex == layerConfigObjects.size()-1);
//            if (layer == null)
//                continue;
//        }
    }

    /**
     * Extract a Map from layer name to configuration. Primary use is for looking up layer properties
     * that are necessary for interpreting weights (e.g., "dim_ordering" for convolutional layers) but
     * that are not stored in DL4J layer configurations.
     *
     * @param modelJson    String storing Keras configuration as valid JSON
     * @return             Map from metadata fields to relevant values
     * @throws IOException
     */
    public static Map<String, Object> getLayerConfigurationAsMap(String modelJson) throws IOException {
        Map<String,Object> kerasConfig = parseJsonString(modelJson);
        List<Object> layerConfigObjects = (List<Object>)kerasConfig.get(KERAS_MODEL_PROPERTY_CONFIG);
        Map<String,Object> layerConfigs = new HashMap<>();
        for (Object layerConfigObject : layerConfigObjects) {
            Map<String,Object> layerConfig = LayerConfiguration.processLayerConfigObject(layerConfigObject);
            String layerName = (String)layerConfig.get(LayerConfiguration.KERAS_LAYER_PROPERTY_NAME);
            layerConfigs.put(layerName, layerConfig);
        }
        return layerConfigs;
    }

    public static boolean modelIsSequential(String modelJson) throws IOException {
        Map<String,Object> modelConfig = parseJsonString(modelJson);
        return modelConfig.containsKey(KERAS_MODEL_PROPERTY_CLASS_NAME) &&
                modelConfig.get(KERAS_MODEL_PROPERTY_CLASS_NAME).equals(KERAS_MODEL_CLASS_NAME_SEQUENTIAL);
    }

    /**
     * Map Keras to DL4J loss functions.
     *
     * @param kerasLoss    String containing Keras activation function name
     * @return             String containing DL4J activation function name
     */
    private static LossFunctions.LossFunction mapLossFunction(String kerasLoss) {
        LossFunctions.LossFunction dl4jLoss = LossFunctions.LossFunction.SQUARED_LOSS;
        switch (kerasLoss) {
            case KERAS_LOSS_SQUARED_LOSS_1:
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
                throw new IncompatibleKerasConfigurationException("Unknown Keras loss function " + kerasLoss);
        }
        return dl4jLoss;
    }

    /**
     * Convenience function for parsing JSON strings.
     *
     * @param json    String containing valid JSON
     * @return        Nested Map with arbitrary depth
     * @throws IOException
     */
    private static Map<String,Object> parseJsonString(String json) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        TypeReference<HashMap<String,Object>> typeRef = new TypeReference<HashMap<String,Object>>() {};
        return mapper.readValue(json, typeRef);
    }
}
