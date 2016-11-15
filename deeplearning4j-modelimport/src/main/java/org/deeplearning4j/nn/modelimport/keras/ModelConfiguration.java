package org.deeplearning4j.nn.modelimport.keras;

import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Routines for importing saved Keras model configurations.
 *
 * @author davekale
 */
public class ModelConfiguration {
    private static Logger log = LoggerFactory.getLogger(Model.class);

    private ModelConfiguration() {}

    /**
     * Imports a Keras Sequential model configuration saved using call to model.to_json().
     *
     * @param configJsonFilename    Path to text file storing Keras configuration as valid JSON.
     * @return                      DL4J MultiLayerConfiguration
     * @throws IOException
     */
    public static MultiLayerConfiguration importSequentialModelConfigFromFile(String configJsonFilename)
            throws IOException {
        String configJson = new String(Files.readAllBytes(Paths.get(configJsonFilename)));
        return importSequentialModelConfig(configJson);
    }

    /**
     * Imports a Keras Functional API model configuration saved using call to model.to_json().
     *
     * @param configJsonFilename    Path to text file storing Keras configuration as valid JSON.
     * @return                      DL4J ComputationGraphConfiguration
     * @throws IOException
     */
    public static ComputationGraphConfiguration importFunctionalApiConfigFromFile(String configJsonFilename)
        throws IOException {
        String configJson = new String(Files.readAllBytes(Paths.get(configJsonFilename)));
        return importFunctionalApiConfig(configJson);
    }

    /**
     * Imports a Keras Sequential model configuration saved using call to model.to_json().
     *
     * @param configJson    String storing Keras configuration as valid JSON.
     * @return              DL4J MultiLayerConfiguration
     * @throws IOException
     */
    public static MultiLayerConfiguration importSequentialModelConfig(String configJson)
            throws IOException {
        Map<String,Object> kerasConfig = parseJsonString(configJson);
        MultiLayerConfiguration modelConfig = importSequentialModelConfig(kerasConfig);
        return modelConfig;
    }

    /**
     * Imports a Keras Functional API model configuration saved using call to model.to_json().
     *
     * @param configJson    String storing Keras configuration as valid JSON.
     * @return              DL4J ComputationGraphConfiguration
     * @throws IOException
     */
    public static ComputationGraphConfiguration importFunctionalApiConfig(String configJson)
            throws IOException {
        Map<String,Object> kerasConfig = parseJsonString(configJson);
        ComputationGraphConfiguration modelConfig = importFunctionalApiConfig(kerasConfig);
        return modelConfig;
    }

    /**
     * Imports a Keras Sequential model configuration saved using call to model.to_json().
     *
     * @param kerasConfig   Nested Map storing Keras configuration read from valid JSON.
     * @return              DL4J MultiLayerConfiguration
     * @throws IOException
     * @throws NotImplementedException
     * @throws IncompatibleKerasConfigurationException
     */
    private static MultiLayerConfiguration importSequentialModelConfig(Map<String,Object> kerasConfig)
            throws IOException, IncompatibleKerasConfigurationException {
        String arch = (String)kerasConfig.get("class_name");
        if (!arch.equals("Sequential"))
            throw new IncompatibleKerasConfigurationException("Expected \"Sequential\" model config, found " + arch);

        /* Make first pass through layer configs to
         * - merge dropout layers into subsequent layers
         * - merge activation layers into previous layers
         * TODO: remove this once Dropout layer added to DL4J
         */
        double prevDropout = 0.0;
        List<Map<String,Object>> layerConfigs = new ArrayList<>();
        for (Object o : (List<Object>)kerasConfig.get("config")) {
            String kerasLayerName = (String)((Map<String,Object>)o).get("class_name");
            Map<String,Object> layerConfig = (Map<String,Object>)((Map<String,Object>)o).get("config");

            switch (kerasLayerName) {
                case "Dropout":
                    /* Store dropout layer so we can merge into subsequent layer.
                     * TODO: remove once Dropout layer added to DL4J.
                     */
                    prevDropout = (double)layerConfig.get("p");
                    continue;
                case "Activation":
                    /* Merge activation function into previous layer.
                     * TODO: we have an Activation layer in DL4J so maybe remove this.
                     */
                    if (layerConfigs.size() == 0)
                        throw new IncompatibleKerasConfigurationException("Plain activation layer applied to input not supported.");
                    String activation = LayerConfiguration.mapActivation((String)layerConfig.get("activation"));
                    layerConfigs.get(layerConfigs.size()-1).put("activation", activation);
                    continue;
            }
            layerConfig.put("keras_class", kerasLayerName);

            /* Merge dropout from previous layer.
             * TODO: remove once Dropout layer added to DL4J.
             */
            if (prevDropout > 0) {
                double oldDropout = layerConfig.containsKey("dropout") ? (double)layerConfig.get("dropout") : 0.0;
                double newDropout = 1.0 - (1.0 - prevDropout) * (1.0 - oldDropout);
                layerConfig.put("dropout", newDropout);
                if (oldDropout != newDropout)
                    log.warn("Changed layer-defined dropout " + oldDropout + " to " + newDropout +
                            " because of previous Dropout=" + newDropout + " layer");
                prevDropout = 0.0;
            }

            layerConfigs.add(layerConfig);
        }

        /* Make pass through layer configs, building each in turn. In addition:
         * - get input shape from "batch_input_shape" field of input layer config
         * - get dim ordering (based on Keras backend)
         * - determine whether model includes recurrent or convolutional layers
         */
        List<Integer> batchInputShape = null;
        String dimOrdering = null;
        boolean isRecurrent = false;
        boolean isConvolutional = false;
        NeuralNetConfiguration.Builder modelBuilder = new NeuralNetConfiguration.Builder();
        NeuralNetConfiguration.ListBuilder listBuilder = modelBuilder.list();
        int layerIndex = 0;
        for (Map<String,Object> layerConfig : layerConfigs) {
            String kerasLayerName = (String)layerConfig.get("keras_class");

            /* Look for "batch_input_shape" field, which should be set
             * for input layer and ONLY for input layer.
             */
            if (layerConfig.containsKey("batch_input_shape")) {
                if (layerIndex > 0)
                    throw new IncompatibleKerasConfigurationException("Non-input layer should not specify \"batch_input_shape\" field");
                else
                    batchInputShape = (List<Integer>) layerConfig.get("batch_input_shape");
            } else if (layerIndex == 0)
                throw new IncompatibleKerasConfigurationException("Input layer must specify \"batch_input_shape\" field");

            /* Look for "dim_ordering" field, which will generally
             * show up only in convolutional and max pooling layers.
             */
            if (layerConfig.containsKey("dim_ordering")) {
                String layerDimOrdering = (String)layerConfig.get("dim_ordering");
                if (!layerDimOrdering.equals("th") && !layerDimOrdering.equals("tf"))
                    throw new IncompatibleKerasConfigurationException("Unknown Keras backend: " + layerDimOrdering);
                if (dimOrdering != null && !layerDimOrdering.equals(dimOrdering))
                    throw new IncompatibleKerasConfigurationException("Found layers with conflicting Keras backends.");
                dimOrdering = layerDimOrdering;
            }

            /* Build layer based on name, config, order. */
            Layer layer = LayerConfiguration.buildLayer(kerasLayerName, layerConfig, (layerIndex == layerConfigs.size()-1));
            if (layer == null)
                continue;

            /* Detect whether layer is recurrent or convolutional. */
            if (layer instanceof BaseRecurrentLayer)
                isRecurrent = true;
            else if (layer instanceof ConvolutionLayer)
                isConvolutional = true;
            if (layer.getL1() > 0 || layer.getL2() > 0)
                modelBuilder.regularization(true);

            /* Add layer to list builder. */
            listBuilder.layer(layerIndex, layer);
            layerIndex++;
        }

        /* If layer is recurrent or convolutional, set input type to appropriate
         * InputType with shape based on "batch_input_shape" field.
         */
        if (isRecurrent && isConvolutional) {
            throw new IncompatibleKerasConfigurationException("Recurrent convolutional architecture not supported.");
        } else if (isRecurrent) {
            listBuilder.setInputType(InputType.recurrent(batchInputShape.get(2)));
            if (batchInputShape.get(1) == null)
                log.warn("Input sequence length must be specified manually for truncated BPTT!");
            else {
                int sequenceLength = batchInputShape.get(1);
                listBuilder.tBPTTForwardLength(sequenceLength).tBPTTBackwardLength(sequenceLength);
            }
        } else if (isConvolutional) {
            int[] imageSize = new int[3];
            if (dimOrdering.equals("tf")) {
                /* TensorFlow convolutional input: # examples, # rows, # cols, # channels */
                imageSize[0] = batchInputShape.get(1);
                imageSize[1] = batchInputShape.get(2);
                imageSize[2] = batchInputShape.get(3);
            } else if (dimOrdering.equals("th")) {
                /* Theano convolutional input: # examples, # channels, # rows, # cols */
                imageSize[0] = batchInputShape.get(2);
                imageSize[1] = batchInputShape.get(3);
                imageSize[2] = batchInputShape.get(1);
            } else {
                throw new IncompatibleKerasConfigurationException("Unknown keras backend " + dimOrdering);
            }
            listBuilder.setInputType(InputType.convolutional(imageSize[0], imageSize[1], imageSize[2]));
        } else {
            listBuilder.setInputType(InputType.feedForward(batchInputShape.get(1)));
        }
        return listBuilder.build();
    }

    /**
     * Imports a Keras Functional API model configuration saved using call to model.to_json().
     *
     * @param kerasConfig   Nested Map storing Keras configuration read from valid JSON.
     * @return              DL4J ComputationGraph
     * @throws IOException
     * @throws NotImplementedException
     * @throws IncompatibleKerasConfigurationException
     */
    private static ComputationGraphConfiguration importFunctionalApiConfig(Map<String,Object> kerasConfig)
            throws IOException, NotImplementedException, IncompatibleKerasConfigurationException {
        throw new NotImplementedException("Import of Keras Functional API model configs not supported.");
    }

    /**
     * Extract Keras configuration properties that may are not relevant for configuring DL4J layers
     * or models but may be important when importing stored model weights. Only relevant property
     * at this time is the Keras backend (stored as "dim_ordering" in convolutional and pooling layers).
     *
     * @param configJson    String storing Keras configuration as valid JSON
     * @return              Map from metadata fields to relevant values
     * @throws IOException
     */
    public static Map<String, Object> extractWeightsMetadataFromConfig(String configJson) throws IOException {
        Map<String,Object> weightsMetadata = new HashMap<>();
        ObjectMapper mapper = new ObjectMapper();
        TypeReference<HashMap<String,Object>> typeRef = new TypeReference<HashMap<String,Object>>() {};
        Map<String,Object> kerasConfig = mapper.readValue(configJson, typeRef);
        List<Map<String,Object>> layers = (List<Map<String,Object>>)kerasConfig.get("config");
        for (Map<String,Object> layer : layers) {
            Map<String,Object> layerConfig = (Map<String,Object>)layer.get("config");
            if (layerConfig.containsKey("dim_ordering") && !weightsMetadata.containsKey("keras_backend"))
                weightsMetadata.put("keras_backend", layerConfig.get("dim_ordering"));
        }
        return weightsMetadata;
    }

    /**
     * Convenience function for parsing JSON strings.
     *
     * @param json  String containing valid JSON
     * @return      Nested Map with arbitrary depth
     * @throws IOException
     */
    private static Map<String,Object> parseJsonString(String json) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        TypeReference<HashMap<String,Object>> typeRef = new TypeReference<HashMap<String,Object>>() {};
        return mapper.readValue(json, typeRef);
    }
}
