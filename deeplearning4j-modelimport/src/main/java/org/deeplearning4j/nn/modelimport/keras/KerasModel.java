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

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Build ComputationGraph from Keras (Functional API) Model or
 * Sequential model configuration.
 *
 * @author dave@skymind.io
 */
public class KerasModel {
    /* Model class name field. */
    public static final String MODEL_FIELD_CLASS_NAME = "class_name";
    public static final String MODEL_CLASS_NAME_SEQUENTIAL = "Sequential";
    public static final String MODEL_CLASS_NAME_MODEL = "Model";

    /* Model configuration field. */
    public static final String MODEL_FIELD_CONFIG = "config";
    public static final String MODEL_CONFIG_FIELD_LAYERS = "layers";
    public static final String MODEL_CONFIG_FIELD_INPUT_LAYERS = "input_layers";
    public static final String MODEL_CONFIG_FIELD_OUTPUT_LAYERS = "output_layers";

    /* Training configuration field. */
    public static final String TRAINING_CONFIG_FIELD_LOSS = "loss";

    /* Default setting for truncated BPTT. */
    public static final int DO_NOT_USE_TRUNCATED_BPTT = -123456789;

    protected String className;               // Keras model class name
    protected List<String> layerNamesOrdered; // ordered list of layer names
    protected Map<String,KerasLayer> layers;  // map from layer name to KerasLayer
    protected ArrayList<String> inputLayerNames;   // list of input layer names
    protected ArrayList<String> outputLayerNames;  // list of output layer names
    protected Map<String,Set<String>> inputToOutput; // graph of input-to-output relationships
    protected Map<String,Set<String>> outputToInput; // graph of output-to-input relationships
    protected int truncatedBPTT = DO_NOT_USE_TRUNCATED_BPTT;   // truncated BPTT value
    protected Map<String,Map<String,INDArray>> weights = null; // map from layer to parameter to weights

    /**
     * Constructor for Model configuration JSON string.
     *
     * @param modelConfigJson       model configuration JSON string
     * @throws IOException
     */
    public KerasModel(String modelConfigJson) throws IOException {
        Map<String,Object> classNameAndLayerLists = parseJsonString(modelConfigJson);
        this.className = (String) checkAndGetModelField(classNameAndLayerLists, MODEL_FIELD_CLASS_NAME);
        if (!this.className.equals(MODEL_CLASS_NAME_MODEL))
            throw new InvalidKerasConfigurationException("Expected model class name Model (found " + this.className + ")");
        Map<String,Object> layerLists = (Map<String,Object>) checkAndGetModelField(classNameAndLayerLists, MODEL_FIELD_CONFIG);

        List<Object> layerConfigs = (List<Object>) checkAndGetModelField(layerLists, MODEL_CONFIG_FIELD_LAYERS);
        helperPrepareLayers(layerConfigs);

        this.inputLayerNames = new ArrayList();
        for (Object inputLayerNameObj : (List<Object>) checkAndGetModelField(layerLists, MODEL_CONFIG_FIELD_INPUT_LAYERS))
            this.inputLayerNames.add((String)((List<Object>)inputLayerNameObj).get(0));
        this.outputLayerNames = new ArrayList();
        for (Object outputLayerNameObj : (List<Object>) checkAndGetModelField(layerLists, MODEL_CONFIG_FIELD_OUTPUT_LAYERS))
            this.outputLayerNames.add((String)((List<Object>)outputLayerNameObj).get(0));
        helperPrepareGraph();
    }

    /**
     * Constructor for Model configuration JSON string and map containing weights.
     *
     * @param modelConfigJson       model configuration JSON string
     * @param weights               map from layer to parameter to weights
     * @throws IOException
     */
    public KerasModel(String modelConfigJson, Map<String, Map<String,INDArray>> weights) throws IOException {
        this(modelConfigJson);
        copyWeightsToModel(weights);
    }

    /**
     * Constructor for Model and training configuration JSON strings.
     *
     * @param modelConfigJson       model configuration JSON string
     * @param trainingConfigJson    training configuration JSON string
     * @throws IOException
     */
    public KerasModel(String modelConfigJson, String trainingConfigJson) throws IOException {
        this(modelConfigJson);
        importTrainingConfiguration(trainingConfigJson);
    }

    /**
     * Constructor for Model and training configuration JSON strings and map containing weights.
     *
     * @param modelConfigJson       model configuration JSON string
     * @param trainingConfigJson    training configuration JSON string
     * @param weights               map from layer to parameter to weights
     * @throws IOException
     */
    public KerasModel(String modelConfigJson, String trainingConfigJson, Map<String, Map<String,INDArray>> weights)
            throws IOException {
        this(modelConfigJson);
        importTrainingConfiguration(trainingConfigJson);
        copyWeightsToModel(weights);
    }

    protected KerasModel() {}

    /**
     * Incorporate training configuration details into model. Includes loss function,
     * optimization details, etc.
     *
     * @param trainingConfigJson
     * @throws IOException
     */
    public void importTrainingConfiguration(String trainingConfigJson) throws IOException {
        Map<String,Object> trainingConfig = parseJsonString(trainingConfigJson);

        /* Add loss layers for each loss function. */
        Map<String,String> kerasLossMap = new HashMap<String,String>();
        helperAddLossLayers(checkAndGetTrainingField(trainingConfig, TRAINING_CONFIG_FIELD_LOSS));

        /* TODO: handle optimizer configuration. */
        /* TODO: handle other configs (loss weights, sample weights). */
    }

    /**
     * Copy weights into model.
     *
     * @param weights       weights stored in map from layer to parameter to weights
     */
    public void copyWeightsToModel(Map<String,Map<String,INDArray>> weights) {
        this.weights = new HashMap<String,Map<String,INDArray>>();
        for (String layerName : weights.keySet()) {
            if (!this.layers.containsKey(layerName))
                throw new InvalidKerasConfigurationException("Attempting to import weights for unknown layer " + layerName);
            if (!this.weights.containsKey(layerName))
                this.weights.put(layerName, new HashMap<String,INDArray>());
            for (String paramName : weights.get(layerName).keySet())
                this.weights.get(layerName).put(paramName, weights.get(layerName).get(paramName));
        }
    }

    /**
     * Configure a ComputationGraph from this Keras Model configuration.
     *
     * @return          ComputationGraph
     */
    public ComputationGraphConfiguration getComputationGraphConfiguration()
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        if (!this.className.equals(MODEL_CLASS_NAME_MODEL) && !this.className.equals(MODEL_CLASS_NAME_SEQUENTIAL))
            throw new InvalidKerasConfigurationException("Keras model class name " + this.className + " incompatible with ComputationGraph");
        NeuralNetConfiguration.Builder modelBuilder = new NeuralNetConfiguration.Builder();

        /* Build String array of input layer names. */
        String[] inputLayerArray = new String[this.inputLayerNames.size()];
        this.inputLayerNames.toArray(inputLayerArray);

        /* Build InputType array of input layer types. */
        List<InputType> inputTypes = new ArrayList<InputType>();
        for (String inputLayerName : this.inputLayerNames)
            inputTypes.add(inferInputType(inputLayerName));
        InputType[] inputTypeArray = new InputType[inputTypes.size()];
        inputTypes.toArray(inputTypeArray);

        /* Build String array of output layer names. */
        String[] outputLayerArray = new String[this.outputLayerNames.size()];
        this.outputLayerNames.toArray(outputLayerArray);

        /* Build ComputationGraph. Add input names and types and output names. */
        ComputationGraphConfiguration.GraphBuilder graphBuilder = modelBuilder.graphBuilder()
                .addInputs(inputLayerArray)
                .setInputTypes(inputTypeArray)
                .setOutputs(outputLayerArray);

        /* Add layerNamesOrdered one at a time. */
        for (String layerName : this.layerNamesOrdered) {
            KerasLayer layer = this.layers.get(layerName);
            if (layer.isDl4jLayer()) { // Ignore "preprocessor" layers for now
                /* Get inbound layer names. */
                List<String> inboundLayerNames = inferInboundLayerNames(layerName);
                String[] inboundLayerArray = new String[inboundLayerNames.size()];
                inboundLayerNames.toArray(inboundLayerArray);
                /* Add DL4J layer. */
                graphBuilder.addLayer(layerName, layer.getDl4jLayer(), inboundLayerArray);
            }
        }

        /* Handle truncated BPTT:
         * - less than zero if no recurrent layerNamesOrdered found
         * - greater than zero if found recurrent layer and truncation length was set
         * - equal to zero if found recurrent layer but no truncation length set (e.g., the
         *   model was built with Theano backend and used scan symbolic loop instead of
         *   unrolling the RNN for a fixed number of steps.
         *
         * TODO: should we not throw an error for truncatedBPTT==0?
         */
        if (this.truncatedBPTT == 0)
            throw new UnsupportedKerasConfigurationException("Cannot import recurrent models without fixed length sequence input.");
        else if (this.truncatedBPTT > 0)
            graphBuilder.tBPTTForwardLength(truncatedBPTT).tBPTTBackwardLength(truncatedBPTT);
        return graphBuilder.build();
    }

    /**
     * Build a ComputationGraph from this Keras Model configuration.
     *
     * @return          ComputationGraph
     */
    public ComputationGraph getComputationGraph() {
        return getComputationGraph(true);
    }

    /**
     * Build a ComputationGraph from this Keras Model configuration and import weights.
     *
     * @return          ComputationGraph
     */
    public ComputationGraph getComputationGraph(boolean importWeights) {
        ComputationGraph model = new ComputationGraph(getComputationGraphConfiguration());
        if (importWeights)
            copyWeightsToModel(model, this.weights, this.layers);
        return model;
    }

    /**
     * Helper method called from constructor. Converts layer configuration
     * JSON into KerasLayer objects.
     *
     * @param layerConfigs     List of Keras layer configuration objects (nested maps).
     */
    protected void helperPrepareLayers(List<Object> layerConfigs) {
        this.layers = new HashMap<String,KerasLayer>();
        this.layerNamesOrdered = new ArrayList<String>();
        for (Object layerConfig : layerConfigs) {
            KerasLayer layer = new KerasLayer((Map<String,Object>)layerConfig);
            this.layerNamesOrdered.add(layer.getName());
            this.layers.put(layer.getName(), layer);
        }
    }

    /**
     * Helper method called from constructor. Builds input-to-output
     * and output-to-input graphs based on inbound layer lists.
     */
    protected void helperPrepareGraph() {
        this.outputToInput = new HashMap<String,Set<String>>();
        this.inputToOutput = new HashMap<String,Set<String>>();
        for (String childName : this.layerNamesOrdered) {
            if (!outputToInput.containsKey(childName))
                outputToInput.put(childName, new HashSet<String>());
            for (String parentName : this.layers.get(childName).getInboundLayerNames()) {
                outputToInput.get(childName).add(parentName);
                if (!inputToOutput.containsKey(parentName))
                    inputToOutput.put(parentName, new HashSet<String>());
                inputToOutput.get(parentName).add(childName);
            }
            if (!inputToOutput.containsKey(childName))
                inputToOutput.put(childName, new HashSet<String>());
        }
    }

    /**
     * Helper method called from constructor. Process a Keras loss
     * configuration object (from training configuration JSON)
     * into one or more LossLayers.
     *
     * @param kerasLossObj      Keras loss configuration
     */
    protected void helperAddLossLayers(Object kerasLossObj) {
        Map<String,KerasLayer> lossLayers = new HashMap<String,KerasLayer>();
        if (kerasLossObj instanceof String) {
            String kerasLoss = (String)kerasLossObj;
            for (String outputLayerName : this.outputLayerNames)
                lossLayers.put(outputLayerName, KerasLayer.createLossLayer(outputLayerName + "_loss", kerasLoss));
            this.outputLayerNames.clear();
        } else if (kerasLossObj instanceof Map) {
            Map<String,Object> kerasLossMap = (Map<String,Object>)kerasLossObj;
            for (String outputLayerName : kerasLossMap.keySet()) {
                this.outputLayerNames.remove(outputLayerName);
                Object kerasLoss = kerasLossMap.get(outputLayerName);
                if (kerasLoss instanceof String)
                    lossLayers.put(outputLayerName, KerasLayer.createLossLayer(outputLayerName + "_loss", (String)kerasLoss));
                else
                    throw new InvalidKerasConfigurationException("Unknown Keras loss " + kerasLoss.toString());
            }
        }

        for (String outputLayerName : lossLayers.keySet()) {
            KerasLayer lossLayer = lossLayers.get(outputLayerName);
            this.layers.put(lossLayer.getName(), lossLayer);
            String lossLayerName = lossLayer.getName();
            outputLayerNames.add(lossLayerName);
            this.layerNamesOrdered.add(lossLayerName);
            if (!this.inputToOutput.containsKey(outputLayerName))
                this.inputToOutput.put(outputLayerName, new HashSet<String>());
            this.inputToOutput.get(outputLayerName).add(lossLayerName);
            if (!this.outputToInput.containsKey(lossLayerName))
                this.outputToInput.put(lossLayerName, new HashSet<String>());
            this.outputToInput.get(lossLayerName).add(outputLayerName);
        }
    }

    /**
     * Infer InputType for input layer.
     *
     * @param inputLayerName    name of input layer
     * @return                  InputType of input layer
     */
    protected InputType inferInputType(String inputLayerName)
            throws UnsupportedOperationException, UnsupportedKerasConfigurationException {
        if (!this.inputLayerNames.contains(inputLayerName))
            throw new UnsupportedOperationException("Cannot infer input type for non-input layer " + inputLayerName);
        int[] inputShape = this.layers.get(inputLayerName).getInputShape();
        InputType inputType = null;
        List<String> layerNameQueue = new ArrayList<String>(this.inputToOutput.get(inputLayerName));
        while (inputType == null && !layerNameQueue.isEmpty()) {
            KerasLayer nextLayer = this.layers.get(layerNameQueue.remove(0));
            if (nextLayer.isDl4jLayer()) {
                Layer dl4jLayer = nextLayer.getDl4jLayer();
                if (dl4jLayer instanceof BaseRecurrentLayer) {
                    if (inputShape.length != 2) // recurrent inputs should have rank 2 (# steps, # channels)
                        throw new UnsupportedKerasConfigurationException("Input to Recurrent layer must have rank 2 (found " + inputShape.length + ")");
                    inputType = InputType.recurrent(inputShape[1]);
                    this.truncatedBPTT = inputShape[0];
                } else if (dl4jLayer instanceof ConvolutionLayer || dl4jLayer instanceof SubsamplingLayer) {
                    if (inputShape.length != 3) // convolutional inputs should have rank 3 (# rows, # cols, # channels)
                        throw new UnsupportedKerasConfigurationException("Input to Convolutional layer must have rank 3 (found " + inputShape.length + ")");
                    inputType = InputType.convolutional(inputShape[0], inputShape[1], inputShape[2]);
                } else {
                    if (inputShape.length != 1) // other inputs should be flat vectors
                        throw new UnsupportedKerasConfigurationException("Input to FeedForward layer must have rank 1 (found " + inputShape.length + ")");
                    inputType = InputType.feedForward(inputShape[0]);
                }
            }
            layerNameQueue.addAll(this.inputToOutput.get(nextLayer.getName()));
        }
        if (inputType == null)
            throw new UnsupportedKerasConfigurationException("Could not infer InputType for input layer " + inputLayerName);
        return inputType;
    }

    /**
     * Infer list of inbound layers for (i.e., layer inputs to) given layer.
     * @param layerName     name of layer
     * @return              list of inbound layer names
     */
    protected List<String> inferInboundLayerNames(String layerName) {
        ArrayList<String> inboundLayerNames = new ArrayList<String>(this.outputToInput.get(layerName));
        for (int i = 0; i < inboundLayerNames.size(); i++) {
            KerasLayer nextLayer = this.layers.get(inboundLayerNames.get(i));
            if (!nextLayer.isValidInboundLayer()) {
                String nextLayerName = inboundLayerNames.remove(i);
                inboundLayerNames.addAll(i--, this.outputToInput.get(nextLayerName));
            }
        }
        return inboundLayerNames;
    }

    /**
     * Convenience function for checking whether a map contains a key,
     * throwing an error if it does not, and returning the corresponding
     * value if it does. We do this over and over again with the maps
     * created by parsing Keras model configuration JSON strings.
     *
     * @param map   Nested (key,value) map of arbitrary depth representing JSON
     * @param key   Key to check for in map
     * @return      Keras configuration object
     * @throws InvalidKerasConfigurationException
     */
    protected static Object checkAndGetModelField(Map<String,Object> map, String key)
            throws InvalidKerasConfigurationException {
        if (!map.containsKey(key))
            throw new InvalidKerasConfigurationException("Field " + key + " missing from model config");
        return map.get(key);
    }

    /**
     * Convenience function for checking whether a map contains a key,
     * throwing an error if it does not, and returning the corresponding
     * value if it does. We do this over and over again with the maps
     * created by parsing Keras training configuration JSON strings.
     *
     * @param map   Nested (key,value) map of arbitrary depth representing JSON
     * @param key   Key to check for in map
     * @return      Keras configuration object
     * @throws InvalidKerasConfigurationException
     */
    protected static Object checkAndGetTrainingField(Map<String,Object> map, String key)
        throws InvalidKerasConfigurationException {
        if (!map.containsKey(key))
            throw new InvalidKerasConfigurationException("Field " + key + " missing from training config");
        return map.get(key);
    }

    /**
     * Convenience function for parsing JSON strings.
     *
     * @param json    String containing valid JSON
     * @return        Nested (key,value) map of arbitrary depth
     * @throws IOException
     */
    protected static Map<String,Object> parseJsonString(String json) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        TypeReference<HashMap<String,Object>> typeRef = new TypeReference<HashMap<String,Object>>() {};
        return mapper.readValue(json, typeRef);
    }

    /**
     * Helper function to import weights from nested Map into existing model. Depends critically
     * on matched layer and parameter names. In general this seems to be straightforward for most
     * Keras models and layerNamesOrdered, but there may be edge cases.
     *
     * @param model             DL4J Model interface
     * @param weights           nested Map from layer names to parameter names to INDArrays
     * @param kerasLayers       Map from layerName to layerConfig
     * @return                  DL4J Model interface
     * @throws InvalidKerasConfigurationException
     */
    protected static org.deeplearning4j.nn.api.Model copyWeightsToModel(org.deeplearning4j.nn.api.Model model,
                                                                        Map<String, Map<String, INDArray>> weights,
                                                                        Map<String,KerasLayer> kerasLayers)
            throws InvalidKerasConfigurationException {
        /* TODO: how might this break?
         * - mismatch between layer/parameter names?
         */
        for (String layerName : weights.keySet()) {
            KerasLayer kerasLayer = kerasLayers.get(layerName);
            org.deeplearning4j.nn.api.Layer layer = null;
            if (model instanceof MultiLayerNetwork)
                layer = ((MultiLayerNetwork)model).getLayer(layerName);
            else
                layer = ((ComputationGraph)model).getLayer(layerName);
            for (String kerasParamName : weights.get(layerName).keySet()) {
                String paramName = null;
                /* TensorFlow backend often appends ":" followed by one
                 * or more digits to parameter names, but this is not
                 * reflected in the model config. We must strip it off.
                 */
                Pattern p = Pattern.compile(":\\d+$");
                Matcher m = p.matcher(kerasParamName);
                if (m.find())
                    paramName = m.replaceFirst("");
                else
                    paramName = kerasParamName;
                INDArray W = weights.get(layerName).get(kerasParamName);
                if (layer instanceof org.deeplearning4j.nn.layers.convolution.ConvolutionLayer && paramName.equals(ConvolutionParamInitializer.WEIGHT_KEY)) {
                    /* Theano and TensorFlow backends store convolutional weights
                     * with a different dimensional ordering than DL4J so we need
                     * to permute them to match.
                     *
                     * DL4J: (# outputs, # channels, # rows, # cols)
                     */

                    switch (kerasLayer.getDimOrder()) {
                        case TENSORFLOW:
                            /* TensorFlow convolutional weights: # rows, # cols, # channels, # outputs */
                            W = W.permute(3, 2, 0, 1);
                            break;
                        case THEANO:
                            /* Theano convolutional weights: # channels, # rows, # cols, # outputs */
                            W = W.permute(3, 0, 1, 2);
                        case UNKNOWN:
                            throw new InvalidKerasConfigurationException("Unknown keras backend " + kerasLayer.getDimOrder());
                    }
                }
                layer.setParam(paramName, W);
            }
        }
        return model;
    }
}
