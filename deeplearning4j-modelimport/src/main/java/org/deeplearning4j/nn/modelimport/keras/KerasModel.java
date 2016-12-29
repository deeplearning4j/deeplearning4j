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

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

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
@Slf4j
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

    /* Keras layer parameter names. */
    public static final String PARAM_NAME_GAMMA = "gamma";
    public static final String PARAM_NAME_BETA = "beta";
    public static final String PARAM_NAME_RUNNING_MEAN = "running_mean";
    public static final String PARAM_NAME_RUNNING_STD = "running_std";
    public static final String PARAM_NAME_W = "W";
    public static final String PARAM_NAME_U = "U";
    public static final String PARAM_NAME_B = "b";
    public static final String PARAM_NAME_W_C = "W_c";
    public static final String PARAM_NAME_W_F = "W_f";
    public static final String PARAM_NAME_W_I = "W_i";
    public static final String PARAM_NAME_W_O = "W_o";
    public static final String PARAM_NAME_U_C = "U_c";
    public static final String PARAM_NAME_U_F = "U_f";
    public static final String PARAM_NAME_U_I = "U_i";
    public static final String PARAM_NAME_U_O = "U_o";
    public static final String PARAM_NAME_B_C = "b_c";
    public static final String PARAM_NAME_B_F = "b_f";
    public static final String PARAM_NAME_B_I = "b_i";
    public static final String PARAM_NAME_B_O = "b_o";

    protected String className;               // Keras model class name
    protected List<String> layerNamesOrdered; // ordered list of layer names
    protected Map<String,KerasLayer> layers;  // map from layer name to KerasLayer
    protected ArrayList<String> inputLayerNames;   // list of input layer names
    protected ArrayList<String> outputLayerNames;  // list of output layer names
    protected Map<String,Set<String>> inputToOutput; // graph of input-to-output relationships
    protected Map<String,Set<String>> outputToInput; // graph of output-to-input relationships
    protected int truncatedBPTT = DO_NOT_USE_TRUNCATED_BPTT;   // truncated BPTT value
    protected Map<String,Map<String,INDArray>> weights = null; // map from layer to parameter to weights
    protected boolean train;                  // whether to build model in training mode

    /**
     * (Recommended) Builder-pattern constructor for (Functional API) Model.
     *
     * @param modelBuilder    builder object
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasModel(ModelBuilder modelBuilder) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        this(modelBuilder.modelJson, modelBuilder.modelYaml, modelBuilder.trainingJson, modelBuilder.weights, modelBuilder.train);
    }

    /**
     * (Not recommended) Constructor for (Functional API) Model from model configuration
     * (JSON or YAML), training configuration (JSON), weights, and "training mode"
     * boolean indicator. When built in training mode, certain unsupported configurations
     * (e.g., unknown regularizers) will throw Exceptions. When train=false, these
     * will generate warnings but will be otherwise ignored.
     *
     * @param modelJson       model configuration JSON string
     * @param modelYaml       model configuration YAML string
     * @param trainingJson    training configuration JSON string
     * @param weights         map from layer to parameter to weights
     * @param train           whether to build model for training
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasModel(String modelJson, String modelYaml, String trainingJson, Map<String, Map<String,INDArray>> weights, boolean train)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String,Object> classNameAndLayerLists;
        if (modelJson != null)
            classNameAndLayerLists = parseJsonString(modelJson);
        else if (modelYaml != null)
            classNameAndLayerLists = parseYamlString(modelYaml);
        else
            throw new InvalidKerasConfigurationException("Requires model configuration as either JSON or YAML string.");

        this.className = (String) checkAndGetModelField(classNameAndLayerLists, MODEL_FIELD_CLASS_NAME);
        if (!this.className.equals(MODEL_CLASS_NAME_MODEL))
            throw new InvalidKerasConfigurationException("Expected model class name Model (found " + this.className + ")");
        this.train = train;

        Map<String,Object> layerLists = (Map<String,Object>) checkAndGetModelField(classNameAndLayerLists, MODEL_FIELD_CONFIG);

        /* Convert layer configuration objects into KerasLayers. */
        helperPrepareLayers((List<Object>) checkAndGetModelField(layerLists, MODEL_CONFIG_FIELD_LAYERS));

        /* Construct lists of input and output layer names. */
        this.inputLayerNames = new ArrayList();
        for (Object inputLayerNameObj : (List<Object>) checkAndGetModelField(layerLists, MODEL_CONFIG_FIELD_INPUT_LAYERS))
            this.inputLayerNames.add((String)((List<Object>)inputLayerNameObj).get(0));
        this.outputLayerNames = new ArrayList();
        for (Object outputLayerNameObj : (List<Object>) checkAndGetModelField(layerLists, MODEL_CONFIG_FIELD_OUTPUT_LAYERS))
            this.outputLayerNames.add((String)((List<Object>)outputLayerNameObj).get(0));

        /* Construct graph. */
        helperPrepareGraph();

        /* Import training configuration. */
        if (trainingJson != null)
            helperImportTrainingConfiguration(trainingJson);

        /* Store weights map (even if null).
         * TODO: should we copy these to prevent user from changing them?
         */
        this.weights = weights;
    }

    /**
     * Helper method called from constructor. Converts layer configuration
     * JSON into KerasLayer objects.
     *
     * @param layerConfigs      List of Keras layer configurations
     */
    protected void helperPrepareLayers(List<Object> layerConfigs)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this.layers = new HashMap<String, KerasLayer>();
        this.layerNamesOrdered = new ArrayList<String>();
        for (Object layerConfig : layerConfigs) {
            KerasLayer layer = new KerasLayer((Map<String, Object>) layerConfig, this.train);
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
     * Helper method called from constructor. Incorporate training configuration details into model.
     * Includes loss function, optimization details, etc.
     *
     * @param trainingConfigJson        JSON containing Keras training configuration
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
     protected void helperImportTrainingConfiguration(String trainingConfigJson)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String,Object> trainingConfig = parseJsonString(trainingConfigJson);

        /* Add loss layers for each loss function. */
        Map<String,KerasLayer> lossLayers = new HashMap<String,KerasLayer>();
        Object kerasLossObj = checkAndGetTrainingField(trainingConfig, TRAINING_CONFIG_FIELD_LOSS);
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

        /* Add loss layers to output layer list and layer graph. */
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

        /* TODO: handle other configs (loss weights, sample weights). */
        /* TODO: handle optimizer configuration. */
    }

    protected KerasModel() {}

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
         *   unrolling the RNN for a fixed number of steps).
         */
        if (this.truncatedBPTT <= 0)
            graphBuilder.backpropType(BackpropType.Standard);
        else if (this.truncatedBPTT > 0)
            graphBuilder.backpropType(BackpropType.TruncatedBPTT)
                    .tBPTTForwardLength(truncatedBPTT)
                    .tBPTTBackwardLength(truncatedBPTT);
        return graphBuilder.build();
    }

    /**
     * Build a ComputationGraph from this Keras Model configuration and import weights.
     *
     * @return          ComputationGraph
     */
    public ComputationGraph getComputationGraph()
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return getComputationGraph(true);
    }

    /**
     * Build a ComputationGraph from this Keras Model configuration and (optionally) import weights.
     *
     * @param importWeights         whether to import weights
     * @return          ComputationGraph
     */
    public ComputationGraph getComputationGraph(boolean importWeights)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        ComputationGraph model = new ComputationGraph(getComputationGraphConfiguration());
        model.init();
        if (importWeights)
            model = (ComputationGraph)copyWeightsToModel(model, this.weights, this.layers);
        return model;
    }

    @Data
    static class ModelBuilder implements Cloneable {
        protected String modelJson;
        protected String modelYaml;
        protected String trainingJson = null;
        protected Map<String,Map<String,INDArray>> weights = null;
        protected boolean train = false;

        public ModelBuilder() {}

        public ModelBuilder modelJson(String modelJson) {
            this.modelJson = modelJson;
            this.modelYaml = null;
            return this;
        }

        public ModelBuilder modelYaml(String modelYaml) {
            this.modelYaml = modelYaml;
            this.modelJson = null;
            return this;
        }

        public ModelBuilder trainingJson(String trainingJson) {
            this.trainingJson = trainingJson;
            return this;
        }

        public ModelBuilder weights(Map<String,Map<String,INDArray>> weights) {
            this.weights = weights;
            return this;
        }

        public ModelBuilder train(boolean train) {
            this.train = train;
            return this;
        }

        public static ModelBuilder builder() {
            return new ModelBuilder();
        }

        public KerasModel buildModel()
                throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
            return new KerasModel(this);
        }

        public KerasSequentialModel buildSequential()
                throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
            return new KerasSequentialModel(this);
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
     * Convenience function for parsing JSON strings.
     *
     * @param json    String containing valid JSON
     * @return        Nested (key,value) map of arbitrary depth
     * @throws IOException
     */
    protected static Map<String,Object> parseYamlString(String json) throws IOException {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
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
     *
     * TODO: try to refactor this -- it's really messy with a lot of one-off "If layer type is X, do Y" logic.
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
                String dl4JParamName = mapParameterName(kerasParamName);
                INDArray kerasParamValue = weights.get(layerName).get(kerasParamName);
                INDArray dl4jParamValue = null;
                if (layer instanceof org.deeplearning4j.nn.layers.convolution.ConvolutionLayer) {
                    if (dl4JParamName.equals(ConvolutionParamInitializer.WEIGHT_KEY)) {
                        /* Theano and TensorFlow backends store convolutional weights
                         * with a different dimensional ordering than DL4J so we need
                         * to permute them to match.
                         *
                         * DL4J: (# outputs, # inputs, # rows, # cols)
                         */
                        switch (kerasLayer.getDimOrder()) {
                            case TENSORFLOW:
                                /* TensorFlow convolutional weights: # rows, # cols, # inputs, # outputs */
                                kerasParamValue = kerasParamValue.permute(3, 2, 0, 1);
                                break;
                            case THEANO:
                                /* Theano convolutional weights match DL4J: # outputs, # inputs, # rows, # cols */
                                break;
                            case NONE:
                                break;
                            case UNKNOWN:
                                throw new InvalidKerasConfigurationException("Unknown keras backend " + kerasLayer.getDimOrder());
                        }
                    }
                    dl4jParamValue = kerasParamValue;
                } else if (layer instanceof org.deeplearning4j.nn.layers.recurrent.GravesLSTM) {
                    /* Keras stores LSTM parameters in distinct arrays (e.g., the recurrent weights
                     * are stored in four matrices: U_c, U_f, U_i, U_o) while DL4J stores them
                     * concatenated into one matrix (e.g., U = [ U_c U_f U_o U_i ]). Thus we have
                     * to map the Keras weight matrix to its corresponding DL4J weight submatrix.
                     */

                    if (kerasParamName.startsWith(PARAM_NAME_W)) {
                        dl4jParamValue = layer.getParam(dl4JParamName);
                        int nIn = ((BaseRecurrentLayer)layer.conf().getLayer()).getNIn();
                        int nOut = ((BaseRecurrentLayer)layer.conf().getLayer()).getNOut();
                        switch (kerasParamName) {
                            case PARAM_NAME_W_C:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.interval(0, nIn),
                                                          NDArrayIndex.interval(0, nOut)}, kerasParamValue);
                                break;
                            case PARAM_NAME_W_F:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.interval(0, nIn),
                                                          NDArrayIndex.interval(nOut, 2*nOut)}, kerasParamValue);
                                break;
                            case PARAM_NAME_W_O:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.interval(0, nIn),
                                        NDArrayIndex.interval(2*nOut, 3*nOut)}, kerasParamValue);
                                break;
                            case PARAM_NAME_W_I:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.interval(0, nIn),
                                        NDArrayIndex.interval(3*nOut, 4*nOut)}, kerasParamValue);
                                break;
                        }
                    } else if (kerasParamName.startsWith(PARAM_NAME_U)) {
                        dl4jParamValue = layer.getParam(dl4JParamName);
                        int nOut = ((BaseRecurrentLayer)layer.conf().getLayer()).getNOut();
                        switch (kerasParamName) {
                            case PARAM_NAME_U_C:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.interval(0, nOut),
                                                          NDArrayIndex.interval(0, nOut)}, kerasParamValue);
                                break;
                            case PARAM_NAME_U_F:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.interval(0, nOut),
                                                          NDArrayIndex.interval(nOut, 2*nOut)}, kerasParamValue);
                                break;
                            case PARAM_NAME_U_O:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.interval(0, nOut),
                                                          NDArrayIndex.interval(2*nOut, 3*nOut)}, kerasParamValue);
                                break;
                            case PARAM_NAME_U_I:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.interval(0, nOut),
                                                          NDArrayIndex.interval(3*nOut, 4*nOut)}, kerasParamValue);
                                break;
                        }
                        /* DL4J has three additional columns in its recurrent weights matrix that don't appear
                         * in Keras LSTMs. These are for peephole connections. Since Keras doesn't use them,
                         * we zero them out.
                         */
                        dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.interval(0, nOut),
                                                  NDArrayIndex.interval(4*nOut, 4*nOut+3)}, Nd4j.zeros(nOut, 3));
                    } else if (kerasParamName.startsWith(PARAM_NAME_B)) {
                        dl4jParamValue = layer.getParam(dl4JParamName);
                        int nOut = ((BaseRecurrentLayer)layer.conf().getLayer()).getNOut();
                        switch (kerasParamName) {
                            case PARAM_NAME_B_C:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.point(0),
                                        NDArrayIndex.interval(0, nOut)}, kerasParamValue);
                                break;
                            case PARAM_NAME_B_F:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.point(0),
                                        NDArrayIndex.interval(nOut, 2*nOut)}, kerasParamValue);
                                break;
                            case PARAM_NAME_B_O:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.point(0),
                                        NDArrayIndex.interval(2*nOut, 3*nOut)}, kerasParamValue);
                                break;
                            case PARAM_NAME_B_I:
                                dl4jParamValue.put(new INDArrayIndex[]{NDArrayIndex.point(0),
                                        NDArrayIndex.interval(3*nOut, 4*nOut)}, kerasParamValue);
                                break;
                        }
                    }
                }
                if (!layer.paramTable().keySet().contains(dl4JParamName))
                    throw new InvalidKerasConfigurationException("Layer " + layerName + ": Keras param " + kerasParamName + " maps to unknown param " + dl4JParamName);
                layer.setParam(dl4JParamName, dl4jParamValue);
            }
        }
        return model;
    }

    /**
     * Helper function to map (typical) Keras layer parameter names to
     * (typical) DL4J parameter names. This could be brittle.
     *
     * @param kerasParamName        Keras parameter name
     * @return                      DL4J parameter name
     *
     * TODO: should this be moved into KerasLayer?
     */
    private static String mapParameterName(String kerasParamName) {
        String paramName = null;
        switch (kerasParamName) {
            case PARAM_NAME_GAMMA:
                paramName = BatchNormalizationParamInitializer.GAMMA;
                break;
            case PARAM_NAME_BETA:
                paramName = BatchNormalizationParamInitializer.BETA;
                break;
            case PARAM_NAME_RUNNING_MEAN:
                paramName = BatchNormalizationParamInitializer.GLOBAL_MEAN;
                break;
            case PARAM_NAME_RUNNING_STD:
                paramName = BatchNormalizationParamInitializer.GLOBAL_VAR;
                break;
            case PARAM_NAME_W:
                paramName = DefaultParamInitializer.WEIGHT_KEY;
                break;
            case PARAM_NAME_B:
                paramName = DefaultParamInitializer.BIAS_KEY;
                break;
            case PARAM_NAME_W_C:
            case PARAM_NAME_W_F:
            case PARAM_NAME_W_I:
            case PARAM_NAME_W_O:
                paramName = GravesLSTMParamInitializer.INPUT_WEIGHT_KEY;
                break;
            case PARAM_NAME_U_C:
            case PARAM_NAME_U_F:
            case PARAM_NAME_U_I:
            case PARAM_NAME_U_O:
                paramName = GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY;
                break;
            case PARAM_NAME_B_C:
            case PARAM_NAME_B_F:
            case PARAM_NAME_B_I:
            case PARAM_NAME_B_O:
                paramName = GravesLSTMParamInitializer.BIAS_KEY;
                break;
        }
        return paramName;
    }
}
