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
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasModelConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasInput;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasLoss;
import org.deeplearning4j.nn.modelimport.keras.layers.recurrent.KerasLstm;
import org.deeplearning4j.nn.modelimport.keras.layers.recurrent.KerasSimpleRnn;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.customLayers;

/**
 * Build ComputationGraph from Keras (Functional API) Model or
 * Sequential model configuration.
 *
 * @author dave@skymind.io, Max Pumperla
 */
@Slf4j
public class KerasModel {

    protected static KerasModelConfiguration config = new KerasModelConfiguration();
    KerasModelBuilder modelBuilder = new KerasModelBuilder(config);

    protected String className; // Keras model class name
    protected boolean enforceTrainingConfig; // whether to build model in training mode
    protected Map<String, KerasLayer> layers; // map from layer name to KerasLayer
    List<KerasLayer> layersOrdered; // ordered list of layers
    Map<String, InputType> outputTypes; // inferred output types for all layers
    ArrayList<String> inputLayerNames; // list of input layers
    ArrayList<String> outputLayerNames; // list of output layers
    boolean useTruncatedBPTT = false; // whether to use truncated BPTT
    int truncatedBPTT = 0; // truncated BPTT value
    int kerasMajorVersion;
    String kerasBackend;

    public KerasModel() {
    }

    public KerasModelBuilder modelBuilder() {
        return this.modelBuilder;
    }

    /**
     * (Recommended) Builder-pattern constructor for (Functional API) Model.
     *
     * @param modelBuilder builder object
     * @throws IOException                            IO exception
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    public KerasModel(KerasModelBuilder modelBuilder)
            throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        this(modelBuilder.getModelJson(), modelBuilder.getModelYaml(), modelBuilder.getWeightsArchive(),
                modelBuilder.getWeightsRoot(), modelBuilder.getTrainingJson(), modelBuilder.getTrainingArchive(),
                modelBuilder.isEnforceTrainingConfig(), modelBuilder.getInputShape());
    }

    /**
     * (Not recommended) Constructor for (Functional API) Model from model configuration
     * (JSON or YAML), training configuration (JSON), weights, and "training mode"
     * boolean indicator. When built in training mode, certain unsupported configurations
     * (e.g., unknown regularizers) will throw Exceptions. When enforceTrainingConfig=false, these
     * will generate warnings but will be otherwise ignored.
     *
     * @param modelJson             model configuration JSON string
     * @param modelYaml             model configuration YAML string
     * @param enforceTrainingConfig whether to enforce training-related configurations
     * @throws IOException                            IO exception
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    protected KerasModel(String modelJson, String modelYaml, Hdf5Archive weightsArchive, String weightsRoot,
                         String trainingJson, Hdf5Archive trainingArchive, boolean enforceTrainingConfig,
                         int[] inputShape)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        Map<String, Object> modelConfig = KerasModelUtils.parseModelConfig(modelJson, modelYaml);
        this.kerasMajorVersion = KerasModelUtils.determineKerasMajorVersion(modelConfig, config);
        this.kerasBackend = KerasModelUtils.determineKerasBackend(modelConfig, config);
        this.enforceTrainingConfig = enforceTrainingConfig;

        /* Determine model configuration type. */
        if (!modelConfig.containsKey(config.getFieldClassName()))
            throw new InvalidKerasConfigurationException(
                    "Could not determine Keras model class (no " + config.getFieldClassName() + " field found)");
        this.className = (String) modelConfig.get(config.getFieldClassName());
        if (!this.className.equals(config.getFieldClassNameModel()))
            throw new InvalidKerasConfigurationException(
                    "Expected model class name " + config.getFieldClassNameModel() + " (found " + this.className + ")");


        /* Retrieve lists of input and output layers, layer configurations. */
        if (!modelConfig.containsKey(config.getModelFieldConfig()))
            throw new InvalidKerasConfigurationException("Could not find model configuration details (no "
                    + config.getModelFieldConfig() + " in model config)");
        Map<String, Object> layerLists = (Map<String, Object>) modelConfig.get(config.getModelFieldConfig());


        /* Construct list of input layers. */
        if (!layerLists.containsKey(config.getModelFieldInputLayers()))
            throw new InvalidKerasConfigurationException("Could not find list of input layers (no "
                    + config.getModelFieldInputLayers() + " field found)");
        this.inputLayerNames = new ArrayList<>();
        for (Object inputLayerNameObj : (List<Object>) layerLists.get(config.getModelFieldInputLayers()))
            this.inputLayerNames.add((String) ((List<Object>) inputLayerNameObj).get(0));

        /* Construct list of output layers. */
        if (!layerLists.containsKey(config.getModelFieldOutputLayers()))
            throw new InvalidKerasConfigurationException("Could not find list of output layers (no "
                    + config.getModelFieldOutputLayers() + " field found)");

        this.outputLayerNames = new ArrayList<>();
        for (Object outputLayerNameObj : (List<Object>) layerLists.get(config.getModelFieldOutputLayers()))
            this.outputLayerNames.add((String) ((List<Object>) outputLayerNameObj).get(0));

        /* Process layer configurations. */
        if (!layerLists.containsKey(config.getModelFieldLayers()))
            throw new InvalidKerasConfigurationException(
                    "Could not find layer configurations (no " + (config.getModelFieldLayers() + " field found)"));
        prepareLayers((List<Object>) layerLists.get((config.getModelFieldLayers())));

        /* Import training configuration. */
        if (trainingJson != null && enforceTrainingConfig)
            importTrainingConfiguration(trainingJson);

        /* Infer output types for each layer. */
        inferOutputTypes(inputShape);

        /* Store weights in layers. */
        if (weightsArchive != null)
            KerasModelUtils.importWeights(weightsArchive, weightsRoot, layers, kerasMajorVersion, kerasBackend);
    }

    /**
     * Helper method called from constructor. Converts layer configuration
     * JSON into KerasLayer objects.
     *
     * @param layerConfigs List of Keras layer configurations
     */
    void prepareLayers(List<Object> layerConfigs)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this.layersOrdered = new ArrayList<>();
        this.layers = new HashMap<>();
        for (Object layerConfig : layerConfigs) {
            Map<String, Object> layerConfigMap = (Map<String, Object>) layerConfig;
            // Append major keras version and backend to each layer config.
            layerConfigMap.put(config.getFieldKerasVersion(), this.kerasMajorVersion);
            if (kerasMajorVersion == 2 && this.kerasBackend != null)
                layerConfigMap.put(config.getFieldBackend(), this.kerasBackend);

            KerasLayerConfiguration kerasLayerConf = new KerasLayer(this.kerasMajorVersion).conf;
            KerasLayer layer = KerasLayerUtils.getKerasLayerFromConfig(layerConfigMap, this.enforceTrainingConfig,
                    kerasLayerConf, customLayers, this.layers);
            this.layersOrdered.add(layer);
            this.layers.put(layer.getLayerName(), layer);
            if (layer instanceof KerasLstm)
                this.useTruncatedBPTT = this.useTruncatedBPTT || ((KerasLstm) layer).getUnroll();
            if (layer instanceof KerasSimpleRnn)
                this.useTruncatedBPTT = this.useTruncatedBPTT || ((KerasSimpleRnn) layer).getUnroll();
        }
    }

    /**
     * Helper method called from constructor. Incorporate training configuration details into model.
     * Includes loss function, optimization details, etc.
     *
     * @param trainingConfigJson JSON containing Keras training configuration
     * @throws IOException                            IO exception
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     */
    void importTrainingConfiguration(String trainingConfigJson)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> trainingConfig = KerasModelUtils.parseJsonString(trainingConfigJson);

        /* Add loss layers for each loss function. */
        List<KerasLayer> lossLayers = new ArrayList<>();
        if (!trainingConfig.containsKey(config.getTrainingLoss()))
            throw new InvalidKerasConfigurationException("Could not determine training loss function (no "
                    + config.getTrainingLoss() + " field found in training config)");
        Object kerasLossObj = trainingConfig.get(config.getTrainingLoss());

        if (kerasLossObj instanceof String) {
            String kerasLoss = (String) kerasLossObj;
            for (String outputLayerName : this.outputLayerNames)
                lossLayers.add(new KerasLoss(outputLayerName + "_loss", outputLayerName, kerasLoss));
        } else if (kerasLossObj instanceof Map) {
            Map<String, Object> kerasLossMap = (Map<String, Object>) kerasLossObj;
            for (String outputLayerName : kerasLossMap.keySet()) {
                Object kerasLoss = kerasLossMap.get(outputLayerName);
                if (kerasLoss instanceof String)
                    lossLayers.add(new KerasLoss(outputLayerName + "_loss", outputLayerName, (String) kerasLoss));
                else
                    throw new InvalidKerasConfigurationException("Unknown Keras loss " + kerasLoss.toString());
            }
        }
        this.outputLayerNames.clear();

        /* Add loss layers to output layer list and layer graph. */
        for (KerasLayer lossLayer : lossLayers) {
            this.layersOrdered.add(lossLayer);
            this.layers.put(lossLayer.getLayerName(), lossLayer);
            this.outputLayerNames.add(lossLayer.getLayerName());
        }
    }

    /**
     * Helper method called from constructor. Infers and records output type
     * for every layer.
     */
    void inferOutputTypes(int[] inputShape)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this.outputTypes = new HashMap<>();
        for (KerasLayer layer : this.layersOrdered) {
            InputType outputType;
            if (layer instanceof KerasInput) {
                if (inputShape != null) {
                    layer.inputShape = inputShape;
                }
                outputType = layer.getOutputType();
                this.truncatedBPTT = ((KerasInput) layer).getTruncatedBptt();
            } else {
                InputType[] inputTypes = new InputType[layer.getInboundLayerNames().size()];
                int i = 0;
                for (String inboundLayerName : layer.getInboundLayerNames())
                    inputTypes[i++] = this.outputTypes.get(inboundLayerName);
                outputType = layer.getOutputType(inputTypes);
            }
            this.outputTypes.put(layer.getLayerName(), outputType);
        }
    }

    /**
     * Configure a ComputationGraph from this Keras Model configuration.
     *
     * @return ComputationGraph
     */
    public ComputationGraphConfiguration getComputationGraphConfiguration()
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        if (!this.className.equals(config.getFieldClassNameModel()) && !this.className.equals(config.getFieldClassNameSequential()))

            throw new InvalidKerasConfigurationException(
                    "Keras model class name " + this.className + " incompatible with ComputationGraph");
        NeuralNetConfiguration.Builder modelBuilder = new NeuralNetConfiguration.Builder();

        ComputationGraphConfiguration.GraphBuilder graphBuilder = modelBuilder.graphBuilder();

        /* Build String array of input layer names, add to ComputationGraph. */
        String[] inputLayerNameArray = new String[this.inputLayerNames.size()];
        this.inputLayerNames.toArray(inputLayerNameArray);
        graphBuilder.addInputs(inputLayerNameArray);

        /* Build InputType array of input layer types, add to ComputationGraph. */
        List<InputType> inputTypeList = new ArrayList<>();
        for (String inputLayerName : this.inputLayerNames)
            inputTypeList.add(this.layers.get(inputLayerName).getOutputType());
        InputType[] inputTypes = new InputType[inputTypeList.size()];
        inputTypeList.toArray(inputTypes);
        graphBuilder.setInputTypes(inputTypes);

        /* Build String array of output layer names, add to ComputationGraph. */
        String[] outputLayerNameArray = new String[this.outputLayerNames.size()];
        this.outputLayerNames.toArray(outputLayerNameArray);
        graphBuilder.setOutputs(outputLayerNameArray);

        Map<String, InputPreProcessor> preprocessors = new HashMap<>();

        /* Add layersOrdered one at a time. */
        for (KerasLayer layer : this.layersOrdered) {
            /* Get inbound layer names. */
            List<String> inboundLayerNames = layer.getInboundLayerNames();
            String[] inboundLayerNamesArray = new String[inboundLayerNames.size()];
            inboundLayerNames.toArray(inboundLayerNamesArray);

            /* Get inbound InputTypes and InputPreProcessor, if necessary. */
            List<InputType> inboundTypeList = new ArrayList<>();
            for (String layerName : inboundLayerNames)
                inboundTypeList.add(this.outputTypes.get(layerName));
            InputType[] inboundTypeArray = new InputType[inboundTypeList.size()];
            inboundTypeList.toArray(inboundTypeArray);
            InputPreProcessor preprocessor = layer.getInputPreprocessor(inboundTypeArray);

            if (layer.isLayer()) {
                if (preprocessor != null)
                    preprocessors.put(layer.getLayerName(), preprocessor);
                graphBuilder.addLayer(layer.getLayerName(), layer.getLayer(), inboundLayerNamesArray);
            } else if (layer.isVertex()) { // Ignore "preprocessor" layers for now
                if (preprocessor != null)
                    preprocessors.put(layer.getLayerName(), preprocessor);
                graphBuilder.addVertex(layer.getLayerName(), layer.getVertex(), inboundLayerNamesArray);
            } else if (layer.isInputPreProcessor()) {
                if (preprocessor == null)
                    throw new UnsupportedKerasConfigurationException("Layer " + layer.getLayerName()
                            + " could not be mapped to Layer, Vertex, or InputPreProcessor");
                graphBuilder.addVertex(layer.getLayerName(), new PreprocessorVertex(preprocessor),
                        inboundLayerNamesArray);
            }
        }
        graphBuilder.setInputPreProcessors(preprocessors);

        /* Whether to use standard backprop (or BPTT) or truncated BPTT. */
        if (this.useTruncatedBPTT && this.truncatedBPTT > 0)
            graphBuilder.backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(truncatedBPTT)
                    .tBPTTBackwardLength(truncatedBPTT);
        else
            graphBuilder.backpropType(BackpropType.Standard);
        return graphBuilder.build();
    }

    /**
     * Build a ComputationGraph from this Keras Model configuration and import weights.
     *
     * @return ComputationGraph
     */
    public ComputationGraph getComputationGraph()
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return getComputationGraph(true);
    }

    /**
     * Build a ComputationGraph from this Keras Model configuration and (optionally) import weights.
     *
     * @param importWeights whether to import weights
     * @return ComputationGraph
     */
    public ComputationGraph getComputationGraph(boolean importWeights)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        ComputationGraph model = new ComputationGraph(getComputationGraphConfiguration());
        model.init();
        if (importWeights)
            model = (ComputationGraph) KerasModelUtils.copyWeightsToModel(model, this.layers);
        return model;
    }
}
