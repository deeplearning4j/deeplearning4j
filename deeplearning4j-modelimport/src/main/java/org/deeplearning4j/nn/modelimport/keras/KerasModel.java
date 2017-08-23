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

import org.deeplearning4j.nn.modelimport.keras.config.KerasModelConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasInput;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasLoss;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasLstm;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;
import org.deeplearning4j.util.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.deeplearning4j.nn.modelimport.keras.KerasLayer.DimOrder;

/**
 * Build ComputationGraph from Keras (Functional API) Model or
 * Sequential model configuration.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasModel {

    protected static KerasModelConfiguration config = new KerasModelConfiguration();
    protected KerasModelBuilder modelBuilder = new KerasModelBuilder(config);

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

    public KerasModelBuilder modelBuilder() {
        return this.modelBuilder;
    }
    /**
     * (Recommended) Builder-pattern constructor for (Functional API) Model.
     *
     * @param modelBuilder builder object
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasModel(KerasModelBuilder modelBuilder)
            throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        this(modelBuilder.getModelJson(), modelBuilder.getModelYaml(), modelBuilder.getWeightsArchive(),
                modelBuilder.getWeightsRoot(), modelBuilder.getTrainingJson(), modelBuilder.getTrainingArchive(),
                modelBuilder.isEnforceTrainingConfig());
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
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    protected KerasModel(String modelJson, String modelYaml, Hdf5Archive weightsArchive, String weightsRoot,
                         String trainingJson, Hdf5Archive trainingArchive, boolean enforceTrainingConfig)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> modelConfig;
        if (modelJson != null)
            modelConfig = parseJsonString(modelJson);
        else if (modelYaml != null)
            modelConfig = parseYamlString(modelYaml);
        else
            throw new InvalidKerasConfigurationException("Requires model configuration not found.");

        this.kerasMajorVersion = KerasModelUtils.determineKerasMajorVersion(modelConfig, config);
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
        this.inputLayerNames = new ArrayList<String>();
        for (Object inputLayerNameObj : (List<Object>) layerLists.get(config.getModelFieldInputLayers()))
            this.inputLayerNames.add((String) ((List<Object>) inputLayerNameObj).get(0));

        /* Construct list of output layers. */
        if (!layerLists.containsKey(config.getModelFieldOutputLayers()))
            throw new InvalidKerasConfigurationException("Could not find list of output layers (no "
                    + config.getModelFieldOutputLayers() + " field found)");

        this.outputLayerNames = new ArrayList<String>();
        for (Object outputLayerNameObj : (List<Object>) layerLists.get(config.getModelFieldOutputLayers()))
            this.outputLayerNames.add((String) ((List<Object>) outputLayerNameObj).get(0));

        /* Process layer configurations. */
        if (!layerLists.containsKey(config.getModelFieldLayers()))
            throw new InvalidKerasConfigurationException(
                    "Could not find layer configurations (no " + (config.getModelFieldLayers() + " field found)"));
        helperPrepareLayers((List<Object>) layerLists.get((config.getModelFieldLayers())));

        /* Import training configuration. */
        if (trainingJson != null)
            helperImportTrainingConfiguration(trainingJson);

        /* Infer output types for each layer. */
        helperInferOutputTypes();

        /* Store weights in layers. */
        if (weightsArchive != null)
            helperImportWeights(weightsArchive, weightsRoot);
    }

    /**
     * Helper method called from constructor. Converts layer configuration
     * JSON into KerasLayer objects.
     *
     * @param layerConfigs List of Keras layer configurations
     */
     void helperPrepareLayers(List<Object> layerConfigs)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this.layersOrdered = new ArrayList<>();
        this.layers = new HashMap<>();
        DimOrder dimOrder = DimOrder.NONE;
        for (Object layerConfig : layerConfigs) {
            Map<String, Object> layerConfigMap = (Map<String, Object>) layerConfig;
            // Append major keras version to each layer config.
            layerConfigMap.put(config.getFieldKerasVersion(), this.kerasMajorVersion);
            KerasLayer layer = new KerasLayer(this.kerasMajorVersion).getKerasLayerFromConfig(layerConfigMap,
                    this.enforceTrainingConfig);
            if (dimOrder == DimOrder.NONE && layer.getDimOrder() != DimOrder.NONE)
                dimOrder = layer.getDimOrder();
            this.layersOrdered.add(layer);
            this.layers.put(layer.getLayerName(), layer);
            if (layer instanceof KerasLstm)
                this.useTruncatedBPTT = this.useTruncatedBPTT || ((KerasLstm) layer).getUnroll();
        }

        for (KerasLayer layer : this.layersOrdered) {
            if (layer.getDimOrder() == DimOrder.NONE)
                layer.setDimOrder(dimOrder);
            else if (layer.getDimOrder() != dimOrder)
                throw new UnsupportedKerasConfigurationException("Keras layer " + layer.getLayerName()
                        + " has conflicting dim_ordering " + layer.getDimOrder() + " (vs. dimOrder)");
        }
    }

    /**
     * Helper method called from constructor. Incorporate training configuration details into model.
     * Includes loss function, optimization details, etc.
     *
     * @param trainingConfigJson JSON containing Keras training configuration
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    protected void helperImportTrainingConfiguration(String trainingConfigJson)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> trainingConfig = parseJsonString(trainingConfigJson);

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

        /* TODO: handle other configs (loss weights, sample weights). */
        /* TODO: handle optimizer configuration. */
    }

    /**
     * Helper method called from constructor. Infers and records output type
     * for every layer.
     */
    protected void helperInferOutputTypes()
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this.outputTypes = new HashMap<String, InputType>();
        for (KerasLayer layer : this.layersOrdered) {
            InputType outputType;
            if (layer instanceof KerasInput) {
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
     * Store weights to import with each associated Keras layer.
     *
     * @param weightsArchive Hdf5Archive
     * @param weightsRoot
     * @throws InvalidKerasConfigurationException
     */
    protected void helperImportWeights(Hdf5Archive weightsArchive, String weightsRoot)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        // check to ensure naming scheme doesn't include forward slash
        boolean includesSlash = false;
        for (String layerName : this.layers.keySet()) {
            if (layerName.contains("/"))
                includesSlash = true;
        }
        List<String> layerGroups;
        if (!includesSlash) {
            layerGroups = weightsRoot != null ? weightsArchive.getGroups(weightsRoot) : weightsArchive.getGroups();
        } else {
            layerGroups = new ArrayList<>(this.layers.keySet());
        }
        /* Set weights in KerasLayer for each entry in weights map. */
        for (String layerName : layerGroups) {
            List<String> layerParamNames;

            // there's a bug where if a layer name contains a forward slash, the first fragment must be appended
            // to the name of the dataset...it appears h5 interprets the forward slash as a data group
            String[] layerFragments = layerName.split("/");

            // Find nested groups when using Tensorflow
            String rootPrefix = weightsRoot != null ? weightsRoot + "/" : "";
            List<String> attributeStrParts = new ArrayList<>();
            String attributeStr = weightsArchive.readAttributeAsString(
                    "weight_names", rootPrefix + layerName
            );
            String attributeJoinStr;
            Matcher attributeMatcher = Pattern.compile(":\\d+").matcher(attributeStr);
            Boolean foundTfGroups = attributeMatcher.find();

            if (foundTfGroups) {
                for (String part : attributeStr.split("/")) {
                    part = part.trim();
                    if (part.length() == 0)
                        break;
                    Matcher tfSuffixMatcher = Pattern.compile(":\\d+").matcher(part);
                    if (tfSuffixMatcher.find())
                        break;
                    attributeStrParts.add(part);
                }
                attributeJoinStr = StringUtils.join("/", attributeStrParts);
            } else {
                attributeJoinStr = layerFragments[0];
            }

            String baseAttributes = layerName + "/" + attributeJoinStr;
            if (layerFragments.length > 1) {
                try {
                    layerParamNames = weightsArchive.getDataSets(rootPrefix + baseAttributes);
                } catch (Exception e) {
                    layerParamNames = weightsArchive.getDataSets(rootPrefix + layerName);
                }
            } else {
                if (foundTfGroups) {
                    layerParamNames = weightsArchive.getDataSets(rootPrefix + baseAttributes);
                } else {
                    layerParamNames = weightsArchive.getDataSets(rootPrefix + layerName);

                }
            }
            if (layerParamNames.isEmpty())
                continue;
            if (!this.layers.containsKey(layerName))
                throw new InvalidKerasConfigurationException(
                        "Found weights for layer not in model (named " + layerName + ")");
            KerasLayer layer = this.layers.get(layerName);
            if (layerParamNames.size() != layer.getNumParams())
                throw new InvalidKerasConfigurationException(
                        "Found " + layerParamNames.size() + " weights for layer with " + layer.getNumParams()
                                + " trainable params (named " + layerName + ")");
            Map<String, INDArray> weights = new HashMap<String, INDArray>();

            for (String layerParamName : layerParamNames) {
                String paramName = KerasModelUtils.findParameterName(layerParamName, layerFragments);
                INDArray paramValue;
                if (foundTfGroups) {
                    paramValue = weightsArchive.readDataSet(layerParamName, rootPrefix + baseAttributes);
                } else {
                    if (layerFragments.length > 1) {
                        paramValue = weightsArchive.readDataSet(layerFragments[0] + "/" + layerParamName, rootPrefix, layerName);
                    } else {
                        paramValue = weightsArchive.readDataSet(layerParamName, rootPrefix, layerName);
                    }
                }
                weights.put(paramName, paramValue);
            }
            layer.setWeights(weights);
        }

        /* Look for layers in model with no corresponding entries in weights map. */
        Set<String> layerNames = new HashSet<String>(this.layers.keySet());
        layerNames.removeAll(layerGroups);
        for (String layerName : layerNames) {
            if (this.layers.get(layerName).getNumParams() > 0)
                throw new InvalidKerasConfigurationException("Could not find weights required for layer " + layerName);
        }
    }

    protected KerasModel() {
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
        List<InputType> inputTypeList = new ArrayList<InputType>();
        for (String inputLayerName : this.inputLayerNames)
            inputTypeList.add(this.layers.get(inputLayerName).getOutputType());
        InputType[] inputTypes = new InputType[inputTypeList.size()];
        inputTypeList.toArray(inputTypes);
        graphBuilder.setInputTypes(inputTypes);

        /* Build String array of output layer names, add to ComputationGraph. */
        String[] outputLayerNameArray = new String[this.outputLayerNames.size()];
        this.outputLayerNames.toArray(outputLayerNameArray);
        graphBuilder.setOutputs(outputLayerNameArray);

        Map<String, InputPreProcessor> preprocessors = new HashMap<String, InputPreProcessor>();

        /* Add layersOrdered one at a time. */
        for (KerasLayer layer : this.layersOrdered) {
            /* Get inbound layer names. */
            List<String> inboundLayerNames = layer.getInboundLayerNames();
            String[] inboundLayerNamesArray = new String[inboundLayerNames.size()];
            inboundLayerNames.toArray(inboundLayerNamesArray);

            /* Get inbound InputTypes and InputPreProcessor, if necessary. */
            List<InputType> inboundTypeList = new ArrayList<InputType>();
            for (String layerName : inboundLayerNames)
                inboundTypeList.add(this.outputTypes.get(layerName));
            InputType[] inboundTypeArray = new InputType[inboundTypeList.size()];
            inboundTypeList.toArray(inboundTypeArray);
            InputPreProcessor preprocessor = layer.getInputPreprocessor(inboundTypeArray);

            if (layer.isLayer()) {
                /* Add DL4J layer. */
                if (preprocessor != null)
                    preprocessors.put(layer.getLayerName(), preprocessor);
                graphBuilder.addLayer(layer.getLayerName(), layer.getLayer(), inboundLayerNamesArray);
                if (this.outputLayerNames.contains(layer.getLayerName()) && !(layer.getLayer() instanceof IOutputLayer))
                    log.warn("Model cannot be trained: output layer " + layer.getLayerName()
                            + " is not an IOutputLayer (no loss function specified)");
            } else if (layer.isVertex()) { // Ignore "preprocessor" layers for now
                /* Add DL4J vertex. */
                if (preprocessor != null)
                    preprocessors.put(layer.getLayerName(), preprocessor);
                graphBuilder.addVertex(layer.getLayerName(), layer.getVertex(), inboundLayerNamesArray);
                if (this.outputLayerNames.contains(layer.getLayerName())
                        && !(layer.getVertex() instanceof IOutputLayer))
                    log.warn("Model cannot be trained: output vertex " + layer.getLayerName()
                            + " is not an IOutputLayer (no loss function specified)");
            } else if (layer.isInputPreProcessor()) {
                if (preprocessor == null)
                    throw new UnsupportedKerasConfigurationException("Layer " + layer.getLayerName()
                            + " could not be mapped to Layer, Vertex, or InputPreProcessor");
                graphBuilder.addVertex(layer.getLayerName(), new PreprocessorVertex(preprocessor),
                        inboundLayerNamesArray);
            }

            if (this.outputLayerNames.contains(layer.getLayerName()))
                log.warn("Model cannot be trained: output " + layer.getLayerName()
                        + " is not an IOutputLayer (no loss function specified)");
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

    /**
     * Convenience function for parsing JSON strings.
     *
     * @param json String containing valid JSON
     * @return Nested (key,value) map of arbitrary depth
     * @throws IOException
     */
    public static Map<String, Object> parseJsonString(String json) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        TypeReference<HashMap<String, Object>> typeRef = new TypeReference<HashMap<String, Object>>() {
        };
        return mapper.readValue(json, typeRef);
    }

    /**
     * Convenience function for parsing YAML strings.
     *
     * @param yaml String containing valid YAML
     * @return Nested (key,value) map of arbitrary depth
     * @throws IOException
     */
    public static Map<String, Object> parseYamlString(String yaml) throws IOException {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        TypeReference<HashMap<String, Object>> typeRef = new TypeReference<HashMap<String, Object>>() {
        };
        return mapper.readValue(yaml, typeRef);
    }
}
