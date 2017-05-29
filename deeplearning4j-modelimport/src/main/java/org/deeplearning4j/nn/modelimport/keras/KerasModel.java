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

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasInput;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasLoss;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasLstm;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
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
    public static final String HDF5_MODEL_WEIGHTS_ROOT = "model_weights";
    public static final String HDF5_MODEL_CONFIG_ATTRIBUTE = "model_config";
    public static final String HDF5_TRAINING_CONFIG_ATTRIBUTE = "training_config";

    protected String className; // Keras model class name
    protected boolean enforceTrainingConfig; // whether to build model in training mode
    protected List<KerasLayer> layersOrdered; // ordered list of layers
    protected Map<String, KerasLayer> layers; // map from layer name to KerasLayer
    protected Map<String, InputType> outputTypes; // inferred output types for all layers
    protected ArrayList<String> inputLayerNames; // list of input layers
    protected ArrayList<String> outputLayerNames; // list of output layers
    protected boolean useTruncatedBPTT = false; // whether to use truncated BPTT
    protected int truncatedBPTT = 0; // truncated BPTT value

    /**
     * (Recommended) Builder-pattern constructor for (Functional API) Model.
     *
     * @param modelBuilder    builder object
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasModel(ModelBuilder modelBuilder)
                    throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        this(modelBuilder.modelJson, modelBuilder.modelYaml, modelBuilder.weightsArchive, modelBuilder.weightsRoot,
                        modelBuilder.trainingJson, modelBuilder.trainingArchive, modelBuilder.enforceTrainingConfig);
    }

    /**
     * (Not recommended) Constructor for (Functional API) Model from model configuration
     * (JSON or YAML), training configuration (JSON), weights, and "training mode"
     * boolean indicator. When built in training mode, certain unsupported configurations
     * (e.g., unknown regularizers) will throw Exceptions. When enforceTrainingConfig=false, these
     * will generate warnings but will be otherwise ignored.
     *
     * @param modelJson                 model configuration JSON string
     * @param modelYaml                 model configuration YAML string
     * @param enforceTrainingConfig     whether to enforce training-related configurations
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

        /* Whether to enforce training-related configurations. */
        this.enforceTrainingConfig = enforceTrainingConfig;

        /* Determine model configuration type. */
        if (!modelConfig.containsKey(MODEL_FIELD_CLASS_NAME))
            throw new InvalidKerasConfigurationException(
                            "Could not determine Keras model class (no " + MODEL_FIELD_CLASS_NAME + " field found)");
        this.className = (String) modelConfig.get(MODEL_FIELD_CLASS_NAME);
        if (!this.className.equals(MODEL_CLASS_NAME_MODEL))
            throw new InvalidKerasConfigurationException(
                            "Expected model class name " + MODEL_CLASS_NAME_MODEL + " (found " + this.className + ")");

        /* Retrieve lists of input and output layers, layer configurations. */
        if (!modelConfig.containsKey(MODEL_FIELD_CONFIG))
            throw new InvalidKerasConfigurationException("Could not find model configuration details (no "
                            + MODEL_FIELD_CONFIG + " in model config)");
        Map<String, Object> layerLists = (Map<String, Object>) modelConfig.get(MODEL_FIELD_CONFIG);

        /* Construct list of input layers. */
        if (!layerLists.containsKey(MODEL_CONFIG_FIELD_INPUT_LAYERS))
            throw new InvalidKerasConfigurationException("Could not find list of input layers (no "
                            + MODEL_CONFIG_FIELD_INPUT_LAYERS + " field found)");
        this.inputLayerNames = new ArrayList<String>();
        for (Object inputLayerNameObj : (List<Object>) layerLists.get(MODEL_CONFIG_FIELD_INPUT_LAYERS))
            this.inputLayerNames.add((String) ((List<Object>) inputLayerNameObj).get(0));

        /* Construct list of output layers. */
        if (!layerLists.containsKey(MODEL_CONFIG_FIELD_OUTPUT_LAYERS))
            throw new InvalidKerasConfigurationException("Could not find list of output layers (no "
                            + MODEL_CONFIG_FIELD_OUTPUT_LAYERS + " field found)");
        this.outputLayerNames = new ArrayList<String>();
        for (Object outputLayerNameObj : (List<Object>) layerLists.get(MODEL_CONFIG_FIELD_OUTPUT_LAYERS))
            this.outputLayerNames.add((String) ((List<Object>) outputLayerNameObj).get(0));

        /* Process layer configurations. */
        if (!layerLists.containsKey(MODEL_CONFIG_FIELD_LAYERS))
            throw new InvalidKerasConfigurationException(
                            "Could not find layer configurations (no " + MODEL_CONFIG_FIELD_LAYERS + " field found)");
        helperPrepareLayers((List<Object>) layerLists.get(MODEL_CONFIG_FIELD_LAYERS));

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
     * @param layerConfigs      List of Keras layer configurations
     */
    protected void helperPrepareLayers(List<Object> layerConfigs)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this.layersOrdered = new ArrayList<KerasLayer>();
        this.layers = new HashMap<String, KerasLayer>();
        DimOrder dimOrder = DimOrder.NONE;
        for (Object layerConfig : layerConfigs) {
            KerasLayer layer = KerasLayer.getKerasLayerFromConfig((Map<String, Object>) layerConfig,
                            this.enforceTrainingConfig);
            if (dimOrder == DimOrder.NONE && layer.getDimOrder() != DimOrder.NONE) // determine dimension order, if any
                dimOrder = layer.getDimOrder();
            this.layersOrdered.add(layer);
            this.layers.put(layer.getLayerName(), layer);
            if (layer instanceof KerasLstm)
                this.useTruncatedBPTT = this.useTruncatedBPTT || ((KerasLstm) layer).getUnroll();
        }

        /* Set dimension ordering for all layers to dimOrder found above.
         * NOTE: this currently assumes that only one dimension ordering is used
         * throughout the model.
         */
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
     * @param trainingConfigJson        JSON containing Keras training configuration
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    protected void helperImportTrainingConfiguration(String trainingConfigJson)
                    throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> trainingConfig = parseJsonString(trainingConfigJson);

        /* Add loss layers for each loss function. */
        List<KerasLayer> lossLayers = new ArrayList<KerasLayer>();
        if (!trainingConfig.containsKey(TRAINING_CONFIG_FIELD_LOSS))
            throw new InvalidKerasConfigurationException("Could not determine training loss function (no "
                            + TRAINING_CONFIG_FIELD_LOSS + " field found in training config)");
        Object kerasLossObj = trainingConfig.get(TRAINING_CONFIG_FIELD_LOSS);
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
            InputType outputType = null;
            if (layer instanceof KerasInput) {
                outputType = layer.getOutputType();
                /*
                 * TODO: figure out how to infer truncated BPTT value for non-sequence inputs
                 *
                 * In Keras, truncated BPTT is specified implicitly by specifying a fixed
                 * size input and by passing in the "unroll" argument to recurrent layers.
                 * Currently, the only setting in which we can confidently determine the
                 * value of truncated BPTT is if the original input has two dimensions,
                 * the first of which is sequence length. Hypothetically, we should be
                 * able to do this for other types of inputs, but that's less straightforward.
                 */
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

    protected List<String> helperRecurseWeightsArchive(Hdf5Archive weightsArchive, String weightsRoot, String layerName){
        return new LinkedList<>();
    }

    /**
     * Store weights to import with each associated Keras layer.
     *
     * @param weightsArchive    Hdf5Archive
     * @param weightsRoot
     * @throws InvalidKerasConfigurationException
     */
    protected void helperImportWeights(Hdf5Archive weightsArchive, String weightsRoot)
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        // check to ensure naming scheme doesn't include forward slash
        boolean includesSlash = false;
        for(String layerName : this.layers.keySet()) {
            if(layerName.contains("/"))
                includesSlash = true;
        }
        List<String> layerGroups;
        if(!includesSlash) {
            layerGroups =
                    weightsRoot != null ? weightsArchive.getGroups(weightsRoot) : weightsArchive.getGroups();
        } else {
            layerGroups = new ArrayList<>(this.layers.keySet());
        }
        /* Set weights in KerasLayer for each entry in weights map. */
        for (String layerName : layerGroups) {
            List<String> layerParamNames;

            // there's a bug where if a layer name contains a forward slash, the first fragment must be appended
            // to the name of the dataset...it appears h5 interprets the forward slash as a data group
            String[] layerFragments = layerName.split("/");
            if(layerFragments.length > 1) {
                try {
                    layerParamNames = weightsRoot != null
                            ? weightsArchive.getDataSets(weightsRoot, layerName + "/" + layerFragments[0])
                            : weightsArchive.getDataSets(layerName + "/" + layerFragments[0]);
                } catch (Exception e) {
                    // TODO: fix this horrible ugliness
                    layerParamNames = weightsRoot != null
                            ? weightsArchive.getDataSets(weightsRoot, layerName)
                            : weightsArchive.getDataSets(layerName);
                }
            } else {
                layerParamNames = weightsRoot != null
                    ? weightsArchive.getDataSets(weightsRoot, layerName)
                    : weightsArchive.getDataSets(layerName);
            }
            if (layerParamNames.isEmpty())
                continue;
            if (!layerParamNames.isEmpty() && !this.layers.containsKey(layerName))
                throw new InvalidKerasConfigurationException(
                                "Found weights for layer not in model (named " + layerName + ")");
            KerasLayer layer = this.layers.get(layerName);
            if (layerParamNames.size() != layer.getNumParams())
                throw new InvalidKerasConfigurationException(
                                "Found " + layerParamNames.size() + " weights for layer with " + layer.getNumParams()
                                                + " trainable params (named " + layerName + ")");
            Map<String, INDArray> weights = new HashMap<String, INDArray>();
            for (String layerParamName : layerParamNames) {
                /* TODO: push this logic into KerasLayer subclasses. Layers know what
                 * parameters they have and should be looking for, so let them handle
                 * it in a layer-specific manner.
                 */

                /* Keras parameter names are typically formatted as [layer name]_[parameter]. For
                 * example, the weight matrix in the first Dense layer with the TensorFlow backend
                 * will be named "dense_1_W:0."
                 */
                // TODO fix the SLASH issue with layer names
                Matcher layerNameMatcher = Pattern.compile(layerFragments[layerFragments.length-1]).matcher(layerParamName);
                if (!layerNameMatcher.find())
                    throw new InvalidKerasConfigurationException(
                                    "Unable to parse layer/parameter name " + layerParamName + " for stored weights.");
                String paramName = layerNameMatcher.replaceFirst("");

                /* Usually layer name is separated from parameter name by an underscore. */
                Matcher paramNameMatcher = Pattern.compile("^_(.+)$").matcher(paramName);
                if (paramNameMatcher.find())
                    paramName = paramNameMatcher.group(1);

                /* TensorFlow backend often appends ":" followed by one or more digits to parameter
                 * names. We strip it off here.
                 */
                Matcher tfSuffixMatcher = Pattern.compile(":\\d+?$").matcher(paramName);
                if (tfSuffixMatcher.find())
                    paramName = tfSuffixMatcher.replaceFirst("");

                /* TensorFlow backend also may append "_" followed by one or more digits to parameter
                 * names. We strip it off here.
                 */
                Matcher tfParamNbMatcher = Pattern.compile("_\\d+$").matcher(paramName);
                if (tfParamNbMatcher.find())
                    paramName = tfParamNbMatcher.replaceFirst("");

                INDArray paramValue;
                if(layerFragments.length > 1) {
                    paramValue =
                            weightsRoot != null ? weightsArchive.readDataSet(layerFragments[0]+"/"+layerParamName, weightsRoot, layerName)
                                    : weightsArchive.readDataSet(layerParamName, layerName);
                } else {
                    paramValue =
                            weightsRoot != null ? weightsArchive.readDataSet(layerParamName, weightsRoot, layerName)
                                    : weightsArchive.readDataSet(layerParamName, layerName);
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

    protected KerasModel() {}

    /**
     * Configure a ComputationGraph from this Keras Model configuration.
     *
     * @return          ComputationGraph
     */
    public ComputationGraphConfiguration getComputationGraphConfiguration()
                    throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        if (!this.className.equals(MODEL_CLASS_NAME_MODEL) && !this.className.equals(MODEL_CLASS_NAME_SEQUENTIAL))
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

            if (layer.usesRegularization())
                modelBuilder.setUseRegularization(true);

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
            model = (ComputationGraph) helperCopyWeightsToModel(model);
        return model;
    }

    @Data
    static class ModelBuilder implements Cloneable {
        protected String modelJson = null;
        protected String modelYaml = null;
        protected String trainingJson = null;
        protected Hdf5Archive weightsArchive = null;
        protected String weightsRoot = null;
        protected Hdf5Archive trainingArchive = null;
        protected boolean enforceTrainingConfig = false;

        public ModelBuilder() {}

        public ModelBuilder modelJson(String modelJson) {
            this.modelJson = modelJson;
            return this;
        }

        public ModelBuilder modelJsonFilename(String modelJsonFilename) throws IOException {
            this.modelJson = new String(Files.readAllBytes(Paths.get(modelJsonFilename)));;
            return this;
        }

        public ModelBuilder modelJsonInputStream(InputStream modelJsonInputStream) throws IOException {
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            IOUtils.copy(modelJsonInputStream, byteArrayOutputStream);
            this.modelJson = new String(byteArrayOutputStream.toByteArray());
            return this;
        }

        public ModelBuilder modelYaml(String modelYaml) {
            this.modelYaml = modelYaml;
            return this;
        }

        public ModelBuilder modelYamlFilename(String modelYamlFilename) throws IOException {
            this.modelJson = new String(Files.readAllBytes(Paths.get(modelYamlFilename)));
            return this;
        }

        public ModelBuilder modelYamlInputStream(InputStream modelYamlInputStream) throws IOException {
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            IOUtils.copy(modelYamlInputStream, byteArrayOutputStream);
            this.modelJson = new String(byteArrayOutputStream.toByteArray());
            return this;
        }

        public ModelBuilder trainingJson(String trainingJson) {
            this.trainingJson = trainingJson;
            return this;
        }

        public ModelBuilder trainingJsonInputStream(InputStream trainingJsonInputStream) throws IOException {
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            IOUtils.copy(trainingJsonInputStream, byteArrayOutputStream);
            this.trainingJson = new String(byteArrayOutputStream.toByteArray());
            return this;
        }

        public ModelBuilder modelHdf5Filename(String modelHdf5Filename)
                        throws UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
            this.weightsArchive = this.trainingArchive = new Hdf5Archive(modelHdf5Filename);
            this.weightsRoot = HDF5_MODEL_WEIGHTS_ROOT;
            if (!this.weightsArchive.hasAttribute(HDF5_MODEL_CONFIG_ATTRIBUTE))
                throw new InvalidKerasConfigurationException(
                                "Model configuration attribute missing from " + modelHdf5Filename + " archive.");
            this.modelJson = this.weightsArchive.readAttributeAsJson(HDF5_MODEL_CONFIG_ATTRIBUTE);
            if (this.trainingArchive.hasAttribute(HDF5_TRAINING_CONFIG_ATTRIBUTE))
                this.trainingJson = this.trainingArchive.readAttributeAsJson(HDF5_TRAINING_CONFIG_ATTRIBUTE);
            return this;
        }

        public ModelBuilder weightsHdf5Filename(String weightsHdf5Filename) {
            this.weightsArchive = new Hdf5Archive(weightsHdf5Filename);
            return this;
        }

        public ModelBuilder enforceTrainingConfig(boolean enforceTrainingConfig) {
            this.enforceTrainingConfig = enforceTrainingConfig;
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
     * Convenience function for parsing JSON strings.
     *
     * @param json    String containing valid JSON
     * @return        Nested (key,value) map of arbitrary depth
     * @throws IOException
     */
    public static Map<String, Object> parseJsonString(String json) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        TypeReference<HashMap<String, Object>> typeRef = new TypeReference<HashMap<String, Object>>() {};
        return mapper.readValue(json, typeRef);
    }

    /**
     * Convenience function for parsing JSON strings.
     *
     * @param json    String containing valid JSON
     * @return        Nested (key,value) map of arbitrary depth
     * @throws IOException
     */
    public static Map<String, Object> parseYamlString(String json) throws IOException {
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        TypeReference<HashMap<String, Object>> typeRef = new TypeReference<HashMap<String, Object>>() {};
        return mapper.readValue(json, typeRef);
    }

    /**
     * Helper function to import weights from nested Map into existing model. Depends critically
     * on matched layer and parameter names. In general this seems to be straightforward for most
     * Keras models and layersOrdered, but there may be edge cases.
     *
     * @param model             DL4J Model interface
     * @return                  DL4J Model interface
     * @throws InvalidKerasConfigurationException
     */
    protected org.deeplearning4j.nn.api.Model helperCopyWeightsToModel(org.deeplearning4j.nn.api.Model model)
                    throws InvalidKerasConfigurationException {
        /* Get list if layers from model. */
        org.deeplearning4j.nn.api.Layer[] layersFromModel;
        if (model instanceof MultiLayerNetwork)
            layersFromModel = ((MultiLayerNetwork) model).getLayers();
        else
            layersFromModel = ((ComputationGraph) model).getLayers();

        /* Iterate over layers in model, setting weights when relevant. */
        Set<String> layerNames = new HashSet<>(this.layers.keySet());
        for (org.deeplearning4j.nn.api.Layer layer : layersFromModel) {
            String layerName = layer.conf().getLayer().getLayerName();
            if (!this.layers.containsKey(layerName))
                throw new InvalidKerasConfigurationException(
                                "No weights found for layer in model (named " + layerName + ")");
            this.layers.get(layerName).copyWeightsToLayer(layer);
            layerNames.remove(layerName);
        }

        for (String layerName : layerNames) {
            if (this.layers.get(layerName).getNumParams() > 0)
                throw new InvalidKerasConfigurationException(
                                "Attemping to copy weights for layer not in model (named " + layerName + ")");
        }
        return model;
    }
}
