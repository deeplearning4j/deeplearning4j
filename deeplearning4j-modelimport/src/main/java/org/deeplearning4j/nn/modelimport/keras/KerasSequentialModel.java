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
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasInput;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Build DL4J MultiLayerNetwork model from Keras Sequential
 * model configuration.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasSequentialModel extends KerasModel {

    /**
     * (Recommended) Builder-pattern constructor for Sequential model.
     *
     * @param modelBuilder    builder object
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public KerasSequentialModel(ModelBuilder modelBuilder) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        this(modelBuilder.modelJson, modelBuilder.modelYaml, modelBuilder.weightsArchive, modelBuilder.weightsRoot,
                modelBuilder.trainingJson, modelBuilder.trainingArchive, modelBuilder.enforceTrainingConfig);
    }

    /**
     * (Not recommended) Constructor for Sequential model from model configuration
     * (JSON or YAML), training configuration (JSON), weights, and "training mode"
     * boolean indicator. When built in training mode, certain unsupported configurations
     * (e.g., unknown regularizers) will throw Exceptions. When enforceTrainingConfig=false, these
     * will generate warnings but will be otherwise ignored.
     *
     * @param modelJson       model configuration JSON string
     * @param modelYaml       model configuration YAML string
     * @param trainingJson    training configuration JSON string
     * @throws IOException
     */
    public KerasSequentialModel(String modelJson, String modelYaml, Hdf5Archive weightsArchive, String weightsRoot,
                                String trainingJson, Hdf5Archive trainingArchive, boolean enforceTrainingConfig)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String,Object> modelConfig;
        if (modelJson != null)
            modelConfig = parseJsonString(modelJson);
        else if (modelYaml != null)
            modelConfig = parseYamlString(modelYaml);
        else
            throw new InvalidKerasConfigurationException("Requires model configuration as either JSON or YAML string.");

        /* Whether to enforce training-related configurations. */
        this.enforceTrainingConfig = enforceTrainingConfig;

        /* Determine model configuration type. */
        if (!modelConfig.containsKey(MODEL_FIELD_CLASS_NAME))
            throw new InvalidKerasConfigurationException("Could not determine Keras model class (no " + MODEL_FIELD_CLASS_NAME + " field found)");
        this.className = (String)modelConfig.get(MODEL_FIELD_CLASS_NAME);
        if (!this.className.equals(MODEL_CLASS_NAME_SEQUENTIAL))
            throw new InvalidKerasConfigurationException("Model class name must be " + MODEL_CLASS_NAME_SEQUENTIAL + " (found " + this.className + ")");

        /* Process layer configurations. */
        if (!modelConfig.containsKey(MODEL_FIELD_CONFIG))
            throw new InvalidKerasConfigurationException("Could not find layer configurations (no " + MODEL_FIELD_CONFIG + " field found)");
        helperPrepareLayers((List<Object>)modelConfig.get(MODEL_FIELD_CONFIG));

        KerasLayer inputLayer;
        if (this.layersOrdered.get(0) instanceof KerasInput) {
            inputLayer = this.layersOrdered.get(0);
	    } else {
            /* Add placeholder input layer and update lists of input and output layers. */
            int[] inputShape = this.layersOrdered.get(0).getInputShape();
            inputLayer = new KerasInput("input1", inputShape);
            inputLayer.setDimOrder(this.layersOrdered.get(0).getDimOrder());
            this.layers.put(inputLayer.getLayerName(), inputLayer);
            this.layersOrdered.add(0, inputLayer);
        }
        this.inputLayerNames = new ArrayList<String>(Arrays.asList(inputLayer.getLayerName()));
        this.outputLayerNames = new ArrayList<String>(Arrays.asList(this.layersOrdered.get(this.layersOrdered.size()-1).getLayerName()));

        /* Update each layer's inbound layer list to include (only) previous layer. */
        KerasLayer prevLayer = null;
        for (KerasLayer layer : this.layersOrdered) {
            if (prevLayer != null)
                layer.setInboundLayerNames(Arrays.asList(prevLayer.getLayerName()));
            prevLayer = layer;
        }

        /* Import training configuration. */
        if (trainingJson != null)
            helperImportTrainingConfiguration(trainingJson);

        /* Infer output types for each layer. */
        helperInferOutputTypes();

        /* Store weights in layers. */
        if (weightsArchive != null)
            helperImportWeights(weightsArchive, weightsRoot);
    }

    protected KerasSequentialModel() {}

    /**
     * Configure a MultiLayerConfiguration from this Keras Sequential model configuration.
     *
     * @return          MultiLayerConfiguration
     */
    public MultiLayerConfiguration getMultiLayerConfiguration()
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        if (!this.className.equals(MODEL_CLASS_NAME_SEQUENTIAL))
            throw new InvalidKerasConfigurationException("Keras model class name " + this.className + " incompatible with MultiLayerNetwork");
        if (this.inputLayerNames.size() != 1)
            throw new InvalidKerasConfigurationException("MultiLayeNetwork expects only 1 input (found " + this.inputLayerNames.size() + ")");
        if (this.outputLayerNames.size() != 1)
            throw new InvalidKerasConfigurationException("MultiLayeNetwork expects only 1 output (found " + this.outputLayerNames.size() + ")");

        NeuralNetConfiguration.Builder modelBuilder = new NeuralNetConfiguration.Builder();
        NeuralNetConfiguration.ListBuilder listBuilder = modelBuilder.list();

        /* Add layers one at a time. */
        KerasLayer prevLayer = null;
        int layerIndex = 0;
        for (KerasLayer layer : this.layersOrdered) {
            if (layer.usesRegularization())
                modelBuilder.setUseRegularization(true);
            if (layer.isLayer()) {
                int nbInbound = layer.getInboundLayerNames().size();
                if (nbInbound != 1)
                    throw new InvalidKerasConfigurationException("Layers in MultiLayerConfiguration must have exactly one inbound layer (found " + nbInbound + " for layer " + layer.getLayerName() + ")");
                listBuilder.layer(layerIndex++, layer.getLayer());
                if (prevLayer != null && prevLayer.isInputPreProcessor()) {
                    InputType[] inputTypes = new InputType[1];
                    inputTypes[0] = this.outputTypes.get(prevLayer.getInboundLayerNames().get(0));
                    InputPreProcessor preprocessor = prevLayer.getInputPreprocessor(inputTypes);
                    if (preprocessor != null)
                        listBuilder.inputPreProcessor(layerIndex-1, preprocessor);
                }
                if (this.outputLayerNames.contains(layer.getLayerName()) && !(layer.getLayer() instanceof IOutputLayer))
                    log.warn("Model cannot be trained: output layer " + layer.getLayerName() + " is not an IOutputLayer (no loss function specified)");
            }
            else if (layer.getVertex() != null)
                throw new InvalidKerasConfigurationException("Cannot add vertex to MultiLayerConfiguration (class name " + layer.getClassName() + ", layer name " + layer.getLayerName() + ")");
            prevLayer = layer;
        }

        InputType inputType = this.layersOrdered.get(0).getOutputType();
        if (inputType != null)
            listBuilder.setInputType(inputType);

        /* Whether to use standard backprop (or BPTT) or truncated BPTT. */
        if (this.useTruncatedBPTT && this.truncatedBPTT > 0)
            listBuilder.backpropType(BackpropType.TruncatedBPTT)
                    .tBPTTForwardLength(truncatedBPTT)
                    .tBPTTBackwardLength(truncatedBPTT);
        else
            listBuilder.backpropType(BackpropType.Standard);
        return listBuilder.build();
    }

    /**
     * Build a MultiLayerNetwork from this Keras Sequential model configuration.
     *
     * @return          MultiLayerNetwork
     */
    public MultiLayerNetwork getMultiLayerNetwork()
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        MultiLayerNetwork model = getMultiLayerNetwork(true);
        return model;
    }

    /**
     * Build a MultiLayerNetwork from this Keras Sequential model configuration and import weights.
     *
     * @return          MultiLayerNetwork
     */
    public MultiLayerNetwork getMultiLayerNetwork(boolean importWeights)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        MultiLayerNetwork model = new MultiLayerNetwork(getMultiLayerConfiguration());
        model.init();
        if (importWeights)
            model = (MultiLayerNetwork)helperCopyWeightsToModel(model);
        return model;
    }
}
