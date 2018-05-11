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
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasInput;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
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
     * @param modelBuilder builder object
     * @throws IOException                            I/O exception
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration
     */
    public KerasSequentialModel(KerasModelBuilder modelBuilder)
            throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        this(modelBuilder.getModelJson(), modelBuilder.getModelYaml(), modelBuilder.getWeightsArchive(),
                modelBuilder.getWeightsRoot(), modelBuilder.getTrainingJson(), modelBuilder.getTrainingArchive(),
                modelBuilder.isEnforceTrainingConfig(), modelBuilder.getInputShape());
    }

    /**
     * (Not recommended) Constructor for Sequential model from model configuration
     * (JSON or YAML), training configuration (JSON), weights, and "training mode"
     * boolean indicator. When built in training mode, certain unsupported configurations
     * (e.g., unknown regularizers) will throw Exceptions. When enforceTrainingConfig=false, these
     * will generate warnings but will be otherwise ignored.
     *
     * @param modelJson    model configuration JSON string
     * @param modelYaml    model configuration YAML string
     * @param trainingJson training configuration JSON string
     * @throws IOException I/O exception
     */
    public KerasSequentialModel(String modelJson, String modelYaml, Hdf5Archive weightsArchive, String weightsRoot,
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
        if (!this.className.equals(config.getFieldClassNameSequential()))
            throw new InvalidKerasConfigurationException("Model class name must be " + config.getFieldClassNameSequential()
                    + " (found " + this.className + ")");

        /* Process layer configurations. */
        if (!modelConfig.containsKey(config.getModelFieldConfig()))
            throw new InvalidKerasConfigurationException(
                    "Could not find layer configurations (no " + config.getModelFieldConfig() + " field found)");
        prepareLayers((List<Object>) modelConfig.get(config.getModelFieldConfig()));

        KerasLayer inputLayer;
        if (this.layersOrdered.get(0) instanceof KerasInput) {
            inputLayer = this.layersOrdered.get(0);
        } else {
            /* Add placeholder input layer and update lists of input and output layers. */
            int[] firstLayerInputShape = this.layersOrdered.get(0).getInputShape();
            inputLayer = new KerasInput("input1", firstLayerInputShape);
            inputLayer.setDimOrder(this.layersOrdered.get(0).getDimOrder());
            this.layers.put(inputLayer.getLayerName(), inputLayer);
            this.layersOrdered.add(0, inputLayer);
        }
        this.inputLayerNames = new ArrayList<>(Collections.singletonList(inputLayer.getLayerName()));
        this.outputLayerNames = new ArrayList<>(
                Collections.singletonList(this.layersOrdered.get(this.layersOrdered.size() - 1).getLayerName()));

        /* Update each layer's inbound layer list to include (only) previous layer. */
        KerasLayer prevLayer = null;
        for (KerasLayer layer : this.layersOrdered) {
            if (prevLayer != null)
                layer.setInboundLayerNames(Collections.singletonList(prevLayer.getLayerName()));
            prevLayer = layer;
        }

        /* Import training configuration. */
        if (enforceTrainingConfig) {
            if (trainingJson != null)
                importTrainingConfiguration(trainingJson);
            else log.warn("If enforceTrainingConfig is true, a training " +
                    "configuration object has to be provided. Usually the only practical way to do this is to store" +
                    " your keras model with `model.save('model_path.h5'. If you store model config and weights" +
                    " separately no training configuration is attached.");
        }
        /* Infer output types for each layer. */
        inferOutputTypes(inputShape);

        /* Store weights in layers. */
        if (weightsArchive != null)
            KerasModelUtils.importWeights(weightsArchive, weightsRoot, layers, kerasMajorVersion, kerasBackend);
    }

    public KerasSequentialModel() {
        super();
    }

    /**
     * Configure a MultiLayerConfiguration from this Keras Sequential model configuration.
     *
     * @return MultiLayerConfiguration
     */
    public MultiLayerConfiguration getMultiLayerConfiguration()
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        if (!this.className.equals(config.getFieldClassNameSequential()))
            throw new InvalidKerasConfigurationException(
                    "Keras model class name " + this.className + " incompatible with MultiLayerNetwork");
        if (this.inputLayerNames.size() != 1)
            throw new InvalidKerasConfigurationException(
                    "MultiLayeNetwork expects only 1 input (found " + this.inputLayerNames.size() + ")");
        if (this.outputLayerNames.size() != 1)
            throw new InvalidKerasConfigurationException(
                    "MultiLayeNetwork expects only 1 output (found " + this.outputLayerNames.size() + ")");

        NeuralNetConfiguration.Builder modelBuilder = new NeuralNetConfiguration.Builder();
        NeuralNetConfiguration.ListBuilder listBuilder = modelBuilder.list();

        /* Add layers one at a time. */
        KerasLayer prevLayer = null;
        int layerIndex = 0;
        for (KerasLayer layer : this.layersOrdered) {
            if (layer.isLayer()) {
                int nbInbound = layer.getInboundLayerNames().size();
                if (nbInbound != 1)
                    throw new InvalidKerasConfigurationException(
                            "Layers in MultiLayerConfiguration must have exactly one inbound layer (found "
                                    + nbInbound + " for layer " + layer.getLayerName() + ")");
                if (prevLayer != null) {
                    InputType[] inputTypes = new InputType[1];
                    InputPreProcessor preprocessor;
                    if (prevLayer.isInputPreProcessor()) {
                        inputTypes[0] = this.outputTypes.get(prevLayer.getInboundLayerNames().get(0));
                        preprocessor = prevLayer.getInputPreprocessor(inputTypes);
                    } else {
                        inputTypes[0] = this.outputTypes.get(prevLayer.getLayerName());
                        preprocessor = layer.getInputPreprocessor(inputTypes);
                    }
                    if (preprocessor != null)
                        listBuilder.inputPreProcessor(layerIndex, preprocessor);
                }
                listBuilder.layer(layerIndex++, layer.getLayer());
                if (this.outputLayerNames.contains(layer.getLayerName()) && !(layer.getLayer() instanceof IOutputLayer)) {
                    // TODO: since this is always true right now, it just clutters the output.
//                    log.warn("Model cannot be trained: output layer " + layer.getLayerName()
//                            + " is not an IOutputLayer (no loss function specified)");
                }
            } else if (layer.getVertex() != null)
                throw new InvalidKerasConfigurationException("Cannot add vertex to MultiLayerConfiguration (class name "
                        + layer.getClassName() + ", layer name " + layer.getLayerName() + ")");
            prevLayer = layer;
        }

        InputType inputType = this.layersOrdered.get(0).getOutputType();
        if (inputType != null)
            listBuilder.setInputType(inputType);

        /* Whether to use standard backprop (or BPTT) or truncated BPTT. */
        if (this.useTruncatedBPTT && this.truncatedBPTT > 0)
            listBuilder.backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(truncatedBPTT)
                    .tBPTTBackwardLength(truncatedBPTT);
        else
            listBuilder.backpropType(BackpropType.Standard);
        return listBuilder.build();
    }

    /**
     * Build a MultiLayerNetwork from this Keras Sequential model configuration.
     *
     * @return MultiLayerNetwork
     */
    public MultiLayerNetwork getMultiLayerNetwork()
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return getMultiLayerNetwork(true);
    }

    /**
     * Build a MultiLayerNetwork from this Keras Sequential model configuration and import weights.
     *
     * @return MultiLayerNetwork
     */
    public MultiLayerNetwork getMultiLayerNetwork(boolean importWeights)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        MultiLayerNetwork model = new MultiLayerNetwork(getMultiLayerConfiguration());
        model.init();
        if (importWeights)
            model = (MultiLayerNetwork) KerasModelUtils.copyWeightsToModel(model, this.layers);
        return model;
    }
}
