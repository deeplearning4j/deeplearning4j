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
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

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
        this(modelBuilder.modelJson, modelBuilder.modelYaml, modelBuilder.trainingJson, modelBuilder.weights, modelBuilder.train);
    }

    /**
     * (Not recommended) Constructor for Sequential model from model configuration
     * (JSON or YAML), training configuration (JSON), weights, and "training mode"
     * boolean indicator. When built in training mode, certain unsupported configurations
     * (e.g., unknown regularizers) will throw Exceptions. When train=false, these
     * will generate warnings but will be otherwise ignored.
     *
     * @param modelJson       model configuration JSON string
     * @param modelYaml       model configuration YAML string
     * @param trainingJson    training configuration JSON string
     * @param weights         map from layer to parameter to weights
     * @throws IOException
     */
    public KerasSequentialModel(String modelJson, String modelYaml, String trainingJson, Map<String,
                                Map<String,INDArray>> weights, boolean train)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String,Object> classNameAndLayers;
        if (modelJson != null)
            classNameAndLayers = parseJsonString(modelJson);
        else if (modelYaml != null)
            classNameAndLayers = parseYamlString(modelYaml);
        else
            throw new InvalidKerasConfigurationException("Requires model configuration as either JSON or YAML string.");

        this.className = (String) checkAndGetModelField(classNameAndLayers, MODEL_FIELD_CLASS_NAME);
        if (!this.className.equals(MODEL_CLASS_NAME_SEQUENTIAL))
            throw new InvalidKerasConfigurationException("Model class name must be " + MODEL_CLASS_NAME_SEQUENTIAL + " (found " + this.className + ")");
        this.train = train;

        /* Convert layer configuration objects into KerasLayers. */
        helperPrepareLayers((List<Object>) checkAndGetModelField(classNameAndLayers, MODEL_FIELD_CONFIG));

        /* Add placeholder input layer and update lists of input and output layers. */
        int[] inputShape = this.layers.get(this.layerNamesOrdered.get(0)).getInputShape();
        KerasLayer inputLayer = KerasLayer.createInputLayer("input1", inputShape);
        this.layers.put(inputLayer.getName(), inputLayer);
        this.inputLayerNames = new ArrayList<String>(Arrays.asList(inputLayer.getName()));
        this.outputLayerNames = new ArrayList<String>(Arrays.asList(this.layerNamesOrdered.get(this.layerNamesOrdered.size()-1)));
        this.layerNamesOrdered.add(0, inputLayer.getName());

        /* Update each layer's inbound layer list to include (only) previous layer. */
        String prevLayerName = null;
        for (String layerName : this.layerNamesOrdered) {
            if (prevLayerName != null)
                this.layers.get(layerName).setInboundLayerNames(Arrays.asList(prevLayerName));
            prevLayerName = layerName;
        }

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
        int layerIndex = 0;
        for (String layerName : this.layerNamesOrdered) {
            KerasLayer layer = this.layers.get(layerName);
            if (layer.isDl4jLayer()) { // Ignore "preprocessor" layers for now
                listBuilder.layer(layerIndex++, layer.getDl4jLayer());
            }
        }

        InputType inputType = inferInputType(this.inputLayerNames.get(0));
        if (inputType != null)
            listBuilder.setInputType(inputType);

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
            listBuilder.tBPTTForwardLength(truncatedBPTT).tBPTTBackwardLength(truncatedBPTT);
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
            model = (MultiLayerNetwork)copyWeightsToModel(model, this.weights, this.layers);
        return model;
    }
}
