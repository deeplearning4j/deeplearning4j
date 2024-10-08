/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.layers.KerasInput;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;

import java.io.IOException;
import java.util.*;

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

        // Prior to Keras 2.2.3 the "config" of a Sequential model was a list of layer configurations. For consistency
        // "config" is now an object containing a "name" and "layers", the latter contain the same data as before.
        // This change only affects Sequential models.
        List<Object> layerList;
        if(modelConfig.get(config.getModelFieldConfig()) instanceof List) {
            layerList = (List<Object>) modelConfig.get(config.getModelFieldConfig());
        } else {
            HashMap layerMap = (HashMap<String, Object>) modelConfig.get(config.getModelFieldConfig());
            layerList =  (List<Object>) layerMap.get("layers");
        }


        Pair<Map<String, KerasLayer>, List<KerasLayer>> layerPair =
                prepareLayers(layerList);
        this.layers = layerPair.getFirst();
        this.layersOrdered = layerPair.getSecond();

        KerasLayer inputLayer;
        if (this.layersOrdered.get(0) instanceof KerasInput) {
            inputLayer = this.layersOrdered.get(0);
        } else {
            /* Add placeholder input layer and update lists of input and output layers. */
            int[] firstLayerInputShape = this.layersOrdered.get(0).getInputShape();
            Preconditions.checkState(ArrayUtil.prod(firstLayerInputShape) > 0,"Input shape must not be zero!");
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


        if(inputShape == null) {
            inputShape = layersOrdered.get(0).getInputShape();
        }

        this.outputTypes = inferOutputTypes(inputShape);

        if (weightsArchive != null)
            KerasModelUtils.importWeights(weightsArchive, weightsRoot, layers, kerasMajorVersion, kerasBackend);
    }

    /**
     * Default constructor
     */
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
                    "MultiLayerNetwork expects only 1 input (found " + this.inputLayerNames.size() + ")");
        if (this.outputLayerNames.size() != 1)
            throw new InvalidKerasConfigurationException(
                    "MultiLayerNetwork expects only 1 output (found " + this.outputLayerNames.size() + ")");

        NeuralNetConfiguration.Builder modelBuilder = new NeuralNetConfiguration.Builder();

        if (optimizer != null) {
            modelBuilder.updater(optimizer);
        }

        ListBuilder listBuilder = modelBuilder.list();
        //don't forcibly over ride for keras import
        listBuilder.overrideNinUponBuild(false);
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
                        KerasModelUtils.setDataFormatIfNeeded(preprocessor,layer);
                        InputType outputType = preprocessor.getOutputType(inputTypes[0]);
                        layer.getLayer().setNIn(outputType,listBuilder.isOverrideNinUponBuild());
                    } else {
                        inputTypes[0] = this.outputTypes.get(prevLayer.getLayerName());
                        preprocessor = layer.getInputPreprocessor(inputTypes);
                        if(preprocessor != null) {
                            InputType outputType = preprocessor.getOutputType(inputTypes[0]);
                            layer.getLayer().setNIn(outputType,listBuilder.isOverrideNinUponBuild());
                        }
                        else
                            layer.getLayer().setNIn(inputTypes[0],listBuilder.isOverrideNinUponBuild());

                        KerasModelUtils.setDataFormatIfNeeded(preprocessor,layer);

                    }
                    if (preprocessor != null)
                        listBuilder.inputPreProcessor(layerIndex, preprocessor);


                }

                listBuilder.layer(layerIndex++, layer.getLayer());
            } else if (layer.getVertex() != null)
                throw new InvalidKerasConfigurationException("Cannot add vertex to MultiLayerConfiguration (class name "
                        + layer.getClassName() + ", layer name " + layer.getLayerName() + ")");
            prevLayer = layer;
        }

        /* Whether to use standard backprop (or BPTT) or truncated BPTT. */
        if (this.useTruncatedBPTT && this.truncatedBPTT > 0)
            listBuilder.backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(truncatedBPTT)
                    .tBPTTBackwardLength(truncatedBPTT);
        else
            listBuilder.backpropType(BackpropType.Standard);

        MultiLayerConfiguration build = listBuilder.build();


        return build;
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
