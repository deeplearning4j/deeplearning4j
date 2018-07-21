/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.arbiter.layers.LayerSpace;
import org.deeplearning4j.arbiter.layers.fixed.FixedLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.TaskCreatorProvider;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.optimize.serde.jackson.JsonMapper;
import org.deeplearning4j.arbiter.optimize.serde.jackson.YamlMapper;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Data
@EqualsAndHashCode(callSuper = true)
public class MultiLayerSpace extends BaseNetworkSpace<DL4JConfiguration> {

    static {
        TaskCreatorProvider.registerDefaultTaskCreatorClass(MultiLayerSpace.class, MultiLayerNetworkTaskCreator.class);
    }

    @JsonProperty
    protected ParameterSpace<InputType> inputType;
    @JsonProperty
    protected ParameterSpace<Map<Integer, InputPreProcessor>> inputPreProcessors;

    //Early stopping configuration / (fixed) number of epochs:
    @JsonProperty
    protected EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration;
    @JsonProperty
    protected int numParameters;
    @JsonProperty
    protected WorkspaceMode trainingWorkspaceMode;
    @JsonProperty
    protected WorkspaceMode inferenceWorkspaceMode;


    protected MultiLayerSpace(Builder builder) {
        super(builder);
        this.inputType = builder.inputType;
        this.inputPreProcessors = builder.inputPreProcessors;

        this.earlyStoppingConfiguration = builder.earlyStoppingConfiguration;

        this.layerSpaces = builder.layerSpaces;

        //Determine total number of parameters:
        //Collect the leaves, and make sure they are unique.
        //Note that the *object instances* must be unique - and consequently we don't want to use .equals(), as
        // this would incorrectly filter out equal range parameter spaces
        List<ParameterSpace> allLeaves = collectLeaves();
        List<ParameterSpace> list = LeafUtils.getUniqueObjects(allLeaves);

        for (ParameterSpace ps : list)
            numParameters += ps.numParameters();

        this.trainingWorkspaceMode = builder.trainingWorkspaceMode;
        this.inferenceWorkspaceMode = builder.inferenceWorkspaceMode;
    }

    protected MultiLayerSpace() {
        //Default constructor for Jackson json/yaml serialization
    }

    @Override
    public DL4JConfiguration getValue(double[] values) {
        //First: create layer configs
        List<org.deeplearning4j.nn.conf.layers.Layer> layers = new ArrayList<>();
        for (LayerConf c : layerSpaces) {
            int n = c.numLayers.getValue(values);
            if (c.duplicateConfig) {
                //Generate N identical configs
                org.deeplearning4j.nn.conf.layers.Layer l = c.layerSpace.getValue(values);
                for (int i = 0; i < n; i++) {
                    layers.add(l.clone());
                }
            } else {
                throw new UnsupportedOperationException("Not yet implemented");
            }
        }

        //Create MultiLayerConfiguration...
        NeuralNetConfiguration.Builder builder = randomGlobalConf(values);

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        for (int i = 0; i < layers.size(); i++) {
            listBuilder.layer(i, layers.get(i));
        }

        if (backprop != null)
            listBuilder.backprop(backprop.getValue(values));
        if (pretrain != null)
            listBuilder.pretrain(pretrain.getValue(values));
        if (backpropType != null)
            listBuilder.backpropType(backpropType.getValue(values));
        if (tbpttFwdLength != null)
            listBuilder.tBPTTForwardLength(tbpttFwdLength.getValue(values));
        if (tbpttBwdLength != null)
            listBuilder.tBPTTBackwardLength(tbpttBwdLength.getValue(values));
        if (inputType != null)
            listBuilder.setInputType(inputType.getValue(values));
        if (inputPreProcessors != null)
            listBuilder.setInputPreProcessors(inputPreProcessors.getValue(values));

        MultiLayerConfiguration configuration = listBuilder.build();

        if (trainingWorkspaceMode != null)
            configuration.setTrainingWorkspaceMode(trainingWorkspaceMode);
        if (inferenceWorkspaceMode != null)
            configuration.setInferenceWorkspaceMode(inferenceWorkspaceMode);

        return new DL4JConfiguration(configuration, earlyStoppingConfiguration, numEpochs);
    }

    @Override
    public int numParameters() {
        return numParameters;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        List<ParameterSpace> list = super.collectLeaves();
        for (LayerConf lc : layerSpaces) {
            list.addAll(lc.numLayers.collectLeaves());
            list.addAll(lc.layerSpace.collectLeaves());
        }
        if (inputType != null)
            list.addAll(inputType.collectLeaves());
        if (inputPreProcessors != null)
            list.addAll(inputPreProcessors.collectLeaves());
        return list;
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(super.toString());

        int i = 0;
        for (LayerConf conf : layerSpaces) {

            sb.append("Layer config ").append(i++).append(": (Number layers:").append(conf.numLayers)
                            .append(", duplicate: ").append(conf.duplicateConfig).append("), ")
                            .append(conf.layerSpace.toString()).append("\n");
        }

        if (inputType != null)
            sb.append("inputType: ").append(inputType).append("\n");
        if (inputPreProcessors != null)
            sb.append("inputPreProcessors: ").append(inputPreProcessors).append("\n");

        if (earlyStoppingConfiguration != null) {
            sb.append("Early stopping configuration:").append(earlyStoppingConfiguration.toString()).append("\n");
        } else {
            sb.append("Training # epochs:").append(numEpochs).append("\n");
        }

        return sb.toString();
    }

    public LayerSpace<?> getLayerSpace(int layerNumber) {
        return layerSpaces.get(layerNumber).getLayerSpace();
    }

    public static class Builder extends BaseNetworkSpace.Builder<Builder> {
        protected List<LayerConf> layerSpaces = new ArrayList<>();
        protected ParameterSpace<InputType> inputType;
        protected ParameterSpace<Map<Integer, InputPreProcessor>> inputPreProcessors;
        protected WorkspaceMode trainingWorkspaceMode;
        protected WorkspaceMode inferenceWorkspaceMode;

        //Early stopping configuration
        protected EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration;



        public Builder setInputType(InputType inputType) {
            return setInputType(new FixedValue<>(inputType));
        }

        public Builder setInputType(ParameterSpace<InputType> inputType) {
            this.inputType = inputType;
            return this;
        }

        public Builder layer(Layer layer){
            return layer(new FixedLayerSpace<>(layer));
        }

        public Builder layer(LayerSpace<?> layerSpace) {
            return layer(layerSpace, new FixedValue<>(1), true);
        }

        public Builder layer(LayerSpace<? extends Layer> layerSpace, ParameterSpace<Integer> numLayersDistribution,
                                boolean duplicateConfig) {
            return addLayer(layerSpace, numLayersDistribution, duplicateConfig);
        }


        public Builder addLayer(LayerSpace<?> layerSpace) {
            return addLayer(layerSpace, new FixedValue<>(1), true);
        }

        /**
         * @param layerSpace
         * @param numLayersDistribution Distribution for number of layers to generate
         * @param duplicateConfig       Only used if more than 1 layer can be generated. If true: generate N identical (stacked) layers.
         *                              If false: generate N independent layers
         */
        public Builder addLayer(LayerSpace<? extends Layer> layerSpace, ParameterSpace<Integer> numLayersDistribution,
                        boolean duplicateConfig) {
            String layerName = "layer_" + layerSpaces.size();
            layerSpaces.add(new LayerConf(layerSpace, layerName, null, numLayersDistribution, duplicateConfig, null));
            return this;
        }

        /**
         * Early stopping configuration (optional). Note if both EarlyStoppingConfiguration and number of epochs is
         * present, early stopping will be used in preference.
         */
        public Builder earlyStoppingConfiguration(
                        EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration) {
            this.earlyStoppingConfiguration = earlyStoppingConfiguration;
            return this;
        }

        /**
         * @param inputPreProcessors Input preprocessors to set for the model
         */
        public Builder setInputPreProcessors(Map<Integer, InputPreProcessor> inputPreProcessors) {
            return setInputPreProcessors(new FixedValue<>(inputPreProcessors));
        }

        /**
         * @param inputPreProcessors Input preprocessors to set for the model
         */
        public Builder setInputPreProcessors(ParameterSpace<Map<Integer, InputPreProcessor>> inputPreProcessors) {
            this.inputPreProcessors = inputPreProcessors;
            return this;
        }

        public Builder trainingWorkspaceMode(WorkspaceMode workspaceMode){
            this.trainingWorkspaceMode = workspaceMode;
            return this;
        }

        public Builder inferenceWorkspaceMode(WorkspaceMode workspaceMode){
            this.inferenceWorkspaceMode = workspaceMode;
            return this;
        }

        @SuppressWarnings("unchecked")
        public MultiLayerSpace build() {
            return new MultiLayerSpace(this);
        }
    }

    public static MultiLayerSpace fromJson(String json) {
        try {
            return JsonMapper.getMapper().readValue(json, MultiLayerSpace.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static MultiLayerSpace fromYaml(String yaml) {
        try {
            return YamlMapper.getMapper().readValue(yaml, MultiLayerSpace.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
