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

import lombok.*;
import org.deeplearning4j.arbiter.layers.LayerSpace;
import org.deeplearning4j.arbiter.layers.fixed.FixedLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.TaskCreatorProvider;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.optimize.serde.jackson.JsonMapper;
import org.deeplearning4j.arbiter.optimize.serde.jackson.YamlMapper;
import org.deeplearning4j.arbiter.task.ComputationGraphTaskCreator;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.annotation.JsonTypeName;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * ComputationGraphSpace: Defines the space of valid hyperparameters for a ComputationGraph.
 * Note that this for fixed graph structures only
 *
 * @author Alex Black
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON ser/de
@Data
@EqualsAndHashCode(callSuper = true)
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "@class")
@JsonTypeName("ComputationGraphSpace")
public class ComputationGraphSpace extends BaseNetworkSpace<GraphConfiguration> {
    static {
        TaskCreatorProvider.registerDefaultTaskCreatorClass(ComputationGraphSpace.class, ComputationGraphTaskCreator.class);
    }

    @JsonProperty
    protected List<LayerConf> layerSpaces = new ArrayList<>();
    @JsonProperty
    protected List<VertexConf> vertices = new ArrayList<>();
    @JsonProperty
    protected String[] networkInputs;
    @JsonProperty
    protected String[] networkOutputs;
    @JsonProperty
    protected ParameterSpace<InputType[]> inputTypes;
    @JsonProperty
    protected int numParameters;
    @JsonProperty
    protected WorkspaceMode trainingWorkspaceMode;
    @JsonProperty
    protected WorkspaceMode inferenceWorkspaceMode;
    @JsonProperty
    protected boolean validateOutputLayerConfig = true;

    //Early stopping configuration / (fixed) number of epochs:
    protected EarlyStoppingConfiguration<ComputationGraph> earlyStoppingConfiguration;

    protected ComputationGraphSpace(Builder builder) {
        super(builder);

        this.earlyStoppingConfiguration = builder.earlyStoppingConfiguration;
        this.layerSpaces = builder.layerList;
        this.vertices = builder.vertexList;

        this.networkInputs = builder.networkInputs;
        this.networkOutputs = builder.networkOutputs;
        this.inputTypes = builder.inputTypes;
        this.trainingWorkspaceMode = builder.trainingWorkspaceMode;
        this.inferenceWorkspaceMode = builder.inferenceWorkspaceMode;
        this.validateOutputLayerConfig = builder.validateOutputLayerConfig;

        //Determine total number of parameters:
        List<ParameterSpace> list = LeafUtils.getUniqueObjects(collectLeaves());
        for (ParameterSpace ps : list)
            numParameters += ps.numParameters();
    }


    @Override
    public GraphConfiguration getValue(double[] values) {
        //Create ComputationGraphConfiguration...
        NeuralNetConfiguration.Builder builder = randomGlobalConf(values);

        ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder();
        graphBuilder.addInputs(this.networkInputs);
        graphBuilder.setOutputs(this.networkOutputs);
        if (inputTypes != null)
            graphBuilder.setInputTypes(inputTypes.getValue(values));

        //Build/add our layers and vertices:
        for (LayerConf c : layerSpaces) {
            org.deeplearning4j.nn.conf.layers.Layer l = c.layerSpace.getValue(values);
            graphBuilder.addLayer(c.getLayerName(), l, c.getPreProcessor(), c.getInputs());
        }
        for (VertexConf gv : vertices) {
            graphBuilder.addVertex(gv.getVertexName(), gv.getGraphVertex(), gv.getInputs());
        }


        if (backprop != null)
            graphBuilder.backprop(backprop.getValue(values));
        if (pretrain != null)
            graphBuilder.pretrain(pretrain.getValue(values));
        if (backpropType != null)
            graphBuilder.backpropType(backpropType.getValue(values));
        if (tbpttFwdLength != null)
            graphBuilder.tBPTTForwardLength(tbpttFwdLength.getValue(values));
        if (tbpttBwdLength != null)
            graphBuilder.tBPTTBackwardLength(tbpttBwdLength.getValue(values));
        graphBuilder.validateOutputLayerConfig(validateOutputLayerConfig);

        ComputationGraphConfiguration configuration = graphBuilder.build();

        if (trainingWorkspaceMode != null)
            configuration.setTrainingWorkspaceMode(trainingWorkspaceMode);
        if (inferenceWorkspaceMode != null)
            configuration.setInferenceWorkspaceMode(inferenceWorkspaceMode);

        return new GraphConfiguration(configuration, earlyStoppingConfiguration, numEpochs);
    }

    @Override
    public int numParameters() {
        return numParameters;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        List<ParameterSpace> list = super.collectLeaves();
        for (LayerConf lc : layerSpaces) {
            list.addAll(lc.layerSpace.collectLeaves());
        }
        if (inputTypes != null)
            list.add(inputTypes);
        return list;
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(super.toString());

        for (LayerConf conf : layerSpaces) {
            sb.append("Layer config: \"").append(conf.layerName).append("\", ").append(conf.layerSpace)
                            .append(", inputs: ").append(conf.inputs == null ? "[]" : Arrays.toString(conf.inputs))
                            .append("\n");
        }

        for (VertexConf conf : vertices) {
            sb.append("GraphVertex: \"").append(conf.vertexName).append("\", ").append(conf.graphVertex)
                            .append(", inputs: ").append(conf.inputs == null ? "[]" : Arrays.toString(conf.inputs))
                            .append("\n");
        }

        if (earlyStoppingConfiguration != null) {
            sb.append("Early stopping configuration:").append(earlyStoppingConfiguration.toString()).append("\n");
        } else {
            sb.append("Training # epochs:").append(numEpochs).append("\n");
        }

        if (inputTypes != null) {
            sb.append("Input types: ").append(inputTypes).append("\n");
        }

        return sb.toString();
    }

    @AllArgsConstructor
    @Data
    @NoArgsConstructor //For Jackson JSON
    protected static class VertexConf {
        protected GraphVertex graphVertex;
        protected String vertexName;
        protected String[] inputs;
    }

    public static class Builder extends BaseNetworkSpace.Builder<Builder> {

        protected List<LayerConf> layerList = new ArrayList<>();
        protected List<VertexConf> vertexList = new ArrayList<>();
        protected EarlyStoppingConfiguration<ComputationGraph> earlyStoppingConfiguration;
        protected String[] networkInputs;
        protected String[] networkOutputs;
        protected ParameterSpace<InputType[]> inputTypes;
        protected WorkspaceMode trainingWorkspaceMode;
        protected WorkspaceMode inferenceWorkspaceMode;

        //Need: input types
        //Early stopping configuration
        //Graph nodes

        /**
         * Early stopping configuration (optional). Note if both EarlyStoppingConfiguration and number of epochs is
         * present, early stopping will be used in preference.
         */
        public Builder earlyStoppingConfiguration(
                        EarlyStoppingConfiguration<ComputationGraph> earlyStoppingConfiguration) {
            this.earlyStoppingConfiguration = earlyStoppingConfiguration;
            return this;
        }

        public Builder layer(String layerName, LayerSpace<? extends Layer> layerSpace, String... layerInputs){
            return addLayer(layerName, layerSpace, layerInputs);
        }

        public Builder layer(String layerName, LayerSpace<? extends Layer> layerSpace, InputPreProcessor preProcessor,
                             String... layerInputs) {
            return addLayer(layerName, layerSpace, preProcessor, layerInputs);
        }

        public Builder layer(String layerName, Layer layer, String... layerInputs){
            return layer(layerName, new FixedLayerSpace<>(layer), layerInputs);
        }

        public Builder addLayer(String layerName, LayerSpace<? extends Layer> layerSpace, String... layerInputs) {
            layerList.add(new LayerConf(layerSpace, layerName, layerInputs, new FixedValue<>(1), false, null));
            return this;
        }

        public Builder addLayer(String layerName, LayerSpace<? extends Layer> layerSpace, InputPreProcessor preProcessor,
                                String... layerInputs){
            layerList.add(new LayerConf(layerSpace, layerName, layerInputs, new FixedValue<>(1), false, preProcessor));
            return this;
        }

        public Builder addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
            vertexList.add(new VertexConf(vertex, vertexName, vertexInputs));
            return this;
        }

        public Builder addInputs(String... networkInputs) {
            this.networkInputs = networkInputs;
            return this;
        }

        public Builder setOutputs(String... networkOutputs) {
            this.networkOutputs = networkOutputs;
            return this;
        }

        public Builder setInputTypes(InputType... inputTypes) {
            return setInputTypes(new FixedValue<InputType[]>(inputTypes));
        }

        public Builder setInputTypes(ParameterSpace<InputType[]> inputTypes) {
            this.inputTypes = inputTypes;
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
        public ComputationGraphSpace build() {
            return new ComputationGraphSpace(this);
        }
    }


    /**
     * Instantiate a computation graph space from
     * a raw json string
     * @param json
     * @return
     */
    public static ComputationGraphSpace fromJson(String json) {
        try {
            return JsonMapper.getMapper().readValue(json, ComputationGraphSpace.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Instantiate a computation graph space
     * from a raw yaml string
     * @param yaml
     * @return
     */
    public static ComputationGraphSpace fromYaml(String yaml) {
        try {
            return YamlMapper.getMapper().readValue(yaml, ComputationGraphSpace.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
