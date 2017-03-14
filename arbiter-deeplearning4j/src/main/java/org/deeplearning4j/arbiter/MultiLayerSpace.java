package org.deeplearning4j.arbiter;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.layers.LayerSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.serde.jackson.JsonMapper;
import org.deeplearning4j.arbiter.optimize.serde.jackson.YamlMapper;
import org.deeplearning4j.arbiter.util.CollectionUtils;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.annotation.JsonTypeName;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Data
public class MultiLayerSpace extends BaseNetworkSpace<DL4JConfiguration> {
    @JsonProperty
    private List<LayerConf> layerSpaces = new ArrayList<>();
    @JsonProperty
    private ParameterSpace<InputType> inputType;

    //Early stopping configuration / (fixed) number of epochs:
    @JsonProperty
    private EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration;
    @JsonProperty
    private int numParameters;

    private MultiLayerSpace(Builder builder) {
        super(builder);
        this.inputType = builder.inputType;

        this.earlyStoppingConfiguration = builder.earlyStoppingConfiguration;

        this.layerSpaces = builder.layerSpaces;

        //Determine total number of parameters:
        List<ParameterSpace> list = CollectionUtils.getUnique(collectLeaves());
        for (ParameterSpace ps : list) numParameters += ps.numParameters();

        //TODO inputs
    }

    private MultiLayerSpace(){
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


        //Set nIn based on nOut of previous layer.
        //TODO This won't work for all cases (at minimum: cast is an issue)
        int lastNOut = ((FeedForwardLayer) layers.get(0)).getNOut();
        for (int i = 1; i < layers.size(); i++) {
            if(!(layers.get(i) instanceof FeedForwardLayer)) continue;
            FeedForwardLayer ffl = (FeedForwardLayer) layers.get(i);
            ffl.setNIn(lastNOut);
            lastNOut = ffl.getNOut();
        }

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        for (int i = 0; i < layers.size(); i++) {
            listBuilder.layer(i, layers.get(i));
        }

        if (backprop != null) listBuilder.backprop(backprop.getValue(values));
        if (pretrain != null) listBuilder.pretrain(pretrain.getValue(values));
        if (backpropType != null) listBuilder.backpropType(backpropType.getValue(values));
        if (tbpttFwdLength != null) listBuilder.tBPTTForwardLength(tbpttFwdLength.getValue(values));
        if (tbpttBwdLength != null) listBuilder.tBPTTBackwardLength(tbpttBwdLength.getValue(values));
        if (cnnInputSize != null) listBuilder.cnnInputSize(cnnInputSize.getValue(values));
        if (inputType != null) listBuilder.setInputType(inputType.getValue(values));

        MultiLayerConfiguration configuration = listBuilder.build();
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
        if (cnnInputSize != null) list.addAll(cnnInputSize.collectLeaves());
        if (inputType != null) list.addAll(inputType.collectLeaves());
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

        if (cnnInputSize != null) sb.append("cnnInputSize: ").append(cnnInputSize).append("\n");
        if (inputType != null) sb.append("inputType: ").append(inputType).append("\n");

        if (earlyStoppingConfiguration != null) {
            sb.append("Early stopping configuration:").append(earlyStoppingConfiguration.toString()).append("\n");
        } else {
            sb.append("Training # epochs:").append(numEpochs).append("\n");
        }

        return sb.toString();
    }

    public LayerSpace<?> getLayerSpace(int layerNumber){
        return layerSpaces.get(layerNumber).getLayerSpace();
    }


    @AllArgsConstructor @NoArgsConstructor @Data    //No-arg for Jackson JSON serialization
    private static class LayerConf {
        private LayerSpace<?> layerSpace;
        private ParameterSpace<Integer> numLayers;
        private boolean duplicateConfig;
    }

    public static class Builder extends BaseNetworkSpace.Builder<Builder> {
        private List<LayerConf> layerSpaces = new ArrayList<>();
        private ParameterSpace<InputType> inputType;

        //Early stopping configuration
        private EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration;



        public Builder setInputType(InputType inputType) {
            return setInputType(new FixedValue<>(inputType));
        }

        public Builder setInputType(ParameterSpace<InputType> inputType) {
            this.inputType = inputType;
            return this;
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
        public Builder addLayer(LayerSpace<? extends org.deeplearning4j.nn.conf.layers.Layer> layerSpace,
                                ParameterSpace<Integer> numLayersDistribution, boolean duplicateConfig) {
            layerSpaces.add(new LayerConf(layerSpace, numLayersDistribution, duplicateConfig));
            return this;
        }

        /**
         * Early stopping configuration (optional). Note if both EarlyStoppingConfiguration and number of epochs is
         * present, early stopping will be used in preference.
         */
        public Builder earlyStoppingConfiguration(EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration) {
            this.earlyStoppingConfiguration = earlyStoppingConfiguration;
            return this;
        }

        @SuppressWarnings("unchecked")
        public MultiLayerSpace build() {
            return new MultiLayerSpace(this);
        }
    }

    public static MultiLayerSpace fromJson(String json){
        try {
            return JsonMapper.getMapper().readValue(json, MultiLayerSpace.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static MultiLayerSpace fromYaml(String yaml){
        try {
            return YamlMapper.getMapper().readValue(yaml, MultiLayerSpace.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
