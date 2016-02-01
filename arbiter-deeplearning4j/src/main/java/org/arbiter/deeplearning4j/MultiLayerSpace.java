package org.arbiter.deeplearning4j;

import lombok.AllArgsConstructor;
import org.arbiter.deeplearning4j.layers.LayerSpace;
import org.arbiter.optimize.api.ModelParameterSpace;
import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

//public class MultiLayerSpace implements ModelParameterSpace<MultiLayerConfiguration> {
public class MultiLayerSpace extends BaseNetworkSpace<DL4JConfiguration> {

    private ParameterSpace<int[]> cnnInputSize;
    private List<LayerConf> layerSpaces = new ArrayList<>();

    //Early stopping configuration / (fixed) number of epochs:
    private EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration;

    private MultiLayerSpace(Builder builder){
        super(builder);
        this.cnnInputSize = builder.cnnInputSize;

        this.earlyStoppingConfiguration = builder.earlyStoppingConfiguration;

        this.layerSpaces = builder.layerSpaces;
    }


    @Override
    public DL4JConfiguration randomCandidate() {

        //First: create layer configs
        List<org.deeplearning4j.nn.conf.layers.Layer> layers = new ArrayList<>();
        for(LayerConf c : layerSpaces){
            int n = c.numLayers.randomValue();
            if(c.duplicateConfig){
                //Generate N identical configs
                org.deeplearning4j.nn.conf.layers.Layer l = c.layerSpace.randomLayer();
                for( int i=0; i<n; i++ ){
                    layers.add(l.clone());
                }
            } else {
                //Generate N indepedent configs
                for( int i=0; i<n; i++ ){
                    layers.add(c.layerSpace.randomLayer());
                }
            }
        }

        //Create MultiLayerConfiguration...
        NeuralNetConfiguration.Builder builder = randomGlobalConf();


        //Set nIn based on nOut of previous layer.
        //TODO This won't work for all cases (at minimum: cast is an issue)
        int lastNOut = ((FeedForwardLayer)layers.get(0)).getNOut();
        for( int i=1; i<layers.size(); i++ ){
            FeedForwardLayer ffl = (FeedForwardLayer)layers.get(i);
            ffl.setNIn(lastNOut);
            lastNOut = ffl.getNOut();
        }

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        for( int i=0; i<layers.size(); i++ ){
            listBuilder.layer(i,layers.get(i));
        }

        if(backprop != null) listBuilder.backprop(backprop.randomValue());
        if(pretrain != null) listBuilder.pretrain(pretrain.randomValue());
        if(backpropType != null) listBuilder.backpropType(backpropType.randomValue());
        if(tbpttFwdLength != null) listBuilder.tBPTTForwardLength(tbpttFwdLength.randomValue());
        if(tbpttBwdLength != null) listBuilder.tBPTTBackwardLength(tbpttBwdLength.randomValue());
        if(cnnInputSize != null) listBuilder.cnnInputSize(cnnInputSize.randomValue());

        MultiLayerConfiguration configuration = listBuilder.build();
        return new DL4JConfiguration(configuration,earlyStoppingConfiguration,numEpochs);
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder(super.toString());

        int i=0;
        for(LayerConf conf : layerSpaces){

            sb.append("Layer config ").append(i++).append(": (Number layers:").append(conf.numLayers)
                    .append(", duplicate: ").append(conf.duplicateConfig).append("), ")
                    .append(conf.layerSpace.toString()).append("\n");
        }

        if(earlyStoppingConfiguration != null){
            sb.append("Early stopping configuration:").append(earlyStoppingConfiguration.toString()).append("\n");
        } else {
            sb.append("Training # epochs:").append(numEpochs).append("\n");
        }

        return sb.toString();
    }

    @AllArgsConstructor
    private static class LayerConf {
        private final LayerSpace<?> layerSpace;
        private final ParameterSpace<Integer> numLayers;
        private final boolean duplicateConfig;
    }

    public static class Builder extends BaseNetworkSpace.Builder<Builder> {

        private ParameterSpace<int[]> cnnInputSize;

        private List<LayerConf> layerSpaces = new ArrayList<>();

        //Early stopping configuration
        private EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration;


        public Builder cnnInputSize(int height, int width, int depth){
            return cnnInputSize(new FixedValue<>(new int[]{height, width, depth}));
        }

        public Builder cnnInputSize(ParameterSpace<int[]> cnnInputSize){
            this.cnnInputSize = cnnInputSize;
            return this;
        }


        public Builder addLayer(LayerSpace<?> layerSpace){
            return addLayer(layerSpace,new FixedValue<>(1),true);
        }

        /**
         * @param layerSpace
         * @param numLayersDistribution Distribution for number of layers to generate
         * @param duplicateConfig Only used if more than 1 layer can be generated. If true: generate N identical (stacked) layers.
         *                        If false: generate N independent layers
         */
        public Builder addLayer(LayerSpace<? extends org.deeplearning4j.nn.conf.layers.Layer> layerSpace,
                                ParameterSpace<Integer> numLayersDistribution, boolean duplicateConfig){
            layerSpaces.add(new LayerConf(layerSpace,numLayersDistribution,duplicateConfig));
            return this;
        }

        /** Early stopping configuration (optional). Note if both EarlyStoppingConfiguration and number of epochs is
         * present, early stopping will be used in preference.
         */
        public Builder earlyStoppingConfiguration(EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration){
            this.earlyStoppingConfiguration = earlyStoppingConfiguration;
            return this;
        }

        @SuppressWarnings("unchecked")
        public MultiLayerSpace build(){
            return new MultiLayerSpace(this);
        }
    }

}
