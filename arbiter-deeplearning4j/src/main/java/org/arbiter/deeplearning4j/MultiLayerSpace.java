package org.arbiter.deeplearning4j;

import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.*;

/**Hyperparameter space for DL4J MultiLayerNetworks
 */
public class MultiLayerSpace implements ParameterSpace<MultiLayerConfiguration> {

    private Map<String,ParameterSpace<?>> globalConfig;
    private List<LayerSpace> layerSpaces;
    private Random r = new Random();    //TODO global seed etc


    public MultiLayerSpace(Builder builder){
        this.globalConfig = builder.globalConfig;
        this.layerSpaces = builder.layerSpaces;
    }

    @Override
    public MultiLayerConfiguration randomValue() {


        //First: create the layer configs
        if(layerSpaces == null || layerSpaces.size() == 0)
            throw new UnsupportedOperationException("Cannot create MultiLayerNetwork with zero layers (no LayerSpaces defined)");




        return null;
    }

    public static class Builder {

        private Map<String,ParameterSpace<?>> globalConfig = new HashMap<>();
        private List<LayerSpace> layerSpaces = new ArrayList<>();


        public Builder add(String configOption, ParameterSpace<?> space ){
            globalConfig.put(configOption,space);
            return this;
        }

        public Builder addLayer( LayerSpace layerSpace ){
            layerSpaces.add(layerSpace);
            return this;
        }

        public MultiLayerSpace build(){
            return new MultiLayerSpace(this);
        }


    }
}
