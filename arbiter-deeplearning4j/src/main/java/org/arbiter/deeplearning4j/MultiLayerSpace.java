package org.arbiter.deeplearning4j;

import org.arbiter.optimize.api.ModelParameterSpace;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.lang.reflect.Method;
import java.util.*;

/**Hyperparameter space for DL4J MultiLayerNetworks
 */
public class MultiLayerSpace implements ModelParameterSpace<MultiLayerConfiguration> {

    private static final Class<NeuralNetConfiguration.Builder> builderClass = NeuralNetConfiguration.Builder.class;
    private static final Class<NeuralNetConfiguration.ListBuilder> listBuilderClass = NeuralNetConfiguration.ListBuilder.class;

    private Map<String,ParameterSpace<?>> globalConfig;
    private List<LayerSpace> layerSpaces;
    private Random r = new Random();    //TODO global seed etc


    public MultiLayerSpace(Builder builder){
        this.globalConfig = builder.globalConfig;
        this.layerSpaces = builder.layerSpaces;
    }

    @Override
    public MultiLayerConfiguration randomCandidate() {


        //First: create the layer configs
        if(layerSpaces == null || layerSpaces.size() == 0)
            throw new UnsupportedOperationException("Cannot create MultiLayerNetwork with zero layers (no LayerSpaces defined)");

        List<Layer> layers = new ArrayList<>();
        for(LayerSpace ls : layerSpaces){
            layers.addAll(ls.randomLayers());
        }

        //Complication: some hyperpameters may be on Builder, others may be on ListBuilder...
        //For now: try builder; if that fails,
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        List<Map.Entry<String,ParameterSpace<?>>> notFoundInBuilder = new ArrayList<>();
        for (Map.Entry<String, ParameterSpace<?>> entry : globalConfig.entrySet()) {

            Method m;
            try{
                m = ReflectUtils.getMethodByName(builderClass,entry.getKey());
            } catch(Exception e ){
                notFoundInBuilder.add(entry);
                continue;
            }

            //TODO make this less brittle...
            Object randomValue = entry.getValue().randomValue();
            try{
                m.invoke(builder,randomValue);
            } catch(Exception e ){
                throw new RuntimeException("Could not set configuration option: \"" + entry.getKey() + "\" with value "
                        + randomValue + " with class " + randomValue.getClass());
            }

        }

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list(layers.size());
        for(Map.Entry<String,ParameterSpace<?>> entry : notFoundInBuilder ){

            Method m;
            try{
                m = ReflectUtils.getMethodByName(listBuilderClass,entry.getKey());
            } catch(Exception e ){
                notFoundInBuilder.add(entry);
                continue;
            }

            //TODO make this less brittle...
            Object randomValue = entry.getValue().randomValue();
            try{
                m.invoke(listBuilder,randomValue);
            } catch(Exception e ){
                throw new RuntimeException("Could not set configuration option: \"" + entry.getKey() + "\" with value "
                        + randomValue + " with class \"" + randomValue.getClass() + "\"; method: " + m
                        , e);
            }
        }

        int layerCount = 0;
        for(Layer l : layers){
            listBuilder.layer(layerCount++, l);
        }

        //Next: handle nIn based on nOut of previous layer...
        //TEST only. This won't work in general - i.e., CNN layers etc.
        int lastLayerNOut = ((FeedForwardLayer)layers.get(0)).getNOut();
        for( int i=1; i<layerCount; i++ ){
            FeedForwardLayer ffl = (FeedForwardLayer) layers.get(i);
            ffl.setNIn(lastLayerNOut);
            lastLayerNOut = ffl.getNOut();
        }


        return listBuilder.build();
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
