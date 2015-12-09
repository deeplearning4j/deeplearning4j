package org.arbiter.deeplearning4j;

import lombok.Builder;
import lombok.Data;
import org.apache.commons.math3.distribution.IntegerDistribution;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.arbiter.optimize.distribution.DegenerateIntegerDistribution;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.lang.reflect.Method;
import java.util.*;

/**LayerSpace: Defines the ranges of hyperparameters for a single DL4J layer
 * Using reflection on the Layer configuration class
 */
@Data
public class LayerSpace {

    private Class<? extends Layer> configClass;
    private Map<String,ParameterSpace<?>> configOptions = new HashMap<>();
    private IntegerDistribution numLayersDistribution;
    private Random r = new Random();    //TODO global rng seed etc

    public LayerSpace(Builder builder){
        this.configClass = builder.configClass;
        this.configOptions = builder.configOptions;
        this.numLayersDistribution = builder.numLayersDistribution;
    }

    public List<Layer> randomLayers(){

        int nLayers = numLayersDistribution.sample();

        //Get the Builder class:
        Class<?>[] classes = configClass.getDeclaredClasses();
        if(classes.length == 0) throw new RuntimeException();
        Class<?> builderClass = null;
        for(Class<?> c : classes ){
            String name = c.getName();
            if(name.toLowerCase().endsWith("builder")){
                builderClass = c;
                break;
            }
        }
        if(builderClass == null) throw new RuntimeException("Could not find Builder class");

        List<Layer> list = new ArrayList<>(nLayers);

        for( int i=0; i<nLayers; i++ ) {
            Object builder;
            try {
                builder = builderClass.newInstance();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            for (Map.Entry<String, ParameterSpace<?>> entry : configOptions.entrySet()) {
                Method m;
                try{
                    m = ReflectUtils.getMethodByName(builderClass, entry.getKey());
                } catch(Exception e ){
                    throw new RuntimeException("Invalid or unknown configuration option: \"" + entry.getKey() + "\"",e);
                }

                //TODO make this less brittle...
                Object randomValue = entry.getValue().randomValue();
                try{
                    m.invoke(builder,randomValue);
                } catch(Exception e ){
                    throw new RuntimeException("Could not set configuration option: \"" + entry.getKey() + "\" with value "
                            + randomValue + " with class " + randomValue.getClass(), e);
                }
            }

            try{
                Method build = builderClass.getMethod("build");
                Layer l = (Layer)build.invoke(builder);
                list.add(l);
            } catch(Exception e){
                throw new RuntimeException(e);
            }
        }

        return list;
    }


    public static class Builder {
        private Class<? extends Layer> configClass;
        private Map<String,ParameterSpace<?>> configOptions = new HashMap<>();
        //Default to degenerate distribution: i.e., always exactly 1 layer
        private IntegerDistribution numLayersDistribution = new DegenerateIntegerDistribution(1);

        public Builder layer( Class<? extends Layer> layer ){
            this.configClass = layer;
            return this;
        }

        public Builder add(String option, ParameterSpace<?> parameterSpace ){
            configOptions.put(option, parameterSpace);
            return this;
        }

        public Builder numLayersDistribution( IntegerDistribution distribution ){
            this.numLayersDistribution = distribution;
            return this;
        }

        public LayerSpace build(){
            return new LayerSpace(this);
        }
    }
}
