package org.arbiter.deeplearning4j;

import lombok.Builder;
import lombok.Data;
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
    //Min and max number of layers with this configuration
    private int minLayers;
    private int maxLayers;
    private Random r = new Random();    //TODO global rng seed etc

    public LayerSpace(Builder builder){
        this.configClass = builder.configClass;
        this.configOptions = builder.configOptions;
        this.minLayers = builder.minLayers;
        this.maxLayers = builder.maxLayers;
    }

    public List<Layer> randomLayers(){

        int nLayers = minLayers + r.nextInt(maxLayers-minLayers+1);

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
                    m = getMethodByName(builderClass,entry.getKey());
//                    m = builderClass.getDeclaredMethod(entry.getKey());
                } catch(Exception e ){
                    throw new RuntimeException("Invalid or unknown configuration option: \"" + entry.getKey() + "\"",e);
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
        private int minLayers = 1;
        private int maxLayers = 1;

        public Builder layer( Class<? extends Layer> layer ){
            this.configClass = layer;
            return this;
        }

        public Builder add(String option, ParameterSpace<?> parameterSpace ){
            configOptions.put(option, parameterSpace);
            return this;
        }

        public Builder minLayers( int minLayers ){
            this.minLayers = minLayers;
            return this;
        }

        public Builder maxLayers( int maxLayers ){
            this.maxLayers = maxLayers;
            return this;
        }

        public LayerSpace build(){
            return new LayerSpace(this);
        }
    }


    //TODO improve this
    private Method getMethodByName( Class<?> c, String name ){
        Method[] methods = c.getMethods();
        for( Method m : methods ){
            if(m.getName().equals(name)) return m;
        }
        throw new RuntimeException("Could not find method: " + name);
    }
}
