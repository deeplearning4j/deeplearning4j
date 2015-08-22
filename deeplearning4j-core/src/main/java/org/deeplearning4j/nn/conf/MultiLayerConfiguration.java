/*
 *
 *  * Copyright 2015 Skymind,Inc.
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



package org.deeplearning4j.nn.conf;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.Serializable;
import java.security.Key;
import java.util.*;

/**
 * Configuration for a multi layer network
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor
public class MultiLayerConfiguration implements Serializable, Cloneable {

    protected List<NeuralNetConfiguration> confs;
    @Deprecated
    protected boolean useGaussNewtonVectorProductBackProp = false;
    protected boolean pretrain = true;
    @Deprecated
    protected double dampingFactor = 100;
    @Deprecated
    protected Map<Integer,OutputPostProcessor> outputPostProcessors = new HashMap<>();
    protected Map<Integer,InputPreProcessor> inputPreProcessors = new HashMap<>();
    @Deprecated
    protected boolean backward = false;
    protected boolean backprop = false;

    /**
     *
     * @return  JSON representation of NN configuration
     */
    public String toYaml() {
        ObjectMapper mapper = NeuralNetConfiguration.mapperYaml();
        try {
            return mapper.writeValueAsString(this);
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration}
     */
    public static MultiLayerConfiguration fromYaml(String json) {
        ObjectMapper mapper = NeuralNetConfiguration.mapperYaml();
        try {
            return mapper.readValue(json, MultiLayerConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }



    /**
     *
     * @return  JSON representation of NN configuration
     */
    public String toJson() {
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        try {
            return mapper.writeValueAsString(this);
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration}
     */
    public static MultiLayerConfiguration fromJson(String json) {
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        try {
            return mapper.readValue(json, MultiLayerConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return toJson();
    }



    public NeuralNetConfiguration getConf(int i) {
        return confs.get(i);
    }


    @Override
    public MultiLayerConfiguration clone() {
        try {
            MultiLayerConfiguration clone = (MultiLayerConfiguration) super.clone();

            if(clone.confs != null) {
                List<NeuralNetConfiguration> list = new ArrayList<>();
                for(NeuralNetConfiguration conf : clone.confs) {
                    list.add(conf.clone());
                }
                clone.confs = list;
            }

            if(clone.inputPreProcessors != null) {
                Map<Integer,InputPreProcessor> map = new HashMap<>();
                for(Map.Entry<Integer,InputPreProcessor> entry : clone.inputPreProcessors.entrySet()) {
                    map.put(entry.getKey(), entry.getValue().clone());
                }
                clone.inputPreProcessors = map;
            }

            if(clone.outputPostProcessors != null) {
                // TODO: deep clone of outputPostProcessor
                clone.outputPostProcessors = new HashMap<>(clone.outputPostProcessors);
            }

            return clone;

        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public InputPreProcessor getInputPreProcess(int curr) {
        return inputPreProcessors.get(curr);
    }

    public OutputPostProcessor getOutputPostProcess(int curr) {
        return outputPostProcessors.get(curr);
    }

    public static class Builder {

        protected List<NeuralNetConfiguration> confs = new ArrayList<>();
        protected boolean pretrain = true;
        protected double dampingFactor = 100;
        protected Map<Integer,OutputPostProcessor> outputPostProcessors = new HashMap<>();
        protected Map<Integer,InputPreProcessor> inputPreProcessors = new HashMap<>();
        protected boolean backward = false;
        protected boolean backprop = false;
        @Deprecated
        protected Map<Integer,ConfOverride> confOverrides = new HashMap<>();


        /**
         * Specify the processors.
         * These are used at each layer for doing things like normalization and
         * shaping of input.
         * @param processor what to use to preProcess the data.
         * @return builder pattern
         */
        public Builder inputPreProcessor(Integer layer, InputPreProcessor processor) {
            inputPreProcessors.put(layer,processor);
            return this;
        }

        public Builder inputPreProcessors(Map<Integer,InputPreProcessor> processors) {
            this.inputPreProcessors = processors;
            return this;
        }

        @Deprecated
        public Builder outputPostProcessor(Integer layer, OutputPostProcessor processor) {
            outputPostProcessors.put(layer, processor);
            return this;
        }

        @Deprecated
        public Builder outputPostProcessors(Map<Integer, OutputPostProcessor> processors) {
            this.outputPostProcessors = processors;
            return this;
        }

        /**
         * Whether to do back prop or not
         * @param backprop whether to do back prop or not
         * @return
         */
        public Builder backprop(boolean backprop) {
            this.backprop = backprop;
            return this;
        }

        @Deprecated
        public Builder dampingFactor(double dampingFactor) {
            this.dampingFactor = dampingFactor;
            return this;
        }
        
        /**
         * Whether to do pre train or not
         * @param pretrain whether to do pre train or not
         * @return builder pattern
         */
        public Builder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        public Builder confs(List<NeuralNetConfiguration> confs) {
            this.confs = confs;
            return this;

        }

        public MultiLayerConfiguration build() {
            MultiLayerConfiguration conf = new MultiLayerConfiguration();
            conf.confs = this.confs;
            conf.pretrain = pretrain;
            conf.dampingFactor = dampingFactor;
            conf.outputPostProcessors = outputPostProcessors;
            conf.backward = backward;
            conf.backprop = backprop;
            conf.inputPreProcessors = inputPreProcessors;
            Nd4j.getRandom().setSeed(conf.getConf(0).getSeed());
            return conf;

        }

        @Override
        public String toString() {
            return "Builder{" +
                    "confs=" + confs +
                    ", pretrain=" + pretrain +
                    ", dampingFactor=" + dampingFactor +
                    ", preProcessors=" + outputPostProcessors +
                    '}';
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Builder)) return false;

            Builder builder = (Builder) o;

            return Double.compare(builder.dampingFactor, dampingFactor) == 0
                    && pretrain == builder.pretrain
                    && !(confs != null ? !confs.equals(builder.confs) : builder.confs != null);

        }

        @Override
        public int hashCode() {
            int result;
            long temp;
            result = confs != null ? confs.hashCode() : 0;
            result = 31 * result + (pretrain ? 1 : 0);
            temp = Double.doubleToLongBits(dampingFactor);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            result = 31 * result + (outputPostProcessors != null ? outputPostProcessors.hashCode() : 0);
            return result;
        }

        @Deprecated
        public Builder override(ConfOverride override) {
            confOverrides.put(confOverrides.size(),override);
            return this;
        }

        @Deprecated
        public Builder override(int layer,ConfOverride override) {
            confOverrides.put(layer,override);
            return this;
        }

    }
}
