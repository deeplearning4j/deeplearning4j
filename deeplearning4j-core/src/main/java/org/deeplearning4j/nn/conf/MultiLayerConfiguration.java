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
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * Configuration for a multi layer network
 *
 * @author Adam Gibson
 */
public class MultiLayerConfiguration implements Serializable {

    protected int[] hiddenLayerSizes;
    protected List<NeuralNetConfiguration> confs;
    protected boolean useDropConnect = false;
    protected boolean useGaussNewtonVectorProductBackProp = false;
    protected boolean pretrain = true;
    /* Sample if true, otherwise use the straight activation function */
    protected boolean useRBMPropUpAsActivations = true;
    protected double dampingFactor = 100;
    protected Map<Integer,OutputPreProcessor> processors = new HashMap<>();
    protected Map<Integer,InputPreProcessor> inputPreProcessors = new HashMap<>();
    protected boolean backward = false;

    
    
    public MultiLayerConfiguration() {}

    public MultiLayerConfiguration(MultiLayerConfiguration multiLayerConfiguration) {
        this.hiddenLayerSizes = multiLayerConfiguration.hiddenLayerSizes;
        this.confs = new ArrayList<>(multiLayerConfiguration.confs);
        this.useDropConnect = multiLayerConfiguration.useDropConnect;
        this.useGaussNewtonVectorProductBackProp = multiLayerConfiguration.useGaussNewtonVectorProductBackProp;
        this.pretrain = multiLayerConfiguration.pretrain;
        this.useRBMPropUpAsActivations = multiLayerConfiguration.useRBMPropUpAsActivations;
        this.dampingFactor = multiLayerConfiguration.dampingFactor;
        this.processors = new HashMap<>(multiLayerConfiguration.processors);
        this.backward = multiLayerConfiguration.backward;
        this.inputPreProcessors = multiLayerConfiguration.inputPreProcessors;

    }
    
    

    public NeuralNetConfiguration getConf(int i) {
        return confs.get(i);
    }

    public OutputPreProcessor getPreProcessor(int layer) {
        return processors.get(layer);
    }

    public InputPreProcessor getInputPreProcess(int layer) {
        return inputPreProcessors.get(layer);
    }

    public double getDampingFactor() {
        return dampingFactor;
    }

    public void setDampingFactor(double dampingFactor) {
        this.dampingFactor = dampingFactor;
    }

    public boolean isUseRBMPropUpAsActivations() {
        return useRBMPropUpAsActivations;
    }

    public void setUseRBMPropUpAsActivations(boolean useRBMPropUpAsActivations) {
        this.useRBMPropUpAsActivations = useRBMPropUpAsActivations;
    }

    public boolean isUseDropConnect() {
        return useDropConnect;
    }

    public void setUseDropConnect(boolean useDropConnect) {
        this.useDropConnect = useDropConnect;
    }

    public boolean isUseGaussNewtonVectorProductBackProp() {
        return useGaussNewtonVectorProductBackProp;
    }

    public void setUseGaussNewtonVectorProductBackProp(boolean useGaussNewtonVectorProductBackProp) {
        this.useGaussNewtonVectorProductBackProp = useGaussNewtonVectorProductBackProp;
    }

    public boolean isPretrain() {
        return pretrain;
    }

    public void setPretrain(boolean pretrain) {
        this.pretrain = pretrain;
    }

    public int[] getHiddenLayerSizes() {
        return hiddenLayerSizes;
    }

    public void setHiddenLayerSizes(int[] hiddenLayerSizes) {
        this.hiddenLayerSizes = hiddenLayerSizes;
    }

    public List<NeuralNetConfiguration> getConfs() {
        return confs;
    }

    public void setConfs(List<NeuralNetConfiguration> confs) {
        this.confs = confs;
    }

    public Map<Integer, OutputPreProcessor> getProcessors() {
        return processors;
    }

    public void setProcessors(Map<Integer, OutputPreProcessor> processors) {
        this.processors = processors;
    }

    public boolean isBackward() {
        return backward;
    }

    public void setBackward(boolean backward) {
        this.backward = backward;
    }

    public Map<Integer, InputPreProcessor> getInputPreProcessors() {
        return inputPreProcessors;
    }

    public void setInputPreProcessors(Map<Integer, InputPreProcessor> inputPreProcessors) {
        this.inputPreProcessors = inputPreProcessors;
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
        return "MultiLayerConfiguration{" +
                "hiddenLayerSizes=" + Arrays.toString(hiddenLayerSizes) +
                ", confs=" + confs +
                ", useDropConnect=" + useDropConnect +
                ", useGaussNewtonVectorProductBackProp=" + useGaussNewtonVectorProductBackProp +
                ", pretrain=" + pretrain +
                ", useRBMPropUpAsActivations=" + useRBMPropUpAsActivations +
                ", dampingFactor=" + dampingFactor +
                ", processors=" + processors +
                ", backward=" + backward +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof MultiLayerConfiguration)) return false;

        MultiLayerConfiguration that = (MultiLayerConfiguration) o;

        if (backward != that.backward) return false;
        if (Double.compare(that.dampingFactor, dampingFactor) != 0) return false;
        if (pretrain != that.pretrain) return false;
        if (useDropConnect != that.useDropConnect) return false;
        if (useGaussNewtonVectorProductBackProp != that.useGaussNewtonVectorProductBackProp) return false;
        if (useRBMPropUpAsActivations != that.useRBMPropUpAsActivations) return false;
        if (confs != null ? !confs.equals(that.confs) : that.confs != null) return false;
        return Arrays.equals(hiddenLayerSizes, that.hiddenLayerSizes);

    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = hiddenLayerSizes != null ? Arrays.hashCode(hiddenLayerSizes) : 0;
        result = 31 * result + (confs != null ? confs.hashCode() : 0);
        result = 31 * result + (useDropConnect ? 1 : 0);
        result = 31 * result + (useGaussNewtonVectorProductBackProp ? 1 : 0);
        result = 31 * result + (pretrain ? 1 : 0);
        result = 31 * result + (useRBMPropUpAsActivations ? 1 : 0);
        temp = Double.doubleToLongBits(dampingFactor);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + (processors != null ? processors.hashCode() : 0);
        result = 31 * result + (backward ? 1 : 0);
        return result;
    }

    @Override
    public MultiLayerConfiguration clone() {
        return new MultiLayerConfiguration(this);
    }

    public static class Builder {

        protected List<NeuralNetConfiguration> confs = new ArrayList<>();
        protected int[] hiddenLayerSizes;
        protected boolean useDropConnect = false;
        protected boolean pretrain = true;
        protected boolean useRBMPropUpAsActivations = false;
        protected double dampingFactor = 100;
        protected Map<Integer,OutputPreProcessor> preProcessors = new HashMap<>();
        protected Map<Integer,InputPreProcessor> inputPreProcessor = new HashMap<>();
        protected boolean backward = false;
        protected Map<Integer,ConfOverride> confOverrides = new HashMap<>();


        /**
         * Specify the input pre processors.
         * These are used at each layer for doing things like normalization and
         * shaping of input.
         * @param inputPreProcessor the input pre processor to use.
         * @return builder pattern
         */
        public Builder inputPreProcessors(Map<Integer,InputPreProcessor> inputPreProcessor) {
            this.inputPreProcessor = inputPreProcessor;
            return this;
        }

        /**
         * Whether to do back prop or not
         * @param backward whether to do back prop or not
         * @return
         */
        public Builder backward(boolean backward) {
            this.backward = backward;
            return this;
        }

        public Builder inputPreProcessor(Integer layer,InputPreProcessor preProcessor) {
            inputPreProcessor.put(layer,preProcessor);
            return this;
        }
        
        public Builder preProcessor(Integer layer,OutputPreProcessor preProcessor) {
            preProcessors.put(layer,preProcessor);
            return this;
        }

        public Builder preProcessors(Map<Integer,OutputPreProcessor> preProcessors) {
            this.preProcessors = preProcessors;
            return this;
        }

        public Builder dampingFactor(double dampingFactor) {
            this.dampingFactor = dampingFactor;
            return this;
        }

        public Builder useRBMPropUpAsActivations(boolean useRBMPropUpAsActivations) {
            this.useRBMPropUpAsActivations = useRBMPropUpAsActivations;
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

        /**
         * Whether to use drop connect or not
         * @param useDropConnect true if drop connect
         *                       should applied or not
         * @return builder pattern
         */
        public Builder useDropConnect(boolean useDropConnect) {
            this.useDropConnect = useDropConnect;
            return this;
        }



        public Builder confs(List<NeuralNetConfiguration> confs) {
            this.confs = confs;
            return this;

        }

        /**
         * Specify the hidden layer sizes.
         * Note that you can specify the layer sizes in continuous order.
         * Whereever number of inputs and outputs are used
         * this will be set for intermediate layers
         * @param hiddenLayerSizes the hidden layer sizes to use
         * @return
         */
        public Builder hiddenLayerSizes(int...hiddenLayerSizes) {
            this.hiddenLayerSizes = hiddenLayerSizes;
            return this;
        }

        public MultiLayerConfiguration build() {
            MultiLayerConfiguration conf = new MultiLayerConfiguration();
            conf.confs = this.confs;
            if(hiddenLayerSizes == null)
                throw new IllegalStateException("Please specify hidden layer sizes");
            conf.hiddenLayerSizes = this.hiddenLayerSizes;
            conf.useDropConnect = useDropConnect;
            conf.pretrain = pretrain;
            conf.useRBMPropUpAsActivations = useRBMPropUpAsActivations;
            conf.dampingFactor = dampingFactor;
            conf.processors = preProcessors;
            conf.backward = backward;
            conf.inputPreProcessors = inputPreProcessor;
            return conf;

        }

        @Override
        public String toString() {
            return "Builder{" +
                    "confs=" + confs +
                    ", hiddenLayerSizes=" + Arrays.toString(hiddenLayerSizes) +
                    ", useDropConnect=" + useDropConnect +
                    ", pretrain=" + pretrain +
                    ", useRBMPropUpAsActivations=" + useRBMPropUpAsActivations +
                    ", dampingFactor=" + dampingFactor +
                    ", preProcessors=" + preProcessors +
                    '}';
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Builder)) return false;

            Builder builder = (Builder) o;

            return Double.compare(builder.dampingFactor, dampingFactor) == 0
                && pretrain == builder.pretrain && useDropConnect == builder.useDropConnect
                && useRBMPropUpAsActivations == builder.useRBMPropUpAsActivations
                && !(confs != null ? !confs.equals(builder.confs) : builder.confs != null)
                && Arrays.equals(hiddenLayerSizes, builder.hiddenLayerSizes);

        }

        @Override
        public int hashCode() {
            int result;
            long temp;
            result = confs != null ? confs.hashCode() : 0;
            result = 31 * result + (hiddenLayerSizes != null ? Arrays.hashCode(hiddenLayerSizes) : 0);
            result = 31 * result + (useDropConnect ? 1 : 0);
            result = 31 * result + (pretrain ? 1 : 0);
            result = 31 * result + (useRBMPropUpAsActivations ? 1 : 0);
            temp = Double.doubleToLongBits(dampingFactor);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            result = 31 * result + (preProcessors != null ? preProcessors.hashCode() : 0);
            return result;
        }

        public Builder override(ConfOverride override) {
            confOverrides.put(confOverrides.size(),override);
            return this;
        }

        public Builder override(int layer,ConfOverride override) {
            confOverrides.put(layer,override);
            return this;
        }


    }




}
