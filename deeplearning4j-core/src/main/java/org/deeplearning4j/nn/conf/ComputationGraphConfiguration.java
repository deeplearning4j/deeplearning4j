/*
 *
 *  * Copyright 2016 Skymind,Inc.
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

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** ComputationGraphConfiguration is a configuration object for neural networks with arbitrary connection structure.
 * It is analogous to {@link MultiLayerConfiguration}, but allows considerably greater flexibility for the network
 * architecture
 */
@Data
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor
public class ComputationGraphConfiguration implements Serializable, Cloneable {


    /** Map between layer numbers, and layer names */
    protected Map<Integer,String> layerNamesMap;
    protected boolean pretrain = true;
    protected boolean backprop = false;
    protected Map<Integer,InputPreProcessor> inputPreProcessors = new HashMap<>();
    protected BackpropType backpropType = BackpropType.Standard;
    protected int tbpttFwdLength = 20;
    protected int tbpttBackLength = 20;
    //whether to redistribute params or not
    protected boolean redistributeParams = false;


    /**
     * @return  JSON representation of configuration
     */
    public String toYaml() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration}
     */
    public static ComputationGraphConfiguration fromYaml(String json) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /**
     * @return  JSON representation of computation graph configuration
     */
    public String toJson() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /**
     * Create a computation graph configuration from json
     * @param json the neural net configuration from json
     * @return {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration}
     */
    public static MultiLayerConfiguration fromJson(String json) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public String toString() {
        return toJson();
    }

    @Override
    public ComputationGraphConfiguration clone(){
        throw new UnsupportedOperationException("Not implemnted");
    }


    /** Check the configuration, make sure it is valid */
    public boolean validate(){

        //Check: at least one input, at least one output

        //Check: each layer has at least one input

        //Check: no graph cycles


        throw new UnsupportedOperationException("Not implemented");
    }

    @Data
    public static class GraphBuilder {
        /** Map between layer numbers, and layer names */
        protected Map<Integer,String> layerNamesMap;

        protected Map<Integer,String> inputs = new HashMap<>();

        protected boolean pretrain = true;
        protected boolean backprop = false;
        protected BackpropType backpropType = BackpropType.Standard;
        protected int tbpttFwdLength = 20;
        protected int tbpttBackLength = 20;

        protected Map<Integer,InputPreProcessor> inputPreProcessors = new HashMap<>();
        //whether to redistribute params or not
        protected boolean redistributeParams = false;


        /**
         * Whether to redistribute parameters as a view or not
         * @param redistributeParams whether to redistribute parameters
         *                           as a view or not
         * @return
         */
        public GraphBuilder redistributeParams(boolean redistributeParams) {
            this.redistributeParams = redistributeParams;
            return this;
        }

        /**
         * Specify the processors.
         * These are used at each layer for doing things like normalization and
         * shaping of input.
         * @param processor what to use to preProcess the data.
         * @return builder pattern
         */
        public GraphBuilder inputPreProcessor(Integer layer, InputPreProcessor processor) {
            inputPreProcessors.put(layer,processor);
            return this;
        }

        public GraphBuilder inputPreProcessors(Map<Integer,InputPreProcessor> processors) {
            this.inputPreProcessors = processors;
            return this;
        }

        /**
         * Whether to do back prop or not
         * @param backprop whether to do back prop or not
         * @return
         */
        public GraphBuilder backprop(boolean backprop) {
            this.backprop = backprop;
            return this;
        }

        /**Whether to do pre train or not
         * @param pretrain whether to do pre train or not
         * @return builder pattern
         */
        public GraphBuilder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        /**The type of backprop. Default setting is used for most networks (MLP, CNN etc),
         * but optionally truncated BPTT can be used for training recurrent neural networks.
         * If using TruncatedBPTT make sure you set both tBPTTForwardLength() and tBPTTBackwardLength()
         */
        public GraphBuilder backpropType(BackpropType type){
            this.backpropType = type;
            return this;
        }

        /**When doing truncated BPTT: how many steps of forward pass should we do
         * before doing (truncated) backprop?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * Typically tBPTTForwardLength parameter is same as the the tBPTTBackwardLength parameter,
         * but may be larger than it in some circumstances (but never smaller)<br>
         * Ideally your training data time series length should be divisible by this
         * This is the k1 parameter on pg23 of
         * http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
         * @param forwardLength Forward length > 0, >= backwardLength
         */
        public GraphBuilder tBPTTForwardLength(int forwardLength){
            this.tbpttFwdLength = forwardLength;
            return this;
        }

        /**When doing truncated BPTT: how many steps of backward should we do?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * This is the k2 parameter on pg23 of
         * http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
         * @param backwardLength <= forwardLength
         */
        public GraphBuilder tBPTTBackwardLength(int backwardLength){
            this.tbpttBackLength = backwardLength;
            return this;
        }

        public GraphBuilder addLayer(int layerNumber, Layer layer, int... layerInputs ){

            throw new UnsupportedOperationException("Not implemented");
        }

        //TODO: code duplication? Layer has a name field...
        public GraphBuilder addLayer(String layerName, Layer layer, String... layerInputs ){

            throw new UnsupportedOperationException("Not implemented");
        }

        //TODO: graph networks can have multiple inputs...
        public GraphBuilder addInput(int inputNumber) {
            return addInput(inputNumber, "input_"+inputNumber);
        }

        public GraphBuilder addInput(int inputNumber, String inputName ){
            inputs.put(inputNumber,inputName);
            return this;
        }


        public ComputationGraphConfiguration build(){

            ComputationGraphConfiguration conf = new ComputationGraphConfiguration();
            conf.backprop = backprop;
            conf.pretrain = pretrain;
            conf.backpropType = backpropType;
            conf.tbpttBackLength = tbpttBackLength;
            conf.tbpttFwdLength = tbpttFwdLength;


            if(!conf.validate()){
                //TODO: provide details
                throw new IllegalStateException("Invalid configuration");
            }
            return conf;
        }
    }

}
